
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "dviz.hpp"
#include <iterator>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
using namespace cv;
using namespace std;
bool Mat_read_binary(cv::Mat &img_vec, string filename) //整体读出
{
  int channl(0);
  int rows(0);
  int cols(0);
  short type(0);
  short em_size(0);
  ifstream fin(filename, ios::binary);
  fin.read((char *)&channl, 1);
  fin.read((char *)&type, 1);
  fin.read((char *)&em_size, 2);
  fin.read((char *)&cols, 4);
  fin.read((char *)&rows, 4);
  printf("SAVE:cols=%d,type=%d,em_size=%d,rows=%d,channels=%d\n", cols, type, em_size, rows, channl);
  img_vec = cv::Mat(rows, cols, type);
  fin.read((char *)&img_vec.data[0], rows * cols * em_size);
  fin.close();
  return true;
}
cv::Mat getCload(cv::Mat &depth)
{
  Mat point_cloud; // = Mat::zeros(height, width, CV_32FC3);
  //point cloud 赋值，其中 fx,fy,cx,cy 为Kinect2 的内参
  double asd = 550;
  double fx = asd, fy = asd, cx = 320.0, cy = 240.0;

  for (int row = 0; row < depth.rows; row++)
    for (int col = 0; col < depth.cols; col++)
    {
      float dz = ((float)depth.at<unsigned short>(row, col)) / 1000.0;

      Vec3f vec;
      // dz ;
      vec[0] = dz * (col - cx) / fx;
      vec[1] = dz * (row - cy) / fy;
      vec[2] = dz;
      point_cloud.push_back(vec);
    }
  return point_cloud;
}

// 相机内参，相机位姿，彩图，深度图，图宽，图高，
//体素xyz  voxel_size=每个体素的长度
//trunc_margin：截断距离
//体素值，颜色，权重
void Integrate(float *cam_K, float *cam2base, uint8_t *rgb_im, float *depth_im,
               int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
               float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
               float *voxel_grid_TSDF, float *voxel_grid_color, float *voxel_grid_weight)
{
#pragma omp parallel
#pragma omp for
  for (int pt_grid_z = 0; pt_grid_z < 500; pt_grid_z++)
  {
    int zindex = pt_grid_z * (pt_grid_z + 1) * (pt_grid_z * 2 + 1) / 6;

    for (int pt_grid_y = 0; pt_grid_y < pt_grid_z + 1; pt_grid_y++)
    {
      int yindex = pt_grid_y * (pt_grid_z + 1);

      for (int pt_grid_x = 0; pt_grid_x < pt_grid_z + 1; pt_grid_x++)
      {
        // 小网格在体素中的索引
        int volume_idx = zindex + yindex + pt_grid_x; //pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
        // 小网格坐标变换为世界坐标（3D）
        float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
        float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
        float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

        // int tindex = pt_grid_z * (pt_grid_z + 1) * (pt_grid_z * 2 + 1) / 6;
        // voxel_grid_color[tindex] = 2;
        // 计算小网格（世界坐标）在相机坐标系下的坐标（归一化坐标）
        float tmp_pt[3] = {0};
        tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
        tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
        tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];

        float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];
        if (pt_cam_z <= 0)
          continue;

        float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
        float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
        //计算小网格的图像坐标
        int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
        int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
        if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
          continue;
        //获取深度值 如果小网格对应像素坐标的深度
        float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];
        if (depth_val <= 0 || depth_val > 6)
          continue;
        // voxel_grid_color[volume_idx] = 2;
        float diff = depth_val - pt_cam_z;

        if (diff <= -trunc_margin)
          continue;

        float dist = fmin(1.0f, diff / trunc_margin);
        float weight_old = voxel_grid_weight[volume_idx];                                             //旧权重
        float weight_new = weight_old + 1.0f;                                                         //新权重
        voxel_grid_weight[volume_idx] = weight_new;                                                   //更新权重
        voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new; //更新tsdf值

        // Integrate color
        // uint8_t rgb_val = rgb_im[pt_pix_y * im_width + pt_pix_x];

        // int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
        // float dist = fmin(1.0f, diff / trunc_margin);
        // float weight_old = voxel_grid_weight[volume_idx];
        // float weight_new = weight_old + 1.0f;
        // voxel_grid_weight[volume_idx] = weight_new;
        // voxel_grid_color[volume_idx] = (rgb_val * weight_new + weight_old * voxel_grid_color[volume_idx]) / weight_new;
      }
    }
  }
}

// #include <pcl/common/impl/io.h>
// Loads a binary file with depth data and generates a TSDF voxel volume (5m x 5m x 5m at 1cm resolution)
// Volume is aligned with respect to the camera coordinates of the first frame (a.k.a. base frame)
int main(int argc, char *argv[])
{

  // Location of camera intrinsic file
  std::string cam_K_file = "../data/camera-intrinsics.txt";

  // Location of folder containing RGB-D frames and camera pose files
  std::string data_path = "../data/data";
  int base_frame_idx = 0;
  int first_frame_idx = 1500;
  int num_frames = 4013;

  float cam_K[3 * 3];
  float base2world[4 * 4];
  float cam2base[4 * 4];
  float cam2world[4 * 4];
  int im_width = 640;
  int im_height = 480;
  float depth_im[im_height * im_width];
  uint8_t RGB_im[im_height * im_width];

  // Voxel grid parameters (change these to change voxel grid resolution, etc.)
  float voxel_grid_origin_x = -1.5f; // Location of voxel grid origin in base frame camera coordinates
  float voxel_grid_origin_y = -1.5f;
  float voxel_grid_origin_z = 0.5f;
  float voxel_size = 0.01f;
  float trunc_margin = voxel_size * 5;
  int voxel_grid_dim_x = 500;
  int voxel_grid_dim_y = 500;
  int voxel_grid_dim_z = 500;
  cv::Mat posss;
  Mat_read_binary(posss, "/home/u16/dataset/img/loop2_pose.bin");
  // cout << posss << endl;
  //读取相机内参
  std::vector<float> cam_K_vec = LoadMatrixFromFile(cam_K_file, 3, 3);
  std::copy(cam_K_vec.begin(), cam_K_vec.end(), cam_K);

  // 读取第一帧的位姿
  std::ostringstream base_frame_prefix;
  base_frame_prefix << std::setw(6) << std::setfill('0') << base_frame_idx;
  std::string base2world_file = data_path + "/frame-" + base_frame_prefix.str() + ".pose.txt";
  std::vector<float> base2world_vec = LoadMatrixFromFile(base2world_file, 4, 4);
  std::copy(base2world_vec.begin(), base2world_vec.end(), base2world);

  // 位姿逆矩阵
  float base2world_inv[16] = {0};
  invert_matrix(base2world, base2world_inv);

  // 初始化格子
  float *voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  float *voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  float *voxel_grid_color = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];

  //所有格子置为1
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
    voxel_grid_TSDF[i] = 1.0f;
  //权重置为0
  memset(voxel_grid_weight, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);

  char *name = new char[256];
  int idx = 1;

  viz::Viz3d window("window");
  //显示坐标系
  window.showWidget("Coordinate", viz::WCoordinateSystem());
  Vec<float, 16> po = posss.at<Vec<float, 16>>(first_frame_idx - 1, 0);
  Affine3f pose2(&po.val[0]);

  cv::Mat mpc;
  float aplen = 400 * voxel_size;
  mpc.push_back(Vec3f(0, 0, 0));
  mpc.push_back(Vec3f(aplen, aplen, aplen));
  mpc.push_back(Vec3f(-aplen, aplen, aplen));
  mpc.push_back(Vec3f(0, 0, 0));
  mpc.push_back(Vec3f(-aplen, -aplen, aplen));
  mpc.push_back(Vec3f(aplen, -aplen, aplen));

  mpc.push_back(Vec3f(0, 0, 0));
  mpc.push_back(Vec3f(aplen, aplen, aplen));
  mpc.push_back(Vec3f(aplen, -aplen, aplen));
  // mpc.push_back(Vec3f(-aplen, -aplen, aplen));
  // mpc.push_back(Vec3f(aplen, -aplen, aplen));
  // mpc.push_back(Vec3f(aplen, aplen, aplen));

  // mpc.push_back(Vec3f(-aplen, aplen, aplen));

  cv::viz::WPolyLine wpl(mpc);
  window.showWidget("mpc", wpl, pose2);
  // 重建 TSDF voxel grid
  for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; frame_idx += 10)
  {
    cv::viz::WCloudCollection cloud;
    std::ostringstream curr_frame_prefix;
    curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;
    po = posss.at<Vec<float, 16>>(frame_idx, 0);
    cout << po << endl;
     cv::Affine3f pose(&po.val[0]);

    // cout << po.val << endl;
    // 读取深度图
    string depth_path = cv::format("/home/u16/dataset/img/depth/%04d.png", frame_idx);
    cv::Mat _depth = cv::imread(depth_path, 2);
    if (_depth.empty())
    {
      string ss = "ls " + depth_path;
      system(ss.c_str());
      assert(0);
    }
    ReadDepth(depth_path, im_height, im_width, depth_im);
    // color_path = data_path + "/frame-" + curr_frame_prefix.str() + ".color.jpg";
    // ReadRGB(color_path, im_height, im_width, RGB_im);
    // 读取位姿
    // sprintf(name, "/home/u16/dataset/home/frame-%06d.color.jpg", frame_idx);
    string color_path = cv::format("/home/u16/dataset/img/rgb/%04d.png", frame_idx);
    // po.
    // std::string color_path(name);

    cv::Mat _reacolor = cv::imread(color_path, 0);
    if (_reacolor.empty())
    {
      string ss = "ls " + color_path;
      system(ss.c_str());
      assert(0);
    }
    cv::imshow("_reacolor", _reacolor);
    cv::imshow("_depth", _depth);

    cv::waitKey(1);
    float *cam2base = &po.val[0]; //(float *)po.data; //[16];
    // continue;
    // std::string cam2world_file = string(name); //data_path + "/frame-" + curr_frame_prefix.str() + ".pose.txt";
    // // std::cout<<name<<endl;
    // LoadMatrixFromFile(cam2world_file, cam2world, 4, 4);
    // Affine3d pose(cam2world);

    // std::string cam2world_file = data_path + "/frame-" + curr_frame_prefix.str() + ".pose.txt";
    // float sadasd[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    // std::vector<float> cam2world_vec = LoadMatrixFromFile(std::string(name), 4, 4);
    //  cam2world_vec.push_back(1);//
    // std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);

    cout << pose.matrix << endl;
    // Affine3d pose = Affine3d::Identity();
    // std::copy(cam2world_vec.begin(), cam2world_vec.end(), std::ostream_iterator<float>(std::cout, " "));
    // 计算相对于第一帧的位姿 (camera-to-base frame)
    // multiply_matrix(base2world_inv, cam2world, cam2base);
    cv::viz::Camera mainCamera = cv::viz::Camera::KinectCamera(Size(640, 480));                //new viz::Camera();         // 初始化相机类
    viz::WCameraPosition camParamsp(mainCamera.getFov(), _reacolor, 1.0, viz::Color::white()); // 相机参数设置
    window.showWidget("Camera", camParamsp, pose);
    std::cout << "Fusing: " << color_path << std::endl;

    Integrate(cam_K, cam2base, RGB_im, depth_im, im_height, im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
              voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
              voxel_grid_TSDF, voxel_grid_color, voxel_grid_weight);

    // sprintf(pname, "/home/lei/dataset/freiburg3_office/dd/%d.png", idx++);
    cv::Mat real;
    int P = 800; //高度
    int H = 640 * 1.3;
    int W = 480 * 1.3;
    int pp = P / 1.3;

    for (int pt_grid_z = 0; pt_grid_z < 512; pt_grid_z++)
    {
      int zindex = pt_grid_z * (pt_grid_z + 1) * (pt_grid_z * 2 + 1) / 6;

      for (int pt_grid_y = 0; pt_grid_y < pt_grid_z + 1; pt_grid_y++)
      {
        int yindex = pt_grid_y * (pt_grid_z + 1);

        for (int pt_grid_x = 0; pt_grid_x < pt_grid_z + 1; pt_grid_x++)
        {
          // int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;

          // voxel_grid_color[zindex + yindex + pt_grid_x] = 2;

          // // if ((voxel_grid_color[tindex]) > 1)
          // // {
          // Vec3f vec;
          // vec[2] = pt_grid_z * voxel_size;
          // vec[1] = pt_grid_y * voxel_size - pt_grid_z * voxel_size * 0.5;
          // vec[0] = pt_grid_x * voxel_size - pt_grid_z * voxel_size * 0.5;
          // real.push_back(vec);
          // }
          // pt_grid_x += 3;
        }
        // pt_grid_y += 3;
      }
      // pt_grid_z += 14;
    }
    for (size_t i = 0; i < 512 * 512 * 512; i++)
    {
      if (voxel_grid_color[i] < 1)
      {
        cout << i << endl;
        break;
      }
    }

    // for (int pt_grid_z = 0; pt_grid_z < 500; pt_grid_z++)
    // {
    //   for (int pt_grid_y = 0; pt_grid_y < 500; pt_grid_y++)
    //   {
    //     for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x)
    //     {
    //       int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;

    //       int tindex = pt_grid_z * (pt_grid_z + 1) * (pt_grid_z * 2 + 1) / 6;
    //       // voxel_grid_color[tindex] = 2;
    //       if ((voxel_grid_color[tindex]) > 1)
    //       {
    //         Vec3f vec;
    //         vec[2] = pt_grid_z * voxel_size;
    //         vec[1] = pt_grid_y * voxel_size;
    //         vec[0] = pt_grid_x * voxel_size - W * voxel_size * 1;
    //         real.push_back(vec);
    //       }
    //       pt_grid_x += 14;
    //     }
    //     pt_grid_y += 14;
    //   }
    //   pt_grid_z += 14;
    // }

    // cloud.addCloud(real, cv::viz::Color::bluberry()); //pose , pose

    cv::Mat tsdfpcloud2 = Voxel2PointCloud(voxel_grid_color, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                           voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                                           voxel_grid_TSDF, voxel_grid_weight, 0.2f, 0.0f); //tsdf阈值 权重阈值
    std::cout << "asd:" << tsdfpcloud2.rows << endl;
    // cloud.addCloud(pcloud2, cv::viz::Color::yellow(), cv::Affine3f(cam2world)); //pose

    Mat depth_cloud = getCload(_depth);
    // if (point_cloud1.rows == 0)
    //   continue;
    // cloud.addCloud(point_cloud1, cv::viz::Color::white(), pose); //pose

    // Matx33f intrisicParams(K(0, 0), 0.0, K(0, 2), 0.0, K(1, 1), K(1, 2), 0.0, 0.0, 1.0);   // 内参矩阵
    // cv::viz::Camera mainCamera = cv::viz::Camera::KinectCamera(Size(640, 480));           //new viz::Camera();         // 初始化相机类
    // viz::WCameraPosition camParamsp(mainCamera.getFov(), _reacolor, 1.0, viz::Color::white()); // 相机参数设置
    // window.showWidget("Camera", camParamsp, pose);                                        // cv::Affine3f(pose)

    // //**********************
    // sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/depth-%d.png", i);
    // depth = cv::imread(name, 2);
    // // sprintf(name, "./infinitam/%04d.pgm", i - 10);
    // // cv::imwrite(name, depth);
    Mat test;
    P = 800; //高度
    H = 640 * 1.3;
    W = 480 * 1.3;
    pp = P / 1.3;
    for (int i = 0; i < P;) //
    {
      for (int j = i * 0.5; j < H - i * 0.5;)
      {
        for (int k = i * 0.5; k < (W - i * 0.5);) //(500-i)*4
        {
          Vec3f vec;
          vec[2] = pp * voxel_size - i * voxel_size;
          vec[1] = j * voxel_size - H * voxel_size * 0.5;
          vec[0] = k * voxel_size - W * voxel_size * 0.5;
          test.push_back(Vec3f(vec));
          k += 15;
        }

        j += 15;
      }
      i += 15;
    }
    Mat largetest;

    P = P * 1.15; //高度
    H *= 1.3;
    W *= 1.3;
    pp = P / 1.3;

    for (int i = 0; i < P;) //
    {
      for (int j = i * 0.5; j < H - i * 0.5;)
      {
        for (int k = i * 0.5; k < (W - i * 0.5);) //(500-i)*4
        {
          Vec3f vec;
          vec[2] = pp * voxel_size - i * voxel_size;
          vec[1] = j * voxel_size - H * voxel_size * 0.5;
          vec[0] = k * voxel_size - W * voxel_size * 0.5;
          largetest.push_back(Vec3f(vec));
          k += 15;
        }
        j += 15;
      }
      i += 15;
    }
    cout << "P:" << P << " W:" << W << " H:" << H << " pp: " << pp << " " << largetest.rows / 1024.0 / 1024.0 * 6 * 15 * 225 << "MB" << std::endl;
    // cloud.addCloud(largetest, cv::viz::Color::blue(), pose); //,cv::Affine3f(pose)
    // cloud.addCloud(test, cv::viz::Color::yellow(), pose);    //,cv::Affine3f(pose)
    // cloud.addCloud(tsdfpcloud2, cv::viz::Color::red()); //,cv::Affine3f(pose) , pose
    Mat jix;
    for (float r = 0; r < 4.0; r += 0.05) //
    {
      for (float fy = 0; fy < 0.1; fy += 0.015)
      {
        for (float thta = 0; thta < 1.1; thta += 0.05) //(500-i)*4
        {
          Vec3f vec;
          vec[2] = r * cosf(thta);
          vec[1] = r * sinf(thta) * sinf(fy);
          vec[0] = r * sinf(thta) * cosf(fy);
          jix.push_back(vec);
          // k += 15;
        }
        // j += 15;
      }
      // i += 15;
    }
    // cloud.addCloud(jix, cv::viz::Color::red(), pose);          //,cv::Affine3f(pose) , pose
    cloud.addCloud(depth_cloud, cv::viz::Color::blue(), pose); //,cv::Affine3f(pose) , pose

    // for (int i = 0; i < 100 - 1; i++)
    // {
    //   cv::Vec3d asd = pcloud.at<cv::Vec3d>(i, 0);
    //   cv::Vec3d asd2 = pcloud.at<cv::Vec3d>(i + 1, 0);
    //   cv::viz::WCube wcube(asd, asd2);
    //   window.showWidget("cloud" + to_string(i), wcube);
    // }
    // if (cloud.rows == 0)
    // {
    //   continue;
    // }
    // cv::viz::WCloud cloud(pcloud);
    cout << mainCamera.getFov() << endl;
    window.showWidget("cloud", cloud);
    window.spinOnce(1, false);
    // window.spin();

    cv::waitKey(1);
    // view.showCloud(cloud); //再这个窗口显示点云
    // if (cloud.size() > 0)
    // {
    // main3d(cloud);
    // }

    // 清除数据并退出
    // cloud->points.clear();
    cout << "Point cloud saved." << endl;
  }

  // TSDF 拷贝到CPU

  // Compute surface points from TSDF voxel grid and save to point cloud .ply file
  std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;
  SaveVoxelGrid2SurfacePointCloud("tsdf.ply", voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                  voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                                  voxel_grid_TSDF, voxel_grid_weight, 0.2f, 0.0f);

  // Save TSDF voxel grid and its parameters to disk as binary file (float array)
  // std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;
  // std::string voxel_grid_saveto_path = "tsdf.bin";
  // std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
  // float voxel_grid_dim_xf = (float)voxel_grid_dim_x;
  // float voxel_grid_dim_yf = (float)voxel_grid_dim_y;
  // float voxel_grid_dim_zf = (float)voxel_grid_dim_z;
  // outFile.write((char *)&voxel_grid_dim_xf, sizeof(float));
  // outFile.write((char *)&voxel_grid_dim_yf, sizeof(float));
  // outFile.write((char *)&voxel_grid_dim_zf, sizeof(float));
  // outFile.write((char *)&voxel_grid_origin_x, sizeof(float));
  // outFile.write((char *)&voxel_grid_origin_y, sizeof(float));
  // outFile.write((char *)&voxel_grid_origin_z, sizeof(float));
  // outFile.write((char *)&voxel_size, sizeof(float));
  // outFile.write((char *)&trunc_margin, sizeof(float));
  // for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
  //   outFile.write((char *)&voxel_grid_TSDF[i], sizeof(float));
  // outFile.close();

  return 0;
}
