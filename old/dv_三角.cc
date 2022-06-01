
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

// // CUDA kernel function to integrate a TSDF voxel volume given depth images
// void Integrate(float *cam_K, float *cam2base, uint8_t *rgb_im, float *depth_im,
//                int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
//                float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
//                float *voxel_grid_TSDF, float *voxel_grid_color, float *voxel_grid_weight)
// {
// #pragma omp parallel
// #pragma omp for
//   for (int pt_grid_z = 0; pt_grid_z < 500; pt_grid_z++)
//     for (int pt_grid_y = 0; pt_grid_y < 500; pt_grid_y++)
//       for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x)
//       {

//         // Convert voxel center from grid coordinates to base frame camera coordinates
//         float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
//         float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
//         float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

//         // Convert from base frame camera coordinates to current frame camera coordinates
//         float tmp_pt[3] = {0};
//         tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
//         tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
//         tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
//         float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
//         float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
//         float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

//         if (pt_cam_z <= 0)
//           continue;

//         int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
//         int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
//         if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
//           continue;

//         float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];

//         if (depth_val <= 0 || depth_val > 6)
//           continue;

//         float diff = depth_val - pt_cam_z;

//         if (diff <= -trunc_margin)
//           continue;

//         // Integrate
//         int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
//         float dist = fmin(1.0f, diff / trunc_margin);
//         float weight_old = voxel_grid_weight[volume_idx];
//         float weight_new = weight_old + 1.0f;
//         voxel_grid_weight[volume_idx] = weight_new;
//         voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;

//         // Integrate color
//         uint8_t rgb_val = rgb_im[pt_pix_y * im_width + pt_pix_x];

//         // int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
//         // float dist = fmin(1.0f, diff / trunc_margin);
//         // float weight_old = voxel_grid_weight[volume_idx];
//         // float weight_new = weight_old + 1.0f;
//         // voxel_grid_weight[volume_idx] = weight_new;
//         voxel_grid_color[volume_idx] = (voxel_grid_color[volume_idx] * weight_old + rgb_val * (weight_new - weight_old)) / weight_new;
//         // cout << voxel_grid_color[volume_idx] << endl;
//       }
// }
// CUDA kernel function to integrate a TSDF voxel volume given depth images
void Integrate(float *cam_K, float *cam2base, uint8_t *rgb_im, float *depth_im,
               int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
               float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
               float *voxel_grid_TSDF, int32_t *voxel_grid_color, float *voxel_grid_weight)
{
#pragma omp parallel
#pragma omp for
  for (int pt_grid_z = 0; pt_grid_z < 500; pt_grid_z++)
  {
    int zindex = pt_grid_z * (pt_grid_z + 1) * (pt_grid_z * 2 + 1) / 6; //pt_grid_z + 1

    for (int pt_grid_y = 0; pt_grid_y < pt_grid_z + 1; pt_grid_y++)
    {
      int yindex = pt_grid_y * (pt_grid_z + 1);

      for (int pt_grid_x = 0; pt_grid_x < pt_grid_z + 1; ++pt_grid_x)
      {
        int volume_idx = zindex + yindex + pt_grid_x; //

        // Convert voxel center from grid coordinates to base frame camera coordinates
        float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
        float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
        float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

        // Convert from base frame camera coordinates to current frame camera coordinates
        float tmp_pt[3] = {0};
        tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
        tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
        tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
        float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
        float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
        float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

        if (pt_cam_z <= 0)
          continue;

        int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
        int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
        if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
          continue;

        float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];

        if (depth_val <= 0 || depth_val > 6)
          continue;

        float diff = depth_val - pt_cam_z;

        if (diff <= -trunc_margin)
          continue;

        // Integrate
        // int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
        float dist = fmin(1.0f, diff / trunc_margin);
        float weight_old = voxel_grid_weight[volume_idx];
        float weight_new = weight_old + 1.0f;
        voxel_grid_weight[volume_idx] = weight_new;
        voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;

        // Integrate color
        uint8_t rgb_val[3];
        rgb_val[0] = rgb_im[pt_pix_y * im_width + pt_pix_x];
        rgb_val[1] = rgb_im[pt_pix_y * im_width + pt_pix_x + 640 * 480];

        rgb_val[2] = rgb_im[pt_pix_y * im_width + pt_pix_x + 640 * 480 * 2];

        uint8_t rgb_valr = voxel_grid_color[volume_idx] & 0xff;
        uint8_t rgb_valg = (voxel_grid_color[volume_idx] >> 8) & 0xff;
        uint8_t rgb_valb = (voxel_grid_color[volume_idx] >> 16) & 0xff;

        // int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
        // float dist = fmin(1.0f, diff / trunc_margin);
        // float weight_old = voxel_grid_weight[volume_idx];
        // float weight_new = weight_old + 1.0f;
        // voxel_grid_weight[volume_idx] = weight_new;
        int mval = (rgb_valr * weight_old + rgb_val[0]) / weight_new;
        if (mval > 255)
        {
          mval = 255;
        }
        else
          rgb_valr = mval;

        mval = (rgb_valg * weight_old + rgb_val[1]) / weight_new;
        if (mval > 255)
        {
          mval = 255;
        }
        else
          rgb_valg = mval;

        mval = (rgb_valb * weight_old + rgb_val[2]) / weight_new;
        if (mval > 255)
        {
          mval = 255;
        }
        else
          rgb_valb = mval;

        // rgb_valr = (rgb_valr * weight_old + rgb_val * (weight_new - weight_old)) / weight_new;

        voxel_grid_color[volume_idx] = rgb_valb << 16 | rgb_valg << 8 | rgb_valr;
        // cout << voxel_grid_color[volume_idx] << endl;
      }
    }
  }
}
// Loads a binary file with depth data and generates a TSDF voxel volume (5m x 5m x 5m at 1cm resolution)
// Volume is aligned with respect to the camera coordinates of the first frame (a.k.a. base frame)
int main(int argc, char *argv[])
{

  // Location of camera intrinsic file
  std::string cam_K_file = "../data/camera-intrinsics.txt";

  // Location of folder containing RGB-D frames and camera pose files
  std::string data_path = "../data/data";
  int base_frame_idx = 0;
  int first_frame_idx = 0;
  int num_frames = 1050;

  float cam_K[3 * 3];
  float base2world[4 * 4];
  float cam2base[4 * 4];
  float cam2world[4 * 4];
  int im_width = 640;
  int im_height = 480;
  float depth_im[im_height * im_width];
  uint8_t RGB_im[im_height * im_width * 3];

  // Voxel grid parameters (change these to change voxel grid resolution, etc.)
  float voxel_grid_origin_x = -1.5f; // Location of voxel grid origin in base frame camera coordinates
  float voxel_grid_origin_y = -1.5f;
  float voxel_grid_origin_z = 0.5f;
  float voxel_size = 0.01f;
  float trunc_margin = voxel_size * 5;
  int voxel_grid_dim_x = 768;
  int voxel_grid_dim_y = 768;
  int voxel_grid_dim_z = 768;

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
  int32_t *voxel_grid_color = new int32_t[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];

  //所有格子置为1
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
    voxel_grid_TSDF[i] = 1.0f;
  //权重置为0
  memset(voxel_grid_weight, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);

  char *pname = new char[256];
  int idx = 1;

  cv::viz::Viz3d window("window");

  // cout << depth << endl;
  //显示坐标系
  window.showWidget("Coordinate", viz::WCoordinateSystem());

  //创建一个储存point cloud的图片
  char *name = new char[256];
  int i = 11;
  cv::viz::WCube cube(Vec3d(-1.5, -1.5, 0.5), Vec3d(1.5, 1.5, 3.5));
  window.showWidget("tsdf", cube);

  cv::Mat mpc;
  float aplen = 512 * voxel_size;

  mpc.push_back(Vec3f(aplen, aplen, aplen));
  mpc.push_back(Vec3f(-aplen, aplen, aplen));
  mpc.push_back(Vec3f(-aplen, -aplen, aplen));
  mpc.push_back(Vec3f(aplen, -aplen, aplen));
  mpc.push_back(Vec3f(0, 0, 0));
  mpc.push_back(Vec3f(aplen, aplen, aplen));
  mpc.push_back(Vec3f(aplen, -aplen, aplen));
  mpc.push_back(Vec3f(-aplen, aplen, aplen));
  mpc.push_back(Vec3f(0, 0, 0));
  mpc.push_back(Vec3f(-aplen, -aplen, aplen));

  // mpc.push_back(Vec3f(0, 0, 0));
  // mpc.push_back(Vec3f(aplen, aplen, aplen));
  // mpc.push_back(Vec3f(aplen, -aplen, aplen));
  //   mpc.push_back(Vec3f(0, 0, 0));
  // mpc.push_back(Vec3f(-aplen, -aplen, aplen));
  // mpc.push_back(Vec3f(aplen, -aplen, aplen));
  // mpc.push_back(Vec3f(aplen, aplen, aplen));
  // mpc.push_back(Vec3f(-aplen, aplen, aplen));
  cv::viz::WPolyLine wpl(mpc);
  Affine3f a3f(cv::Mat_<float>::eye(Size(3, 3)), cv::Vec3f(0.0, 0.0, 0.5));
  window.showWidget("mpc", wpl, a3f); //, Affine3f(base2world)

  // 重建 TSDF voxel grid
  for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; frame_idx += 5)
  {
    cv::viz::WCloudCollection cloud;

    std::ostringstream curr_frame_prefix;
    curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;
    // sprintf(name, "/home/lei/dataset/infiniTAM/Files/tum/Frames/%04d.pgm", i++);
    // sprintf(name, "/home/lei/dataset/surfulwrap/ubnm/frame-%06d.depth.png", i++);
    // std::string depth_im_file = "/home/lei/图片/img/" + std::to_string(i) + ".png";
    i += 2;
    // 读取深度图
    // sprintf(name, "/home/lei/docker/res/killingFusionCuda-master/data/hat/depth_%06d.png", frame_idx);
    // sprintf(name, "/home/lei/dataset/infiniTAM/Files/tum/Frames/%04d.pgm", i++);

    std::string depth_im_file = data_path + "/frame-" + curr_frame_prefix.str() + ".depth.png";
    ReadDepth(depth_im_file, im_height, im_width, depth_im);
    // ReadDepth(string(name), im_height, im_width, depth_im);
    depth_im_file = data_path + "/frame-" + curr_frame_prefix.str() + ".color.jpg";
    ReadRGB(depth_im_file, im_height, im_width, RGB_im);
    // 读取位姿
    std::string cam2world_file = data_path + "/frame-" + curr_frame_prefix.str() + ".pose.txt";
    std::vector<float> cam2world_vec = LoadMatrixFromFile(cam2world_file, 4, 4);
    std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);
    float sadasd[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    // std::vector<float> cam2world_vec(sadasd);// = LoadMatrixFromFile(cam2world_file, 4, 4);
    //  cam2world_vec.push_back(1);//
    // std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);

    // 计算相对于第一帧的位姿 (camera-to-base frame)
    // multiply_matrix(base2world_inv, sadasd, cam2base);
    // std::copy(cam2world_vec.begin(), cam2world_vec.end(), std::ostream_iterator<float>(std::cout, " "));
    multiply_matrix(base2world_inv, cam2world, cam2base);

    cv::viz::Camera mainCamera = cv::viz::Camera::KinectCamera(Size(640, 480));     //new viz::Camera();         // 初始化相机类
    viz::WCameraPosition camParamsp(mainCamera.getFov(), 5.0, viz::Color::white()); // 相机参数设置
    window.showWidget("Camera", camParamsp, Affine3f(cam2base));
    std::cout << "Fusing: " << depth_im_file << std::endl;

    // std::cout << "Fusing: " << depth_im_file << std::endl;
    // voxel_grid_origin_x += 0.1;
    Integrate(cam_K, cam2base, RGB_im, depth_im, im_height, im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
              voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
              voxel_grid_TSDF, voxel_grid_color, voxel_grid_weight);

    // sprintf(pname, "/home/lei/dataset/freiburg3_office/dd/%d.png", idx++);
    cv::Mat color;
    cv::Mat point_cloud = Voxel2PointCloud(voxel_grid_color, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                           voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                                           voxel_grid_TSDF, voxel_grid_weight, 0.2f, 0.0f, color); // cv::imread(pname, CV_16UC1);
    if (point_cloud.rows == 0)
    {
      continue;
    }
    // cv::viz::WCloud cloud(point_cloud, color);
   
    Mat jix;
    for (int fy = -250; fy <250; fy +=10)
    {
      for (int r = 0; r < 400; r +=10) //
      {
        for (int thta = -100; thta <100; thta += 10) //(500-i)*4
        {
          Vec3f vec;
          vec[0] = fy*voxel_size;
          vec[1] = r*voxel_size * sinf(thta*voxel_size);
          vec[2] = r*voxel_size * cosf(thta*voxel_size);
          jix.push_back(vec);
          // k += 15;
        }
        // j += 15;
      }
      // i += 15;
    }
    // cloud.addCloud(jix, cv::viz::Color::red(), pose);          //,cv::Affine3f(pose) , pose
    cloud.addCloud(jix, cv::viz::Color::blue()); //,cv::Affine3f(pose) , pose
 window.showWidget("cloud", cloud); //,Affine3f(cv::Mat_<float>::eye(Size(3, 3)), cv::Vec3f(voxel_grid_origin_x, 0.0, 0))

    window.spinOnce(1, false);

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
