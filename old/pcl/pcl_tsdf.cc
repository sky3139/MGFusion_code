// ---------------------------------------------------------
// Author: Andy Zeng, Princeton University, 2016
// ---------------------------------------------------------

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "pcl_tsdf.hpp"
#include <iterator>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
using namespace cv;
using namespace std;
#include <algorithm>
#include <cmath>

// CUDA kernel function to integrate a TSDF voxel volume given depth images
void Integrate(float *cam_K, float *cam2base, int *rgb_im, float *depth_im,
               int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
               float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
               float *voxel_grid_TSDF, float *voxel_grid_color, float *voxel_grid_weight)
{
#pragma omp parallel
#pragma omp for
  for (int pt_grid_z = 0; pt_grid_z < 500; pt_grid_z++)
    for (int pt_grid_y = 0; pt_grid_y < 500; pt_grid_y++)
      for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x)
      {

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
        int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
        float dist = fmin(1.0f, diff / trunc_margin);
        float weight_old = voxel_grid_weight[volume_idx];
        float obs_weight = 1.0f;
        float weight_new = weight_old + obs_weight;
        voxel_grid_weight[volume_idx] = weight_new;
        voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;

        // Integrate color

        color old_color;
        old_color.val = voxel_grid_color[volume_idx];
        uint8_t old_b = old_color.rgb[0];
        uint8_t old_g = old_color.rgb[1];
        uint8_t old_r = old_color.rgb[2];

        color new_color;
        new_color.val = rgb_im[pt_pix_y * im_width + pt_pix_x];

        uint8_t new_b = new_color.rgb[0];
        uint8_t new_g = new_color.rgb[1];
        uint8_t new_r = new_color.rgb[2];
        // cout << __LINE__ << ":" << (int)old_color.rgb[0] << " " << (int)old_color.rgb[1] << " " << (int)old_color.rgb[2] << endl;
        // cout << endl;
        // cout << endl;

        // cout << __LINE__ << ":" << (int)new_color.rgb[0] << " " << (int)new_color.rgb[1] << " " << (int)new_color.rgb[2] << endl;

        int new_bf = std::min(255, (int)((weight_old * old_b + obs_weight * new_b) / weight_new));
        int new_gf = std::min(255, (int)((weight_old * old_g + obs_weight * new_g) / weight_new));
        int new_rf = std::min(255, (int)((weight_old * old_r + obs_weight * new_r) / weight_new));
        //  self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_b*self._color_const + new_g*256 + new_r
        new_color.rgb[0] = (uint8_t)new_bf;
        new_color.rgb[1] = (uint8_t)new_gf;
        new_color.rgb[2] = (uint8_t)new_rf;
        // cout << endl;
        // cout << endl;
        // cout << __LINE__ << ":" << new_bf << " " << new_gf << " " << new_rf << endl;
        // cout << endl;
        // cout << endl;
        voxel_grid_color[volume_idx] = new_color.val;
        // cout << __LINE__ << ":" << voxel_grid_color[volume_idx] << endl;

        // int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
        // float dist = fmin(1.0f, diff / trunc_margin);
        // float weight_old = voxel_grid_weight[volume_idx];
        // float weight_new = weight_old + 1.0f;
        // voxel_grid_weight[volume_idx] = weight_new;
        //  voxel_grid_color[volume_idx] = (rgb_val * weight_new + weight_old * voxel_grid_color[volume_idx]) / weight_new;
      }
}
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <pcl/io/ply_io.h>
void viewerOneOff(pcl::visualization::PCLVisualizer &viewer)
{
  viewer.setBackgroundColor(1.0, 1.0, 1.0); //设置背景颜色
  std::cout << "I only run once" << std::endl;
}
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
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
  int first_frame_idx = 85;
  int num_frames = 1050;

  float cam_K[3 * 3];
  float base2world[4 * 4];
  float cam2base[4 * 4];
  float cam2world[4 * 4];
  int im_width = 640;
  int im_height = 480;
  float depth_im[im_height * im_width];
  int RGB_im[im_height * im_width];

  // Voxel grid parameters (change these to change voxel grid resolution, etc.)
  float voxel_grid_origin_x = -1.5f; // Location of voxel grid origin in base frame camera coordinates
  float voxel_grid_origin_y = -1.5f;
  float voxel_grid_origin_z = 0.5f;
  float voxel_size = 0.006f;
  float trunc_margin = voxel_size * 5;
  int voxel_grid_dim_x = 500;
  int voxel_grid_dim_y = 500;
  int voxel_grid_dim_z = 500;

  //读取相机内参
  std::vector<float> cam_K_vec = LoadMatrixFromFile(cam_K_file, 3, 3);
  std::copy(cam_K_vec.begin(), cam_K_vec.end(), cam_K);
  char *name = new char[256];
  // 读取第一帧的位姿
  std::ostringstream base_frame_prefix;
  base_frame_prefix << std::setw(6) << std::setfill('0') << base_frame_idx;
  sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/%d.txt", 85);

  std::string base2world_file = string(name); // data_path + "/frame-" + base_frame_prefix.str() + ".pose.txt";
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

  char *pname = new char[256];
  int idx = 1;
  PointCloud::Ptr cloud(new PointCloud);
  pcl::visualization::CloudViewer view("Simple Cloud Viewer"); //直接创造一个显示窗口
  // pcl::visualization::PointCloudColorHandlerCustom<PointT> single_color(cloud, 0, 255, 0); //设置背景色
  // view.runOnVisualizationThreadOnce(viewerOneOff);
  // 重建 TSDF voxel grid

  for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; ++frame_idx)
  {

    std::ostringstream curr_frame_prefix;
    curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;

    sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/depth-%d.png", frame_idx);

    // 读取深度图
    std::string depth_im_file(name); // = data_path + "/frame-" + curr_frame_prefix.str() + ".depth.png";
    ReadDepth(depth_im_file, im_height, im_width, depth_im);
    sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/color-%d.png", frame_idx);

    depth_im_file = string(name); // data_path + "/frame-" + curr_frame_prefix.str() + ".color.jpg";
    ReadRGB(depth_im_file, im_height, im_width, RGB_im);
    // 读取位姿
    sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/%d.txt", frame_idx);

    std::string cam2world_file = string(name); //data_path + "/frame-" + curr_frame_prefix.str() + ".pose.txt";
    std::vector<float> cam2world_vec = LoadMatrixFromFile(cam2world_file, 4, 4);
    std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);

    // std::copy(cam2world_vec.begin(), cam2world_vec.end(), std::ostream_iterator<float>(std::cout, " "));
    // 计算相对于第一帧的位姿 (camera-to-base frame)
    multiply_matrix(base2world_inv, cam2world, cam2base);

    std::cout << "Fusing: " << depth_im_file << std::endl;

    Integrate(cam_K, cam2base, RGB_im, depth_im, im_height, im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
              voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
              voxel_grid_TSDF, voxel_grid_color, voxel_grid_weight);

    // sprintf(pname, "/home/lei/dataset/freiburg3_office/dd/%d.png", idx++);

    cloud = CloudMem("tsdf.ply", voxel_grid_color, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                     voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                     voxel_grid_TSDF, voxel_grid_weight, 0.2f, 0.0f); // cv::imread(pname, CV_16UC1);

    cv::waitKey(1);
    view.showCloud(cloud); //再这个窗口显示点云
    // if (cloud.size() > 0)
    // {
    // main3d(cloud);
    // }

    // 清除数据并退出
    cloud->points.clear();
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
