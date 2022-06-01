// ---------------------------------------------------------
// Author: Andy Zeng, Princeton University, 2016
// ---------------------------------------------------------

#include <vector>
#include <opencv2/opencv.hpp>

// #include <pcl/common/common_headers.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/visualization/cloud_viewer.h>
// #include <pcl/console/parse.h>
// #include <pcl/io/ply_io.h>

using namespace std;
// 定义点云类型

// typedef pcl::PointXYZRGBA PointT;
// typedef pcl::PointCloud<PointT> PointCloud;

// Compute surface points from TSDF voxel grid and save points to point cloud file
void SaveVoxelGrid2SurfacePointCloud(const std::string &file_name, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                                     float voxel_size, float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
                                     float *voxel_grid_TSDF, float *voxel_grid_weight,
                                     float tsdf_thresh, float weight_thresh)
{

  // Count total number of points in point cloud
  int num_pts = 0;
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++)
    if (std::abs(voxel_grid_TSDF[i]) < tsdf_thresh && voxel_grid_weight[i] > weight_thresh)
      num_pts++;

  // Create header for .ply file
  FILE *fp = fopen(file_name.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_pts);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "end_header\n");

  // Create point cloud content for ply file
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++)
  {

    // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
    if (std::abs(voxel_grid_TSDF[i]) < tsdf_thresh && voxel_grid_weight[i] > weight_thresh)
    {

      // Compute voxel indices in int for higher positive number range
      int z = floor(i / (voxel_grid_dim_x * voxel_grid_dim_y));
      int y = floor((i - (z * voxel_grid_dim_x * voxel_grid_dim_y)) / voxel_grid_dim_x);
      int x = i - (z * voxel_grid_dim_x * voxel_grid_dim_y) - (y * voxel_grid_dim_x);

      // Convert voxel indices to float, and save coordinates to ply file
      float pt_base_x = voxel_grid_origin_x + (float)x * voxel_size;
      float pt_base_y = voxel_grid_origin_y + (float)y * voxel_size;
      float pt_base_z = voxel_grid_origin_z + (float)z * voxel_size;
      fwrite(&pt_base_x, sizeof(float), 1, fp);
      fwrite(&pt_base_y, sizeof(float), 1, fp);
      fwrite(&pt_base_z, sizeof(float), 1, fp);
    }
  }
  fclose(fp);
}

cv::Mat SaveVoxelGrid2SurfacePointCloudMem(float *voxel_grid_color, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                                           float voxel_size, float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
                                           float *voxel_grid_TSDF, float *voxel_grid_weight,
                                           float tsdf_thresh, float weight_thresh, cv::Mat &color)
{

  cv::Mat cloud;
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++)
  {
    // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
    if (std::abs(voxel_grid_TSDF[i]) < tsdf_thresh && voxel_grid_weight[i] > weight_thresh)
    {

      // Compute voxel indices in int for higher positive number range
      int z = floor(i / (voxel_grid_dim_x * voxel_grid_dim_y));                          //计算体素Z
      int y = floor((i - (z * voxel_grid_dim_x * voxel_grid_dim_y)) / voxel_grid_dim_x); //计算Y
      int x = i - (z * voxel_grid_dim_x * voxel_grid_dim_y) - (y * voxel_grid_dim_x);    //计算X
      cv::Vec<double, 3> ves;
      // Convert voxel indices to float, and save coordinates to ply file
      // float pt_base_x
      //base+体素坐标×体素长度==世界坐标
      ves[0] = voxel_grid_origin_x + (double)x * voxel_size;
      ves[1] = voxel_grid_origin_y + (double)y * voxel_size;
      ves[2] = voxel_grid_origin_z + (double)z * voxel_size;

      // ves[3] = voxel_grid_weight[i];

      // ves[4] = voxel_grid_origin_z + (double)z * voxel_size;

      // ves[5] = voxel_grid_origin_z + (double)z * voxel_size;
      cv::Vec<uint8_t, 3> vescolor;
      vescolor[0] = voxel_grid_color[i];
      vescolor[1] = voxel_grid_color[i];
      vescolor[2] = voxel_grid_color[i];
      // cout << vescolor << endl;
      // 从rgb图像中获取它的颜色
      // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
      // p.b = voxel_grid_color[i];
      // p.g = voxel_grid_color[i];
      // p.r = voxel_grid_color[i];
      // cout << voxel_grid_color[i] << endl;
      // 把p加入到点云中
      cloud.push_back(ves);
      color.push_back(vescolor);
      // Eigen::Vector3f center(p.x, p.y, p.z);
      // Eigen::Quaternionf rotation(1, 0, 0, 0);
      // string cube = "cude" + to_string(i);
      // viewer->addCube(center, rotation, voxel, voxel, voxel, cube);
    }
  }
  return cloud;
}
cv::Mat Voxel2PointCloud(int32_t *voxel_grid_color, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                         float voxel_size, float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
                         float *voxel_grid_TSDF, float *voxel_grid_weight,
                         float tsdf_thresh, float weight_thresh, cv::Mat &color)
{
  cv::Mat cloud;
  for (int pt_grid_z = 0; pt_grid_z < voxel_grid_dim_z; pt_grid_z++)
  {
    int zindex = pt_grid_z * (pt_grid_z + 1) * (pt_grid_z * 2 + 1) / 6;
    for (int pt_grid_y = 0; pt_grid_y < pt_grid_z + 1; pt_grid_y++)
    {
      int yindex = pt_grid_y * (pt_grid_z + 1);
      for (int pt_grid_x = 0; pt_grid_x < pt_grid_z + 1; pt_grid_x++)
      {
        int volume_idx = zindex + yindex + pt_grid_x;
        // voxel_grid_color = 2;
        if (std::abs(voxel_grid_TSDF[volume_idx]) < tsdf_thresh && voxel_grid_weight[volume_idx] > weight_thresh) //
        // // if ((voxel_grid_color[tindex]) > 1)
        {
          cv::Vec3f vec;
          vec[2] = voxel_grid_origin_z + pt_grid_z * voxel_size;
          vec[1] = voxel_grid_origin_y + pt_grid_y * voxel_size; //- pt_grid_z * 0.006 * 0.5 - pt_grid_z * 0.006 * 0.5
          vec[0] = voxel_grid_origin_x + pt_grid_x * voxel_size;
          cloud.push_back(vec);

          cv::Vec<uint8_t, 3> vescolor;
          vescolor[0] = (voxel_grid_color[volume_idx]) & 0xff;
          vescolor[1] = (voxel_grid_color[volume_idx] >> 8) & 0xff;
          vescolor[2] = (voxel_grid_color[volume_idx] >> 16) & 0xff;
          // cout << vescolor << endl;
          // 从rgb图像中获取它的颜色
          // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
          // p.b = voxel_grid_color[i];
          // p.g = voxel_grid_color[i];
          // p.r = voxel_grid_color[i];
          // cout << voxel_grid_color[i] << endl;
          // 把p加入到点云中
          color.push_back(vescolor);
        }
        // pt_grid_x += 3;
      }
      // pt_grid_y += 3;
    }
    // pt_grid_z += 14;
  }

  //   for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++)
  //   {
  //     // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
  //     if (std::abs(voxel_grid_TSDF[i]) < tsdf_thresh && voxel_grid_weight[i] > weight_thresh)
  //     {

  //       // Compute voxel indices in int for higher positive number range
  //       int z = floor(i / (voxel_grid_dim_x * voxel_grid_dim_y));//计算体素Z
  //       int y = floor((i - (z * voxel_grid_dim_x * voxel_grid_dim_y)) / voxel_grid_dim_x);//计算Y
  //       int x = i - (z * voxel_grid_dim_x * voxel_grid_dim_y) - (y * voxel_grid_dim_x);//计算X
  //       cv::Vec3d ves;
  //       // Convert voxel indices to float, and save coordinates to ply file
  //       // float pt_base_x
  // //base+体素坐标×体素长度==世界坐标
  //       ves[0] = voxel_grid_origin_x + (double)x * voxel_size;
  //       ves[1] = voxel_grid_origin_y + (double)y * voxel_size;
  //       ves[2] = voxel_grid_origin_z + (double)z * voxel_size;
  //       // 从rgb图像中获取它的颜色
  //       // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
  //       // p.b = voxel_grid_color[i];
  //       // p.g = voxel_grid_color[i];
  //       // p.r = voxel_grid_color[i];
  //       // cout << voxel_grid_color[i] << endl;
  //       // 把p加入到点云中
  //       cloud.push_back(ves);

  //       // Eigen::Vector3f center(p.x, p.y, p.z);
  //       // Eigen::Quaternionf rotation(1, 0, 0, 0);
  //       // string cube = "cude" + to_string(i);
  //       // viewer->addCube(center, rotation, voxel, voxel, voxel, cube);
  //     }
  //   }
  return cloud;
}
// Load an M x N matrix from a text file (numbers delimited by spaces/tabs)
// Return the matrix as a float vector of the matrix in row-major order
std::vector<float> LoadMatrixFromFile(std::string filename, int M, int N)
{
  std::vector<float> matrix;
  FILE *fp = fopen(filename.c_str(), "r");
  for (int i = 0; i < M * N; i++)
  {
    float tmp;
    int iret = fscanf(fp, "%f", &tmp);
    matrix.push_back(tmp);
  }
  fclose(fp);
  return matrix;
}

// Read a depth image with size H x W and save the depth values (in meters) into a float array (in row-major order)
// The depth image file is assumed to be in 16-bit PNG format, depth in millimeters
void ReadDepth(std::string filename, int H, int W, float *depth)
{
  cv::Mat depth_mat = cv::imread(filename, -1);
  // std::cout<<depth_mat<<std::endl;
  if (depth_mat.empty())
  {
    std::cout << "Error: depth image  at:" << filename << std::endl;
    cv::waitKey(0);
  }
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c)
    {
      depth[r * W + c] = (float)(depth_mat.at<unsigned short>(r, c)) / 1000.0f;
      // std::cout << depth[r * W + c] << " "/;
      if (depth[r * W + c] > 6.0f) // Only consider depth < 6m
        depth[r * W + c] = 0;
    }
  // std::cout << std::endl;
}
void ReadRGB(std::string filename, int H, int W, uint8_t *depth)
{
  cv::Mat srcImage = cv::imread(filename);
  std::vector<cv::Mat> channels;
  cv::split(srcImage, channels);
  for (int i = 0; i < 3; i++)
  {
    // std::cout<<depth_mat<<std::endl;
    if (channels[i].empty())
    {
      std::cout << "Error: rgb image  at:" << filename << std::endl;
      cv::waitKey(0);
    }
    for (int r = 0; r < H; ++r)
      for (int c = 0; c < W; ++c)
      {
        depth[i * 640 * 480 + r * W + c] = (uint8_t)(channels[i].at<uint8_t>(r, c));
        // if (depth[r * W + c] > 6.0f) // Only consider depth < 6m
        // depth[r * W + c] = 0;
      }
  }
}
// 4x4 matrix multiplication (matrices are stored as float arrays in row-major order)
void multiply_matrix(const float m1[16], const float m2[16], float mOut[16])
{
  mOut[0] = m1[0] * m2[0] + m1[1] * m2[4] + m1[2] * m2[8] + m1[3] * m2[12];
  mOut[1] = m1[0] * m2[1] + m1[1] * m2[5] + m1[2] * m2[9] + m1[3] * m2[13];
  mOut[2] = m1[0] * m2[2] + m1[1] * m2[6] + m1[2] * m2[10] + m1[3] * m2[14];
  mOut[3] = m1[0] * m2[3] + m1[1] * m2[7] + m1[2] * m2[11] + m1[3] * m2[15];

  mOut[4] = m1[4] * m2[0] + m1[5] * m2[4] + m1[6] * m2[8] + m1[7] * m2[12];
  mOut[5] = m1[4] * m2[1] + m1[5] * m2[5] + m1[6] * m2[9] + m1[7] * m2[13];
  mOut[6] = m1[4] * m2[2] + m1[5] * m2[6] + m1[6] * m2[10] + m1[7] * m2[14];
  mOut[7] = m1[4] * m2[3] + m1[5] * m2[7] + m1[6] * m2[11] + m1[7] * m2[15];

  mOut[8] = m1[8] * m2[0] + m1[9] * m2[4] + m1[10] * m2[8] + m1[11] * m2[12];
  mOut[9] = m1[8] * m2[1] + m1[9] * m2[5] + m1[10] * m2[9] + m1[11] * m2[13];
  mOut[10] = m1[8] * m2[2] + m1[9] * m2[6] + m1[10] * m2[10] + m1[11] * m2[14];
  mOut[11] = m1[8] * m2[3] + m1[9] * m2[7] + m1[10] * m2[11] + m1[11] * m2[15];

  mOut[12] = m1[12] * m2[0] + m1[13] * m2[4] + m1[14] * m2[8] + m1[15] * m2[12];
  mOut[13] = m1[12] * m2[1] + m1[13] * m2[5] + m1[14] * m2[9] + m1[15] * m2[13];
  mOut[14] = m1[12] * m2[2] + m1[13] * m2[6] + m1[14] * m2[10] + m1[15] * m2[14];
  mOut[15] = m1[12] * m2[3] + m1[13] * m2[7] + m1[14] * m2[11] + m1[15] * m2[15];
}

// 4x4 matrix inversion (matrices are stored as float arrays in row-major order)
bool invert_matrix(const float m[16], float invOut[16])
{
  float inv[16], det;
  int i;
  inv[0] = m[5] * m[10] * m[15] -
           m[5] * m[11] * m[14] -
           m[9] * m[6] * m[15] +
           m[9] * m[7] * m[14] +
           m[13] * m[6] * m[11] -
           m[13] * m[7] * m[10];

  inv[4] = -m[4] * m[10] * m[15] +
           m[4] * m[11] * m[14] +
           m[8] * m[6] * m[15] -
           m[8] * m[7] * m[14] -
           m[12] * m[6] * m[11] +
           m[12] * m[7] * m[10];

  inv[8] = m[4] * m[9] * m[15] -
           m[4] * m[11] * m[13] -
           m[8] * m[5] * m[15] +
           m[8] * m[7] * m[13] +
           m[12] * m[5] * m[11] -
           m[12] * m[7] * m[9];

  inv[12] = -m[4] * m[9] * m[14] +
            m[4] * m[10] * m[13] +
            m[8] * m[5] * m[14] -
            m[8] * m[6] * m[13] -
            m[12] * m[5] * m[10] +
            m[12] * m[6] * m[9];

  inv[1] = -m[1] * m[10] * m[15] +
           m[1] * m[11] * m[14] +
           m[9] * m[2] * m[15] -
           m[9] * m[3] * m[14] -
           m[13] * m[2] * m[11] +
           m[13] * m[3] * m[10];

  inv[5] = m[0] * m[10] * m[15] -
           m[0] * m[11] * m[14] -
           m[8] * m[2] * m[15] +
           m[8] * m[3] * m[14] +
           m[12] * m[2] * m[11] -
           m[12] * m[3] * m[10];

  inv[9] = -m[0] * m[9] * m[15] +
           m[0] * m[11] * m[13] +
           m[8] * m[1] * m[15] -
           m[8] * m[3] * m[13] -
           m[12] * m[1] * m[11] +
           m[12] * m[3] * m[9];

  inv[13] = m[0] * m[9] * m[14] -
            m[0] * m[10] * m[13] -
            m[8] * m[1] * m[14] +
            m[8] * m[2] * m[13] +
            m[12] * m[1] * m[10] -
            m[12] * m[2] * m[9];

  inv[2] = m[1] * m[6] * m[15] -
           m[1] * m[7] * m[14] -
           m[5] * m[2] * m[15] +
           m[5] * m[3] * m[14] +
           m[13] * m[2] * m[7] -
           m[13] * m[3] * m[6];

  inv[6] = -m[0] * m[6] * m[15] +
           m[0] * m[7] * m[14] +
           m[4] * m[2] * m[15] -
           m[4] * m[3] * m[14] -
           m[12] * m[2] * m[7] +
           m[12] * m[3] * m[6];

  inv[10] = m[0] * m[5] * m[15] -
            m[0] * m[7] * m[13] -
            m[4] * m[1] * m[15] +
            m[4] * m[3] * m[13] +
            m[12] * m[1] * m[7] -
            m[12] * m[3] * m[5];

  inv[14] = -m[0] * m[5] * m[14] +
            m[0] * m[6] * m[13] +
            m[4] * m[1] * m[14] -
            m[4] * m[2] * m[13] -
            m[12] * m[1] * m[6] +
            m[12] * m[2] * m[5];

  inv[3] = -m[1] * m[6] * m[11] +
           m[1] * m[7] * m[10] +
           m[5] * m[2] * m[11] -
           m[5] * m[3] * m[10] -
           m[9] * m[2] * m[7] +
           m[9] * m[3] * m[6];

  inv[7] = m[0] * m[6] * m[11] -
           m[0] * m[7] * m[10] -
           m[4] * m[2] * m[11] +
           m[4] * m[3] * m[10] +
           m[8] * m[2] * m[7] -
           m[8] * m[3] * m[6];

  inv[11] = -m[0] * m[5] * m[11] +
            m[0] * m[7] * m[9] +
            m[4] * m[1] * m[11] -
            m[4] * m[3] * m[9] -
            m[8] * m[1] * m[7] +
            m[8] * m[3] * m[5];

  inv[15] = m[0] * m[5] * m[10] -
            m[0] * m[6] * m[9] -
            m[4] * m[1] * m[10] +
            m[4] * m[2] * m[9] +
            m[8] * m[1] * m[6] -
            m[8] * m[2] * m[5];

  det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

  if (det == 0)
    return false;

  det = 1.0 / det;

  for (i = 0; i < 16; i++)
    invOut[i] = inv[i] * det;

  return true;
}

void save()
{
  // TSDF 拷贝到CPU

  // Compute surface points from TSDF voxel grid and save to point cloud .ply file
  std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;
  // SaveVoxelGrid2SurfacePointCloud("tsdf.ply", voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
  //                                 voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
  //                                 voxel_grid_TSDF, voxel_grid_weight, 0.2f, 0.0f);

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
}

cv::Mat getCload(cv::Mat &depth)
{
  cv::Mat point_cloud; // = Mat::zeros(height, width, CV_32FC3);
  //point cloud 赋值，其中 fx,fy,cx,cy 为Kinect2 的内参
  double asd = 550;
  double fx = asd, fy = asd, cx = 320.0, cy = 240.0;
  cv::Mat outdepth;
  depth.convertTo(outdepth, CV_32FC1, 0.001f);
  for (int row = 0; row < depth.rows; row++)
    for (int col = 0; col < depth.cols; col++)
    {
      // float dz = ((float)depth.at<unsigned short>(row, col)) / 1000.0;
      float dz = outdepth.at<float>(row, col);
      cv::Vec4f vec;
      // dz ;
      if (dz > 5.0)
        continue;
      if (dz < 0.32)
        continue;
      {
        vec[0] = dz * (col - cx) / fx;
        vec[1] = dz * (row - cy) / fy;
        vec[2] = dz;
        vec[3] = 1.0f;

        // if (dz < 1.6 && vec[1] > 0.3)
        point_cloud.push_back(vec);
      }
    }
  return point_cloud;
}

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
bool printMAT(cv::Mat &img_vec) //整体读出
{

  printf("SAVE:cols=%d,type=%d,em_size=%ld,rows=%d,channels=%d\n", img_vec.cols, img_vec.type(), img_vec.elemSize1(), img_vec.rows, img_vec.channels());

  return true;
}
template <typename T>
void memsetSimple(T *begin, int count, const T &value)
{
  T *end = begin + count;
  while (begin != end)
    *begin++ = value;
}
