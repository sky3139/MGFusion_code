#include <vector>
#include "pcl_tsdf.hpp"
#include <vector>
#include <opencv2/opencv.hpp>

void ReadRGB(std::string filename, int H, int W, int *depth)
{
  cv::Mat depth_mat = cv::imread(filename);

  // std::cout<<depth_mat<<std::endl;
  if (depth_mat.empty())
  {
    std::cout << "Error: rgb image  at:" << filename << std::endl;
    cv::waitKey(0);
  }
  cv::imshow("rgb", depth_mat);
  cv::waitKey(1);
  for (int r = 0; r < H; ++r)
  {
    for (int c = 0; c < W; ++c)
    {

      uchar *temp_ptr = &((uchar *)(depth_mat.data + depth_mat.step * r))[c * 3];

      color co;
      co.rgb[0] = temp_ptr[0];
      co.rgb[1] = temp_ptr[1];
      co.rgb[2] = temp_ptr[2];

      depth[r * W + c] = co.val;

      // cout << (int)co.rgb[0] << " " <<(int) co.rgb[1] << " " << (int)co.rgb[2] << endl;
      // cout << depth[r * W + c] << endl;

      // if (depth[r * W + c] > 6.0f) // Only consider depth < 6m
      // depth[r * W + c] = 0;
    }
  }
}

#include <vector>
#include <opencv2/opencv.hpp>

// #include <pcl/common/common_headers.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/visualization/cloud_viewer.h>
// #include <pcl/console/parse.h>
// #include <pcl/io/ply_io.h>

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
    std::cout << "Error: depth image file not read!" << std::endl;
    cv::waitKey(0);
  }
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c)
    {
      depth[r * W + c] = (float)(depth_mat.at<unsigned short>(r, c)) / 5000.0f;
      if (depth[r * W + c] > 8.0f) // Only consider depth < 6m
        depth[r * W + c] = 0;
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

// void FatalError(const int lineNumber = 0)
// {
//   std::cerr << "FatalError";
//   if (lineNumber != 0)
//     std::cerr << " at LINE " << lineNumber;
//   std::cerr << ". Program Terminated." << std::endl;
//   cudaDeviceReset();
//   exit(EXIT_FAILURE);
// }

// void checkCUDA(const int lineNumber, cudaError_t status)
// {
//   if (status != cudaSuccess)
//   {
//     std::cerr << "CUDA failure at LINE " << lineNumber << ": " << status << std::endl;
//     FatalError();
//   }
// }

PointCloud::Ptr CloudMem(const std::string &file_name, float *voxel_grid_color, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                         float voxel_size, float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
                         float *voxel_grid_TSDF, float *voxel_grid_weight,
                         float tsdf_thresh, float weight_thresh)
{

  PointCloud::Ptr cloud(new PointCloud);

  // Count total number of points in point cloud
  int num_pts = 0;
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++)
    if (std::abs(voxel_grid_TSDF[i]) < tsdf_thresh && voxel_grid_weight[i] > weight_thresh)
      num_pts++;

  // Create header for .ply file
  double voxel = 10;
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
      cv::Vec3f ves;
      // Convert voxel indices to float, and save coordinates to ply file
      // float pt_base_x
      PointT p;
      p.x = voxel_grid_origin_x + (float)x * voxel_size;
      p.y = voxel_grid_origin_y + (float)y * voxel_size;
      p.z = voxel_grid_origin_z + (float)z * voxel_size;
      // 从rgb图像中获取它的颜色
      // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
      color co;
      co.val = voxel_grid_color[i];
      p.b = co.rgb[0];
      p.g = co.rgb[1];
      p.r = co.rgb[2];
      // cout << (int)co.rgb[0] << " " <<(int) co.rgb[1] << " " << (int)co.rgb[2] << endl;
      // 把p加入到点云中
      cloud->points.push_back(p);

      // Eigen::Vector3f center(p.x, p.y, p.z);
      // Eigen::Quaternionf rotation(1, 0, 0, 0);
      // string cube = "cude" + to_string(i);
    }
  }
  return cloud;
}