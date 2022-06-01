// ---------------------------------------------------------
// Author: Andy Zeng, Princeton University, 2016
// ---------------------------------------------------------

#include <vector>
#include <opencv2/opencv.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <pcl/io/ply_io.h>

using namespace std;
// 定义点云类型

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;
union color
{
  uint8_t rgb[4];
  int val;

  color()
  {
    val = 0;
  }
};

// Compute surface points from TSDF voxel grid and save points to point cloud file
void SaveVoxelGrid2SurfacePointCloud(const std::string &file_name, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                                     float voxel_size, float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
                                     float *voxel_grid_TSDF, float *voxel_grid_weight,
                                     float tsdf_thresh, float weight_thresh);
PointCloud::Ptr CloudMem(const std::string &file_name, float *voxel_grid_color, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                                                   float voxel_size, float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
                                                   float *voxel_grid_TSDF, float *voxel_grid_weight,
                                                   float tsdf_thresh, float weight_thresh);

// Load an M x N matrix from a text file (numbers delimited by spaces/tabs)
// Return the matrix as a float vector of the matrix in row-major order
std::vector<float> LoadMatrixFromFile(std::string filename, int M, int N);

// Read a depth image with size H x W and save the depth values (in meters) into a float array (in row-major order)
// The depth image file is assumed to be in 16-bit PNG format, depth in millimeters
void ReadDepth(std::string filename, int H, int W, float *depth);
void ReadRGB(std::string filename, int H, int W, int *depth);
// 4x4 matrix multiplication (matrices are stored as float arrays in row-major order)
void multiply_matrix(const float m1[16], const float m2[16], float mOut[16]);

// 4x4 matrix inversion (matrices are stored as float arrays in row-major order)
bool invert_matrix(const float m[16], float invOut[16]);
void LoadMatrixFromFile(std::string filename, double *matrix, int M, int N);