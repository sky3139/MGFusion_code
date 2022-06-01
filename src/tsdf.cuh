#ifndef FOO_CUH
#define FOO_CUH

#include "tool/tool.h"
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <vector>

#define checkCUDA(res)                                                          \
    {                                                                           \
        if (res != cudaSuccess)                                                 \
        {                                                                       \
            printf("Error ：%s:%d , ", __FILE__, __LINE__);                     \
            printf("code : %d , reason : %s \n", res, cudaGetErrorString(res)); \
            exit(-1);                                                           \
        }                                                                       \
    }
#define ck(val)                                                                                \
    {                                                                                          \
        if (val != cudaSuccess)                                                                \
                                                                                               \
        {                                                                                      \
            printf("Error ：%s:%d , ", __FILE__, __LINE__);                                    \
            printf("code : %d , reason : %s \n", cudaGetLastError(), cudaGetErrorString(val)); \
            exit(-1);                                                                          \
        }                                                                                      \
    }

struct TSDF;

// Read a depth image with size H x W and save the depth values (in meters) into a float array (in row-major order)
// The depth image file is assumed to be in 16-bit PNG format, depth in millimeters

void ReadDepth(std::string filename, int H, int W, float *depth);

extern "C" void useCUDA();

__global__ void Integrate(float *cam_K, float *cam2base,
                          int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                          float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
                          union voxel *_TSDF, union points *point);

#endif

// __host__ int mainrun(void);
// // Save TSDF voxel grid and its parameters to disk as binary file (float array)
// std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;
// std::string voxel_grid_saveto_path = "tsdf.bin";
// std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
// float voxel_grid_dim_xf = (float)voxel_grid_dim_x;
// float voxel_grid_dim_yf = (float)voxel_grid_dim_y;
// float voxel_grid_dim_zf = (float)voxel_grid_dim_z;
// outFile.writcmakee((char *)&voxel_grid_dim_xf, sizeof(float));
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