
#pragma once

#include <cuda_runtime_api.h>

struct Volume
{
    struct Vovel
    {
        float tsdf;
        float weight;
#ifdef USE_COLOR
        uchar3 color;
#endif
    };
    typedef struct Vovel CudaData;

    float3 center;    //中心坐标
    uint3 size;       //体素格子尺寸
    CudaData *m_data; //数据首指针
    float sca;

    Volume(uint3 size);
    void save();
    __device__ inline Volume::CudaData &operator()(int x, int y, int z);
    __device__ inline void set(int x, int y, int z, const Volume::CudaData &val);
    __device__ inline float3 getWorld(int3 pos_vol);
    //世界坐标
    __device__ inline Volume::CudaData fetch(const float3 &p, bool &exi) const;
    __device__ float interpolate(const float3 &p_voxels);
    __host__ void savefile();
};
