
// #include "volume.h"

// #include <fstream>
// #include <cuda_runtime_api.h>
// // #include "utils/cutil_math.h"
//  #include <math_constants.h>
// Volume::Volume(uint3 size) : size(size)
// {
//     cudaMalloc((void **)&m_data, sizeof(Volume::CudaData) * size.x * size.y * size.z);
//     cudaMemset((void **)&m_data, 0, sizeof(Volume::CudaData) * size.x * size.y * size.z);
//     sca = 0.01f;
// }
// //获取指定网格坐标的体素
// __device__ inline Volume::CudaData &Volume::operator()(int x, int y, int z)
// {
//     // assert(x < size.x && y < size.y && z < size.z);
//     return m_data[x + size.x * y + z * size.x * size.y];
// }
// //设置
// __device__ inline void Volume::set(int x, int y, int z, const Volume::CudaData &val)
// {
//     m_data[x + size.x * y + z * size.x * size.y]= val;
// }
// __device__ inline float3 Volume::getWorld(int3 pos_vol)
// {

//     return pos_vol * sca + center;
// }
// //世界坐标
// __device__ inline   Volume::CudaData Volume::fetch(const float3 &p, bool &exi) const
// {
//     // rounding to nearest even
//     float3 intpose = (p - center) / sca;

//     int x = __float2int_rn(intpose.x);
//     int y = __float2int_rn(intpose.y);
//     int z = __float2int_rn(intpose.z);
//     if (x >= size.x || y >= size.y || z >= size.z || x < 0 || y < 0 || z < 0)
//     {
//         exi = false;
//         return m_data[0];
//     }
//     else
//         exi = true;
//     return   m_data[x + size.x * y + z * size.x * size.y];

//     // printf("%d,%d,%d\n",x,y,z);
// }
// __device__ float Volume::interpolate(const float3 &p_voxels)
// {
//     float3 cf = p_voxels;

//     // // rounding to negative infinity
//     int3 g = make_int3(__float2int_rd(cf.x), __float2int_rd(cf.y), __float2int_rd(cf.z));

//     if (g.x < 0 || g.x >= size.x - 1 || g.y < 0 || g.y >= size.y - 1 || g.z < 0 || g.z >= size.z - 1)
//         return CUDART_NAN_F;

//     float a = cf.x - g.x;
//     float b = cf.y - g.y;
//     float c = cf.z - g.z;

//     float tsdf = 0.f;
//     tsdf += (this->operator()(g.x + 0, g.y + 0, g.z + 0)).tsdf * (1 - a) * (1 - b) * (1 - c);
//     tsdf += (this->operator()(g.x + 0, g.y + 0, g.z + 1)).tsdf * (1 - a) * (1 - b) * c;
//     tsdf += (this->operator()(g.x + 0, g.y + 1, g.z + 0)).tsdf * (1 - a) * b * (1 - c);
//     tsdf += (this->operator()(g.x + 0, g.y + 1, g.z + 1)).tsdf * (1 - a) * b * c;
//     tsdf += (this->operator()(g.x + 1, g.y + 0, g.z + 0)).tsdf * a * (1 - b) * (1 - c);
//     tsdf += (this->operator()(g.x + 1, g.y + 0, g.z + 1)).tsdf * a * (1 - b) * c;
//     tsdf += (this->operator()(g.x + 1, g.y + 1, g.z + 0)).tsdf * a * b * (1 - c);
//     tsdf += (this->operator()(g.x + 1, g.y + 1, g.z + 1)).tsdf * a * b * c;
//     return tsdf;
// }
// __host__ void Volume::savefile()
// {
//     // std::fstream f("tsdf.bin", std::ios::out);
//     // struct _Vovel *phost;
//     // cudaMallocHost((void **)&phost, sizeof(struct _Vovel));
//     // cudaMemcpy(phost, m_data, sizeof(struct _Vovel), cudaMemcpyDeviceToHost);
//     // for (size_t i = 0; i < 512; i++)
//     //     for (size_t j = 0; j < 512; j++)
//     //         for (size_t k = 0; k < 512; k++)
//     //         {
//     //             f.write((const char *)&phost->m_data[i][j][k].tsdf, 8);
//     //             // if (phost->m_data[i][j][k].weight > 0.3)
//     //             //     printf("a\n");
//     //         }
//     // // cudaFreeHost(phost);
//     // f.close();
//     // assert(0);
// }

// void Volume::save()
// {
//     // _Vovel *cpu;
//     // cudaMallocHost((void **)&cpu, sizeof(struct _Vovel));
//     // ck(cudaMemcpy(cpu, m_data, sizeof(struct _Vovel), cudaMemcpyDeviceToHost));
//     // std::fstream f("t.bin", std::ios::out);
//     // assert(f.is_open());
//     // f.write((const char *)cpu, sizeof(struct _Vovel));
//     // cout << sizeof(struct _Vovel) << endl;
// }
// int main()
// {
    
// }