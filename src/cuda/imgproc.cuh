#pragma once
#include <cuda_runtime_api.h>

#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>


#include "device_array.hpp"
#include "../app/cuVector.cuh"
#include "datatype.cuh"
namespace device
{
    //  void depthBilateralFilter(const Depth& in, Depth& out, int ksz, float sigma_spatial, float sigma_depth);

    //  void depthTruncation(Depth& depth, float threshold);

    //  void depthBuildPyramid(const Depth& depth, Depth& pyramid, float sigma_depth);
    __global__ void
    mappoints_cube2tsdf(struct Voxel32 *dev_pbox, struct ex_buf_ *point_buf);
    //  void computeNormalsAndMaskDepth(const Intr& intr, Depth& depth, Normals& normals);
    __global__ void
    mcp(uint64_t val, unsigned int inval, struct exmatcloud_para *devpa);
    //  void computePointNormals(const Intr& intr, const Depth& depth, Cloud& points, Normals& normals);
    __global__ void compute_dists_kernel();
    void computeDists(const ushort *depth[], ushort *dists[], float2 finv, float2 c);
    __global__ void
    scaleDepth( uint32_t *kset,const Patch<unsigned short> depth, Patch<PosType> scaled, Patch<PosType> gcloud,
               const Intr intr, struct kernelPara gpu_kpara);
    __host__ void bilateralFilter(const Patch<unsigned short> &src, const Patch<unsigned short> &dst, int kernel_size,
                                  float sigma_spatial, float sigma_depth);
    __global__ void Integrate32(float4 intr,
                                int im_height, int im_width,
                                float voxel_size, float trunc_margin,
                                struct Voxel32 **dev_boxptr, struct kernelPara *gpu_kpara,
                                const Patch<PosType> depthScaled);

    __global__ void scaleDepthCloud(const PtrStepSz<unsigned short> depth, PtrStep<PosType> scaled, float *campose, const Intr intr);
    __global__ void
    mappoints_cube2tsdf_batch(struct Voxel32 *dev_pbox, struct ex_buf_ *point_buf);
    __global__ void bilateral_kernel(const Patch<unsigned short> src, Patch<unsigned short> dst, const int ksz, const float sigma_spatial2_inv_half, const float sigma_depth2_inv_half);
    __global__ void
    cloud2grids(struct Voxel32 *dev_pbox, UPoints *ps, size_t len_point);
    __global__ void
    cloud2grids_init(struct Voxel32 *pboxmap);

    __global__ void
    extract_kernel(ex_buf *output_base, CUVector<struct Voxel32 *> g_use, exmatcloud_para *para);
    __global__ void
    extract_kernel_test(Point3dim *output_base, struct Voxel32 *dev_pbox, exmatcloud_para *para);

    __global__ void
    kernel_change_type(struct Voxel32 *dev_pbox, uint8_t val);

    __global__ void
    extract_kernel(ex_buf *output_base, struct Voxel32 *dev_pbox, exmatcloud_para *para);
    __global__ void update_loacl_index(struct Voxel32 **pboxmap, u64B4 src_center, u64B4 now_center, u32B4 *srcid, u32B4 *nowid, bool *mask);
    __global__ void Integrate32F(struct Tnte fun, struct Voxel32 **dev_boxptr, struct kernelPara gpu_kpara,
                                 const Patch<PosType> depthScaled);

    struct Tnte
    {
        float4 intr;
        int im_height;
        int im_width;
        float voxel_size;
        float trunc_margin;
        __device__ void operator()(
            struct Voxel32 **&dev_boxptr, struct kernelPara &gpu_kpara,
            const Patch<PosType> &depthScaled);
    };
    //  void cloudToDepth(const Cloud& cloud, Depth& depth);

    //  void resizeDepthNormals(const Depth& depth, const Normals& normals, Depth& depth_out, Normals& normals_out);

    //  void resizePointsNormals(const Cloud& points, const Normals& normals, Cloud& points_out, Normals& normals_out);

    //  void waitAllDefaultStream();

    //  void renderTangentColors(const Normals& normals, Image& image);

    //  void renderImage(const Depth& depth, const Normals& normals, const Intr& intr, const Vec3f& light_pose, Image& image);

    //  void renderImage(const Cloud& points, const Normals& normals, const Intr& intr, const Vec3f& light_pose, Image& image);
}
