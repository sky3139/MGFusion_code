#include "imgproc.cuh"

namespace device
{

    __global__ void
    mcp(uint64_t val, unsigned int inval, struct exmatcloud_para *devpa)
    {
        // devpa->center.u64 = val;
        // devpa->dev_points_num=inval;
    }
    __global__ void bilateral_kernel(const PtrStepSz<unsigned short> src, PtrStepSz<unsigned short> dst, const int ksz, const float sigma_spatial2_inv_half, const float sigma_depth2_inv_half)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= src.cols || y >= src.rows)
            return;

        int value = src.ptr(y)[x];

        int tx = min(x - ksz / 2 + ksz, src.cols - 1);
        int ty = min(y - ksz / 2 + ksz, src.rows - 1);

        float sum1 = 0;
        float sum2 = 0;

        for (int cy = max(y - ksz / 2, 0); cy < ty; ++cy)
        {
            for (int cx = max(x - ksz / 2, 0); cx < tx; ++cx)
            {
                int depth = src.ptr(y)[x];

                float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                float color2 = (value - depth) * (value - depth);

                float weight = __expf(-(space2 * sigma_spatial2_inv_half + color2 * sigma_depth2_inv_half));

                sum1 += depth * weight;
                sum2 += weight;
            }
        }
        dst.ptr(y)[x] = __float2int_rn(sum1 / sum2);
    }
    void bilateralFilter(const PtrStepSz<unsigned short> &src, const PtrStepSz<unsigned short> &dst, int kernel_size,
                         float sigma_spatial, float sigma_depth){
        // sigma_depth *= 1000; // meters -> mm

        // dim3 block (32, 8);
        // dim3 grid (divUp (src.cols(), block.x), divUp (src.rows (), block.y));

        // cudaSafeCall( cudaFuncSetCacheConfig (bilateral_kernel, cudaFuncCachePreferL1) );
        // bilateral_kernel<<<grid, block>>>(src, dst, kernel_size, 0.5f / (sigma_spatial * sigma_spatial), 0.5f / (sigma_depth * sigma_depth));
        // cudaSafeCall ( cudaGetLastError () );
    };

    __global__ void
    scaleDepthCloud(const PtrStepSz<unsigned short> depth, PtrStep<xyzPoints> scaled, float *campose, const Intr intr)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= depth.cols || y >= depth.rows)
            return;
        int Dp = depth.ptr(y)[x];

        xyzPoints xp;
        xp.z = (float)Dp * 0.001f;
        xp.x = xp.z * (x - 320) / 500;
        xp.y = xp.z * (y - 240) / 500;

        // float xl = (x - intr.cx) / intr.fx;
        // float yl = (y - intr.cy) / intr.fy;
        // float lambda = sqrtf(xl * xl + yl * yl + 1);

        xyzPoints gp;

        __shared__ float cam[16];

        if (0 == threadIdx.x && threadIdx.y == 0)
        {
            for (int i = 0; i < 16; i++)
            {
                cam[i] = campose[i];
            }
        }
        __syncthreads();

        gp.x = cam[0 * 4 + 0] * xp.x + cam[0 * 4 + 1] * xp.y + cam[0 * 4 + 2] * xp.z + cam[0 * 4 + 3];
        gp.y = cam[1 * 4 + 0] * xp.x + cam[1 * 4 + 1] * xp.y + cam[1 * 4 + 2] * xp.z + cam[1 * 4 + 3];
        gp.z = cam[2 * 4 + 0] * xp.x + cam[2 * 4 + 1] * xp.y + cam[2 * 4 + 2] * xp.z + cam[2 * 4 + 3];
        scaled.ptr(y)[x] = gp; // Dp * 0.0002f; //  / 1000.f; //meters

        // u32_4byte vecs;
        // int8_t vx= std::floor( gp.x / (float)(VOXELSIZE)); //向下取整
        // int8_t vy= std::floor( gp.y  / (float)(VOXELSIZE)); //向下取整
        // int8_t vz= std::floor( gp.z/ (float)(VOXELSIZE)); //向下取整

        // vecs.x =(int8_t)vx;
        // vecs.y =(int8_t)vy;
        // vecs.z =(int8_t)vz;
    }

    __global__ void
    mappoints_cube2tsdf_batch(struct box32 *dev_pbox, struct ex_buf_ *point_buf)
    {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; // z blockId
        int pt_grid_x = threadIdx.x;
        int pt_grid_y = threadIdx.y;
        u32_4byte index = dev_pbox[blockId].index;
        u64_4byte center = point_buf[blockId].center;

        for (int pt_grid_z = 0; pt_grid_z < 32; pt_grid_z++)
        {
            int volume_idx = pt_grid_z * CUBEVOXELSIZE * CUBEVOXELSIZE + pt_grid_y * CUBEVOXELSIZE + pt_grid_x;
            union voxel &voxel = dev_pbox[blockId].pVoxel[volume_idx];
            if (std::abs(voxel.tsdf) < 0.2f && voxel.weight > 0.0f) // || (pt_grid_x == 0 && pt_grid_y == 0 && pt_grid_z == 0))
                                                                    // if (pt_grid_x % 5 == 0 && pt_grid_y % 5 == 0 && pt_grid_z % 5 == 0)
            {

                unsigned int val = atomicInc(&point_buf[blockId].dev_points_num, 0xffffff);

                union UPoints &up = point_buf[blockId].up[val];

                up.xyz[0] = (index.x + 1 * center.x) * VOXELSIZE + pt_grid_x * VOXELSIZE_PCUBE;
                up.xyz[1] = (index.y + 1 * center.y) * VOXELSIZE + pt_grid_y * VOXELSIZE_PCUBE;
                up.xyz[2] = (index.z + 1 * center.z) * VOXELSIZE + pt_grid_z * VOXELSIZE_PCUBE;

                up.rgb[0] = voxel.rgb[0];
                up.rgb[1] = voxel.rgb[1];
                up.rgb[2] = voxel.rgb[2];
            }
        }
    }

    __global__ void
    mappoints_cube2tsdf(struct box32 *dev_pbox, struct ex_buf_ *point_buf)
    {
        int pt_grid_z = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; // z blockId
        int pt_grid_x = threadIdx.x;
        int pt_grid_y = threadIdx.y;
        u32_4byte index = dev_pbox->index;
        u64_4byte center = point_buf->center;
        int volume_idx = pt_grid_z * CUBEVOXELSIZE * CUBEVOXELSIZE + pt_grid_y * CUBEVOXELSIZE + pt_grid_x;
        union voxel &voxel = dev_pbox->pVoxel[volume_idx];
        // if (voxel.weight < 0.001f)
        // {
        //     continue;
        // }
        if (std::abs(voxel.tsdf) < 0.2f && voxel.weight > 0.0f) // || (pt_grid_x == 0 && pt_grid_y == 0 && pt_grid_z == 0))
                                                                // if (pt_grid_x % 5 == 0 && pt_grid_y % 5 == 0 && pt_grid_z % 5 == 0)
        {

            unsigned int val = atomicInc(&point_buf->dev_points_num, 0xffffff);

            union UPoints &up = point_buf->up[val];

            up.xyz[0] = (index.x + 1 * center.x) * VOXELSIZE + pt_grid_x * VOXELSIZE_PCUBE;
            up.xyz[1] = (index.y + 1 * center.y) * VOXELSIZE + pt_grid_y * VOXELSIZE_PCUBE;
            up.xyz[2] = (index.z + 1 * center.z) * VOXELSIZE + pt_grid_z * VOXELSIZE_PCUBE;

            up.rgb[0] = voxel.rgb[0];
            up.rgb[1] = voxel.rgb[1];
            up.rgb[2] = voxel.rgb[2];

            // points.push_back(vec);
            // color.push_back(cv::Vec3b(voxel.rgb[0], voxel.rgb[1], voxel.rgb[2]));
        }
    }
    __global__ void
    cloud2grids(struct box32 *pboxmap, UPoints *ps, size_t len_point)
    {
        int pt_grid_z = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        union UPoints &up = ps[pt_grid_z];
        u32_4byte u64;
        u64.x = std::floor(3.125f * up.xyz[0]);
        u64.y = std::floor(3.125f * up.xyz[1]);
        u64.z = std::floor(3.125f * up.xyz[2]);

        u32_4byte u32;
        u32.x = (up.xyz[0] - u64.x * 0.32f) * 100;
        u32.y = (up.xyz[1] - u64.y * 0.32f) * 100;
        u32.z = (up.xyz[2] - u64.z * 0.32f) * 100;
        assert(u32.x < 32);
        assert(u32.y < 32);
        assert(u32.z < 32);
        int index = u32.x + u32.y * 32 + u32.z * 32 * 32;
        pboxmap->index = u64;
        pboxmap->pVoxel[index].tsdf = 0.0f;
        pboxmap->pVoxel[index].weight = 50.0f;
        pboxmap->pVoxel[index].rgb[0] = up.rgb[0];
        pboxmap->pVoxel[index].rgb[1] = up.rgb[1];
        pboxmap->pVoxel[index].rgb[2] = up.rgb[2];
    }
    __global__ void
    cloud2grids_init(struct box32 *pboxmap)
    {
        int pt_grid_z = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        pboxmap->index.u32 = 0;
        pboxmap->pVoxel[pt_grid_z].tsdf = 1.0f;
        pboxmap->pVoxel[pt_grid_z].weight = 0;
    }
    __global__ void
    scaleDepth(const PtrStepSz<unsigned short> depth, PtrStep<xyzPoints> scaled, PtrStep<xyzPoints> gcloud,
               PtrStep<u32_4byte> zin, float *campose, const Intr intr)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= depth.cols || y >= depth.rows)
            return;
        int Dp = depth.ptr(y)[x];

        xyzPoints xp;
        xp.z = Dp * 0.0002f;
        xp.x = xp.z * (x - intr.cx) / intr.fx;
        xp.y = xp.z * (y - intr.cy) / intr.fy;

        // float xl = (x - intr.cx) / intr.fx;
        // float yl = (y - intr.cy) / intr.fy;
        // float lambda = sqrtf(xl * xl + yl * yl + 1);
        scaled.ptr(y)[x] = xp; // Dp * 0.0002f; //  / 1000.f; //meters

        xyzPoints gp;

        __shared__ float cam[16];

        if (0 == threadIdx.x && threadIdx.y == 0)
        {
            for (int i = 0; i < 16; i++)
            {
                cam[i] = campose[i];
            }
        }
        __syncthreads();

        gp.x = cam[0 * 4 + 0] * xp.x + cam[0 * 4 + 1] * xp.y + cam[0 * 4 + 2] * xp.z + cam[0 * 4 + 3];
        gp.y = cam[1 * 4 + 0] * xp.x + cam[1 * 4 + 1] * xp.y + cam[1 * 4 + 2] * xp.z + cam[1 * 4 + 3];
        gp.z = cam[2 * 4 + 0] * xp.x + cam[2 * 4 + 1] * xp.y + cam[2 * 4 + 2] * xp.z + cam[2 * 4 + 3];

        gcloud.ptr(y)[x] = gp; // Dp * 0.0002f; //  / 1000.f; //meters

        u32_4byte vecs;
        int8_t vx = std::floor (gp.x / (float)(VOXELSIZE)); //向下取整 __float2int_rn __float2int_rd rn是求最近的偶数，rz是逼近零，ru是向上舍入[到正无穷]，rd是向下舍入[到负无穷]。  std::floor
        int8_t vy = std::floor(gp.y / (float)(VOXELSIZE)); //向下取整
        int8_t vz = std::floor(gp.z / (float)(VOXELSIZE)); //向下取整

        vecs.x = (int8_t)vx;
        vecs.y = (int8_t)vy;
        vecs.z = (int8_t)vz;

        zin.ptr(y)[x] = vecs;
    }

    //写论文测试用
    __global__ void
    extract_kernel_test(Point3dim *output_base, struct box32 *dev_pbox, exmatcloud_para *para)
    {
        // printf("d:%d ,%f\n", *pos_index, device::devData);
        // device::devData += 2;
        // *pos_index += 2;
        int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        // output_base = output;

        if ((dev_pbox + blockId) == NULL)
            return;

        struct box32 &vdev_pbox = dev_pbox[blockId];
        int pt_grid_x = threadIdx.x;
        int pt_grid_y = threadIdx.y;
        u32_4byte index = vdev_pbox.index;

        for (int pt_grid_z = 0; pt_grid_z < 32; pt_grid_z++)
        {
            int volume_idx = pt_grid_z * 32 * 32 + pt_grid_y * 32 + pt_grid_x;
            union voxel &voxel = vdev_pbox.pVoxel[volume_idx];

            unsigned int val = atomicInc(&para->dev_points_num, 0xffffff);
            Point3dim &pos = output_base[val];
            pos.x = (index.x + para->center.x) * 0.32f + pt_grid_x * 0.01f;
            pos.y = (index.y + para->center.y) * 0.32f + pt_grid_y * 0.01f;
            pos.z = (index.z + para->center.z) * 0.32f + pt_grid_z * 0.01f;

            if (std::abs(voxel.tsdf) < 0.2f && voxel.weight > 0)
            {
                pos.rgb[0] = voxel.rgb[0];
                pos.rgb[1] = voxel.rgb[1];
                pos.rgb[2] = voxel.rgb[2];
            }
            else
            {
                pos.rgb[0] = 0;
                pos.rgb[1] = 255;
                pos.rgb[2] = 255;
            }
        }
        if (vdev_pbox.index.cnt > 0 && pt_grid_x == 0 && pt_grid_y == 0)
        {
            vdev_pbox.index.cnt--;
            // printf("aaaaaaa %d\n",vdev_pbox.index.cnt);//
        }
    }
    __global__ void
    extract_activate_kernel(Point3dim *output_base, struct box32 *dev_pbox, exmatcloud_para *para)
    {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        if ((dev_pbox + blockId) == NULL)
            return;

        struct box32 &vdev_pbox = dev_pbox[blockId];

        u32_4byte index = vdev_pbox.index;
        if (index.cnt != 0)
        {
            return;
        }
        int pt_grid_x = threadIdx.x;
        int pt_grid_y = threadIdx.y;

        for (int pt_grid_z = 0; pt_grid_z < 32; pt_grid_z++)
        {
            int volume_idx = pt_grid_z * 32 * 32 + pt_grid_y * 32 + pt_grid_x;
            union voxel &voxel = vdev_pbox.pVoxel[volume_idx];
            if (std::abs(voxel.tsdf) < 0.2f && voxel.weight > 0.50f)
            {
                unsigned int val = atomicInc(&para->dev_points_num, 0xffffff);
                Point3dim &pos = output_base[val];
                pos.x = (index.x + para->center.x) * 0.32f + pt_grid_x * 0.01f;
                pos.y = (index.y + para->center.y) * 0.32f + pt_grid_y * 0.01f;
                pos.z = (index.z + para->center.z) * 0.32f + pt_grid_z * 0.01f;

                pos.rgb[0] = voxel.rgb[0];
                pos.rgb[1] = voxel.rgb[1];
                pos.rgb[2] = voxel.rgb[2];
            }
        }
    }

    // __global__ void
    // kernel_change_type(struct box32 *dev_pbox, uint8_t val)
    // {
    //     int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    //     if ((dev_pbox + blockId) == NULL)
    //         return;
    //     struct box32 &vdev_pbox = dev_pbox[blockId];
    //     // int pt_grid_x = threadIdx.x;
    //     // int pt_grid_y = threadIdx.y;
    //     vdev_pbox.index.type = val;
    // }
    __global__ void
    extract_kernel(ex_buf *output_base, struct box32 *dev_pbox, exmatcloud_para *para)
    {
        // printf("d:%d ,%f\n", *pos_index, device::devData);
        // device::devData += 2;
        // *pos_index += 2;
        int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        // __shared__  Point3dim *output_base; //
        // output_base = output;

        if ((dev_pbox + blockId) == NULL)
            return;

        struct box32 &vdev_pbox = dev_pbox[blockId];
        int pt_grid_x = threadIdx.x;
        int pt_grid_y = threadIdx.y;
        u32_4byte index = vdev_pbox.index;
        int8_t asdasd = vdev_pbox.index.cnt;

        for (int pt_grid_z = 0; pt_grid_z < 32; pt_grid_z++)
        {
            int volume_idx = pt_grid_z * 32 * 32 + pt_grid_y * 32 + pt_grid_x;
            union voxel &voxel = vdev_pbox.pVoxel[volume_idx];
            if (std::abs(voxel.tsdf) < 0.2f && voxel.weight > 0.50f)
            {
                unsigned int val = atomicInc(&para->dev_points_num, 0xffffff);
                struct posevoxel &pos = output_base->pose[val];
                struct colorvoxel &_color = output_base->color[val];

                pos.x = (index.x + para->center.x) * 0.32f + pt_grid_x * 0.01f;
                pos.y = (index.y + para->center.y) * 0.32f + pt_grid_y * 0.01f;
                pos.z = (index.z + para->center.z) * 0.32f + pt_grid_z * 0.01f;

                if (vdev_pbox.index.cnt == 0)
                {
                    _color.rgb[0] = voxel.rgb[0];
                    _color.rgb[1] = voxel.rgb[1];
                    _color.rgb[2] = voxel.rgb[2];
                }
                // pos.rgb[0] = 255;//voxel.rgb[0];
                // pos.rgb[1] = 255;//index.x;//voxel.rgb[1];
                // pos.rgb[2] = 255;//index.x;//voxel.rgb[2];
                //    "     }
                else if (vdev_pbox.index.cnt > 5)
                {
                    _color.rgb[0] = 255; // voxel.rgb[0];
                    _color.rgb[1] = 0;   // index.x;//voxel.rgb[1];
                    _color.rgb[2] = 0;   // index.x;//voxel.rgb[2];
                }
                else // if (index.type == 2)
                {
                    _color.rgb[0] = 0;   // voxel.rgb[0];
                    _color.rgb[1] = 255; // index.x;//voxel.rgb[1];
                    _color.rgb[2] = 0;   // index.x;//voxel.rgb[2];
                }
            }
        }
        if (asdasd > 0 && pt_grid_x == 0 && pt_grid_y == 0)
        {
            //         if (asdasd != 0)
            // printf("%d\n", asdasd);
            vdev_pbox.index.cnt--;
            // printf("aaaaaaa %d\n",vdev_pbox.index.cnt);//
        }
    }

    // __global__ void
    // scaleDepth(const PtrStepSz<unsigned short> depth, PtrStep<xyzPoints> scaled, const Intr intr)
    // {
    //     int x = threadIdx.x + blockIdx.x * blockDim.x;
    //     int y = threadIdx.y + blockIdx.y * blockDim.y;

    //     if (x >= depth.cols || y >= depth.rows)
    //         return;
    //     int Dp = depth.ptr(y)[x];

    //     xyzPoints xp;
    //     xp.z=Dp * 0.0002f;
    //     xp.x= xp.z*(x - intr.cx) / intr.fx;
    //     xp.y=xp.z*(y - intr.cy) / intr.fy;

    //     // float xl = (x - intr.cx) / intr.fx;
    //     // float yl = (y - intr.cy) / intr.fy;
    //     // float lambda = sqrtf(xl * xl + yl * yl + 1);

    //     scaled.ptr(y)[x] =xp;// Dp * 0.0002f; //  / 1000.f; //meters
    // }
    __global__ void Integrate32(float *cam_K,
                                int im_height, int im_width,
                                float voxel_size, float trunc_margin,
                                struct box32 **dev_boxptr, struct kernelPara *gpu_kpara,
                                const PtrStepSz<xyzPoints> depthScaled)
    {
        // int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        int pt_grid_z = threadIdx.x; // threadId 32
        int pt_grid_y = threadIdx.y; // threadId 32
        // int pt_grid_num = gridDim.x; //num
        int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        __shared__ float cam2base[16];
        // printf("%d %d %d\n", pt_grid_y, pt_grid_z, blockId);
        if (0 == pt_grid_z && pt_grid_y == 0)
        {
            for (int i = 0; i < 16; i++)
                cam2base[i] = gpu_kpara->cam2base[i];
        }
        __syncthreads();

        struct box32 *pbox = dev_boxptr[blockId];
        union u32_4byte u32 = pbox->index;
        for (int pt_grid_x = 0; pt_grid_x < CUBEVOXELSIZE; pt_grid_x++)
        {
            // 计算小体素的世界坐标
            float pt_base_x = (u32.x + gpu_kpara->center.x) * VOXELSIZE + pt_grid_x * voxel_size;
            float pt_base_y = (u32.y + gpu_kpara->center.y) * VOXELSIZE + pt_grid_y * voxel_size;
            float pt_base_z = (u32.z + gpu_kpara->center.z) * VOXELSIZE + pt_grid_z * voxel_size;

            //     //计算体素在相机坐标系的坐标
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

            // uint16_t img_depu = gpu_kpara->dev_depthdata[pt_pix_y * im_width + pt_pix_x];
            // union points *pt = (union points *)&point[pt_pix_y * im_width + pt_pix_x];
            xyzPoints img_dep = depthScaled.ptr(pt_pix_y)[pt_pix_x]; // meters

            // float img_dep = img_depu * 0.001f; //pt->xyz;//
            if (img_dep.z <= 0 || img_dep.z > 6)
                continue;
            float diff = img_dep.z - pt_cam_z;
            if (diff <= -trunc_margin)
                continue;
            uint32_t vox_index = pt_grid_x + pt_grid_y * CUBEVOXELSIZE + pt_grid_z * CUBEVOXELSIZE * CUBEVOXELSIZE;

            union voxel &p = pbox->pVoxel[vox_index];

            float dist = fmin(1.0f, diff / trunc_margin);
            // float weight_old = p.weight;//> 128 ? 128 : p.weight;
            // float weight_new = weight_old + 1.0f;
            float weight_old = (float)p.weight; //> 128 ? 128 : p.weight;
            p.weight = p.weight > 250 ? p.weight : p.weight + 1;
            float weight_new = (float)p.weight;
            p.tsdf = (p.tsdf * weight_old + dist) / weight_new;

            uint8_t *rgb_val = &gpu_kpara->dev_rgbdata[pt_pix_y * im_width + pt_pix_x][0]; // pt->rgb[0];

            uint16_t mval = (p.rgb[0] * weight_old + rgb_val[0]) / weight_new;
            p.rgb[0] = mval > 255 ? 255 : mval;
            mval = (p.rgb[1] * weight_old + rgb_val[1]) / weight_new;
            p.rgb[1] = mval > 255 ? 255 : mval;
            mval = (p.rgb[2] * weight_old + rgb_val[2]) / weight_new;
            p.rgb[2] = mval > 255 ? 255 : mval;
        }
        // }
    }

}