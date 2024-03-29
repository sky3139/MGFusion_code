#include "imgproc.cuh"

namespace device
{

    __global__ void
    mcp(uint64_t val, unsigned int inval, struct exmatcloud_para *devpa)
    {
        // devpa->center.u64 = val;
        // devpa->dev_points_num=inval;
    }
    __global__ void bilateral_kernel(const Patch<unsigned short> src, Patch<unsigned short> dst, const int ksz, const float sigma_spatial2_inv_half, const float sigma_depth2_inv_half)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= src.cols || y >= src.rows)
            return;

        int value = src(y, x);

        int tx = min(x - ksz / 2 + ksz, src.cols - 1);
        int ty = min(y - ksz / 2 + ksz, src.rows - 1);

        float sum1 = 0;
        float sum2 = 0;

        for (int cy = max(y - ksz / 2, 0); cy < ty; ++cy)
        {
            for (int cx = max(x - ksz / 2, 0); cx < tx; ++cx)
            {
                int depth = src(y, x);

                float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                float color2 = (value - depth) * (value - depth);

                float weight = __expf(-(space2 * sigma_spatial2_inv_half + color2 * sigma_depth2_inv_half));

                sum1 += depth * weight;
                sum2 += weight;
            }
        }
        dst(y, x) = __float2int_rn(sum1 / sum2);
    }
    void bilateralFilter(const Patch<unsigned short> &src, const Patch<unsigned short> &dst, int kernel_size,
                         float sigma_spatial, float sigma_depth){
        // sigma_depth *= 1000; // meters -> mm

        // dim3 block (32, 8);
        // dim3 grid (divUp (src.cols(), block.x), divUp (src.rows (), block.y));

        // cudaSafeCall( cudaFuncSetCacheConfig (bilateral_kernel, cudaFuncCachePreferL1) );
        // bilateral_kernel<<<grid, block>>>(src, dst, kernel_size, 0.5f / (sigma_spatial * sigma_spatial), 0.5f / (sigma_depth * sigma_depth));
        // cudaSafeCall ( cudaGetLastError () );
    };

    __global__ void
    scaleDepthCloud(const Patch<unsigned short> depth, Patch<PosType> scaled, float *campose, const Intr intr)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x >= depth.cols || y >= depth.rows)
            return;
        int Dp = depth(y, x);
        float3 xp = intr.reprojector(x, y, Dp * 0.001f);
        PosType gp;
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
        scaled(y, x) = gp; // Dp * 0.0002f; //  / 1000.f; //meters
    }

    __global__ void
    mappoints_cube2tsdf_batch(struct Voxel32 *dev_pbox, struct ex_buf_ *point_buf)
    {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; // z blockId
        int pt_grid_x = threadIdx.x;
        int pt_grid_y = threadIdx.y;
        u32B4 index = dev_pbox[blockId].index;
        u64B4 center = point_buf[blockId].center;

        for (int pt_grid_z = 0; pt_grid_z < 32; pt_grid_z++)
        {
            int volume_idx = pt_grid_z * CUBEVOXELSIZE * CUBEVOXELSIZE + pt_grid_y * CUBEVOXELSIZE + pt_grid_x;
            union voxel &voxel = dev_pbox[blockId].pVoxel[volume_idx];
            if (std::abs(voxel.tsdf) < 0.2f && voxel.weight > 0.0f) // || (pt_grid_x == 0 && pt_grid_y == 0 && pt_grid_z == 0))
                                                                    // if (pt_grid_x % 5 == 0 && pt_grid_y % 5 == 0 && pt_grid_z % 5 == 0)
            {

                unsigned int val = atomicInc(&point_buf[blockId].dev_points_num, 0xffffff);

                UPoints &up = point_buf[blockId].up[val];

                up.pos.x = (index.x + 1 * center.x) * VOXELSIZE + pt_grid_x * VOXELSIZE_PCUBE;
                up.pos.y = (index.y + 1 * center.y) * VOXELSIZE + pt_grid_y * VOXELSIZE_PCUBE;
                up.pos.z = (index.z + 1 * center.z) * VOXELSIZE + pt_grid_z * VOXELSIZE_PCUBE;

                up.rgb[0] = voxel.rgb[0];
                up.rgb[1] = voxel.rgb[1];
                up.rgb[2] = voxel.rgb[2];
            }
        }
    }

    __global__ void
    mappoints_cube2tsdf(struct Voxel32 *dev_pbox, struct ex_buf_ *point_buf)
    {
        int pt_grid_z = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; // z blockId
        int pt_grid_x = threadIdx.x;
        int pt_grid_y = threadIdx.y;
        u32B4 index = dev_pbox->index;
        u64B4 center = point_buf->center;
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

            UPoints &up = point_buf->up[val];

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
    __global__ void update_loacl_index(struct Voxel32 **d_pbox_use, u64B4 dst, u64B4 now_center, u32B4 *srcid, u32B4 *nowid, bool *mask)
    {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;

        struct Voxel32 *temp = d_pbox_use[blockId];
        mask[blockId] = true;
        const u32B4 u32_src = temp->index;
        srcid[blockId] = temp->index;
        uint8_t now_int = temp->index.cnt;
        u32B4 u32new;
        u64B4 src_lo(u32_src);
        u64B4 now_local;

        now_local.x = src_lo.x + dst.x;
        now_local.y = src_lo.y + dst.y;
        now_local.z = src_lo.z + dst.z;
        if ((abs(now_local.x) > 126) || (abs(now_local.y) > 126) || (abs(now_local.z) > 126))
        {
            mask[blockId] = false; //转换成点云
            // temp->index = u32B4(0);
            nowid[blockId].u32 = -1;
            srcid[blockId].u32 = -1;
            return;
        }
        u32new.x = now_local.x, u32new.y = now_local.y, u32new.z = now_local.z;
        temp->index = u32new;
        // if (u32new.u32 == 0)
        // printf("u32_src %x+%lx-%lx=%lx %lx\n", u32new.u32, dst.u64, src_lo.u64, now_local.u64, now_local.u64);
        nowid[blockId] = u32new;
        // temp->index.cnt = now_int;
        if (now_int == 0)
        {
            mask[blockId] = false; //转换成点云
        }
    }
    __global__ void
    cloud2grids(struct Voxel32 *pboxmap, UPoints *ps, size_t len_point)
    {
        int pt_grid_z = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        UPoints &up = ps[pt_grid_z];
        u32B4 u64;
        u64.x = std::floor(3.125f * up.xyz[0]);
        u64.y = std::floor(3.125f * up.xyz[1]);
        u64.z = std::floor(3.125f * up.xyz[2]);

        u32B4 u32;
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
    cloud2grids_init(struct Voxel32 *pboxmap)
    {
        int pt_grid_z = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        pboxmap->index.u32 = 0;
        pboxmap->pVoxel[pt_grid_z].tsdf = 1.0f;
        pboxmap->pVoxel[pt_grid_z].weight = 0;
    }
    __global__ void
    scaleDepth(uint32_t *kset, const Patch<unsigned short> depth, Patch<PosType> scaled, Patch<PosType> gcloud,
               const Intr intr, struct kernelPara gpu_kpara, uint32_t *bitmap)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x >= depth.cols || y >= depth.rows)
            return;

        unsigned short dpval = depth(y, x);
        if (dpval < 10)
        {
            kset[y * depth.cols + x] = 0;
            return;
        }
        float Dp = (dpval-1000) * intr.sca;

        PosType xp = intr.reprojector(x, y, Dp);

        // float xl = (x - intr.cx) / intr.fx;
        // float yl = (y - intr.cy) / intr.fy;
        // float lambda = sqrtf(xl * xl + yl * yl + 1);
        scaled(y, x) = xp; // Dp * 0.0002f; //  / 1000.f; //meters

        PosType gp;
        __shared__ float cam[12];

        if (0 == threadIdx.x && threadIdx.y == 0)
        {
            for (int i = 0; i < 12; i++)
            {
                cam[i] = gpu_kpara.cam2base[i];
            }
        }
        __syncthreads();

        gp.x = cam[0 * 4 + 0] * xp.x + cam[0 * 4 + 1] * xp.y + cam[0 * 4 + 2] * xp.z + cam[0 * 4 + 3];
        gp.y = cam[1 * 4 + 0] * xp.x + cam[1 * 4 + 1] * xp.y + cam[1 * 4 + 2] * xp.z + cam[1 * 4 + 3];
        gp.z = cam[2 * 4 + 0] * xp.x + cam[2 * 4 + 1] * xp.y + cam[2 * 4 + 2] * xp.z + cam[2 * 4 + 3];

        gcloud(y, x) = gp; // Dp * 0.0002f; //  / 1000.f; //meters

        u32B4 vecs;
        int16_t vx = __float2int_rd(gp.x * (VOXELSIZE_INV)); //向下取整 __float2int_rn __float2int_rd rn是求最近的偶数，rz是逼近零，ru是向上舍入[到正无穷]，rd是向下舍入[到负无穷]。  std::floor
        int16_t vy = __float2int_rd(gp.y * (VOXELSIZE_INV)); //向下取整
        int16_t vz = __float2int_rd(gp.z * (VOXELSIZE_INV)); //向下取整
        u64B4 center = gpu_kpara.center;
        vecs.x = (int8_t)(vx - center.x);
        vecs.y = (int8_t)(vy - center.y);
        vecs.z = (int8_t)(vz - center.z);
        vecs.cnt = 0;
        // auto p = kset.get();
        kset[y * depth.cols + x] = vecs.u32;
        uint32_t *p = &bitmap[vecs.u32 >> 5];
        uint32_t val = (1 << (vecs.u32 & 0x1f));
        atomicOr(p, val);
    }

    //写论文测试用
    __global__ void
    extract_kernel_test(Point3dim *output_base, struct Voxel32 *dev_pbox, exmatcloud_para *para)
    {
        // printf("d:%d ,%f\n", *pos_index, device::devData);
        // device::devData += 2;
        // *pos_index += 2;
        int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        // output_base = output;

        if ((dev_pbox + blockId) == NULL)
            return;

        struct Voxel32 &vdev_pbox = dev_pbox[blockId];
        int pt_grid_x = threadIdx.x;
        int pt_grid_y = threadIdx.y;
        u32B4 index = vdev_pbox.index;

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
        // if (vdev_pbox.index.cnt > 0 && pt_grid_x == 0 && pt_grid_y == 0)
        // {
        //     vdev_pbox.index.cnt--;
        //     // printf("aaaaaaa %d\n",vdev_pbox.index.cnt);//
        // }
    }
    __global__ void
    extract_activate_kernel(Point3dim *output_base, struct Voxel32 *dev_pbox, exmatcloud_para *para)
    {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        if ((dev_pbox + blockId) == NULL)
            return;

        struct Voxel32 &vdev_pbox = dev_pbox[blockId];

        u32B4 index = vdev_pbox.index;
        // if (index.cnt != 0)
        // {
        //     return;
        // }
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
    // kernel_change_type(struct Voxel32 *dev_pbox, uint8_t val)
    // {
    //     int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    //     if ((dev_pbox + blockId) == NULL)
    //         return;
    //     struct Voxel32 &vdev_pbox = dev_pbox[blockId];
    //     // int pt_grid_x = threadIdx.x;
    //     // int pt_grid_y = threadIdx.y;
    //     vdev_pbox.index.type = val;
    // }
    __global__ void
    extract_kernel(ex_buf *output_base, CUVector<struct Voxel32 *> g_use, exmatcloud_para *para)
    {
        // printf("d:%d ,%f\n", *pos_index, device::devData);
        // device::devData += 2;
        // *pos_index += 2;
        int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        // __shared__  Point3dim *output_base; //
        // output_base = output;

        if ((g_use.len <= blockId))
            return;

        struct Voxel32 *vdev_pbox = g_use[blockId];
        int pt_grid_x = threadIdx.x;
        int pt_grid_z = threadIdx.y;
        u32B4 index = vdev_pbox->index;
        __syncthreads();
        uint8_t indexcnt = index.cnt;
        para->mask[blockId] = indexcnt < 2 ? true : false;
        if (indexcnt > 0 && pt_grid_x == 0 && pt_grid_z == 0)
        {
            //         if (asdasd != 0)
            // printf("%d\n", asdasd);
            vdev_pbox->index.cnt--;
            // printf("aaaaaaa %d\n",vdev_pbox.index.cnt);//
        }

        if (indexcnt != 0 && para->extall == false)
            return;

        for (int pt_grid_y = 0; pt_grid_y < 32; pt_grid_y++)
        {
            int volume_idx = pt_grid_z * 32 * 32 + pt_grid_y * 32 + pt_grid_x;
            union voxel voxel = vdev_pbox->pVoxel[volume_idx];
            if (std::abs(voxel.tsdf) < 0.2f && voxel.weight > 0)
            {
                // if (indexcnt > 1)
                {
                    unsigned int val = atomicInc(&para->dev_points_num, 0xffffff);
                    // float3 &pos = output_base[val].pos;

                    output_base->color[val] = voxel.color;
                    // else                        vec[0] = (index.x + 1 * center.x) * VOXELSIZE + pt_grid_x * VOXELSIZE_PCUBE;

                    // output_base->color[val] = make_uchar3(0, 255, 0);
                    output_base->pose[val].x = (index.x + para->center.x) * VOXELSIZE+ pt_grid_x * VOXELSIZE_PCUBE;
                    output_base->pose[val].y = (index.y + para->center.y) * VOXELSIZE + pt_grid_y * VOXELSIZE_PCUBE;
                    output_base->pose[val].z = (index.z + para->center.z) * VOXELSIZE+ pt_grid_z * VOXELSIZE_PCUBE;
                }

                // assert(0);
                // if (vdev_pbox.index.cnt == 0)

                // pos.rgb[0] = 255;//voxel.rgb[0];
                // pos.rgb[1] = 255;//index.x;//voxel.rgb[1];
                // pos.rgb[2] = 255;//index.x;//voxel.rgb[2];
                //    "     }
                //     else if (vdev_pbox.index.cnt > 7)
                //     {
                //         _color.x = 255; // voxel.rgb[0];
                //         _color.y = 0;   // index.x;//voxel.rgb[1];
                //         _color.z = 0;   // index.x;//voxel.rgb[2];
                //     }
                //     else // if (index.type == 2)
                //     {
                //         _color.x = 0;   // voxel.rgb[0];
                //         _color.y = 255; // index.x;//voxel.rgb[1];
                //         _color.z = 0;   // index.x;//voxel.rgb[2];
                //     }
            }
        }
    }

    // __global__ void
    // scaleDepth(const Patch<unsigned short> depth, Patch<PosType> scaled, const Intr intr)
    // {
    //     int x = threadIdx.x + blockIdx.x * blockDim.x;
    //     int y = threadIdx.y + blockIdx.y * blockDim.y;

    //     if (x >= depth.cols || y >= depth.rows)
    //         return;
    //     int Dp = depth(y,x);

    //     PosType xp;
    //     xp.z=Dp * 0.0002f;
    //     xp.x= xp.z*(x - intr.cx) / intr.fx;
    //     xp.y=xp.z*(y - intr.cy) / intr.fy;

    //     // float xl = (x - intr.cx) / intr.fx;
    //     // float yl = (y - intr.cy) / intr.fy;
    //     // float lambda = sqrtf(xl * xl + yl * yl + 1);

    //     scaled(y,x) =xp;// Dp * 0.0002f; //  / 1000.f; //meters
    // }

    __device__ void Tnte::operator()(struct Voxel32 **&dev_boxptr, struct kernelPara &gpu_kpara, const Patch<PosType> &depthScaled)
    {
        int pt_grid_x = threadIdx.x; // threadId 32
        int pt_grid_y = threadIdx.y; // threadId 32
        int blockId = blockIdx.x;
        __shared__ float cam2base[12];
        struct Voxel32 *pbox = dev_boxptr[blockId];
        struct u32B4 u32 = pbox->index;
        if (0 == pt_grid_x && pt_grid_y == 0)
        {
            pbox->index.cnt = 8;
            for (int i = 0; i < 12; i++)
                cam2base[i] = gpu_kpara.cam2base[i];
        }
        // float4 intr = make_float4(cam_K[0], cam_K[4], cam_K[2], cam_K[5]);

        u64B4 center = gpu_kpara.center;
        __syncthreads();
        for (int pt_grid_z = 0; pt_grid_z < CUBEVOXELSIZE; pt_grid_z++)
        {
            // 计算小体素的世界坐标
            float pt_base_x = (u32.x + center.x) * VOXELSIZE + pt_grid_x * VOXELSIZE_PCUBE;
            float pt_base_y = (u32.y + center.y) * VOXELSIZE + pt_grid_y * VOXELSIZE_PCUBE;
            float pt_base_z = (u32.z + center.z) * VOXELSIZE + pt_grid_z * VOXELSIZE_PCUBE;

            //计算体素在相机坐标系的坐标
            float tmp_pt[3] = {0};
            tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
            tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
            tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
            float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
            float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
            float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

            if (pt_cam_z <= 0)
                continue;
            // __float2int_rd(intr.fx * __fdividef(pt_cam_x, pt_cam_z) + intr.cx);
            int pt_pix_x = __float2int_rd(intr.x * __fdividef(pt_cam_x, pt_cam_z) + intr.z);
            int pt_pix_y = __float2int_rd(intr.y * __fdividef(pt_cam_y, pt_cam_z) + intr.w);
            if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
                continue;

            // uint16_t img_depu = gpu_kpara->dev_depthdata[pt_pix_y * im_width + pt_pix_x];
            // union points *pt = (union points *)&point[pt_pix_y * im_width + pt_pix_x];
            PosType img_dep = depthScaled(pt_pix_y, pt_pix_x); // meters

            // float img_dep = img_depu * 0.001f; //pt->xyz;//
            if (img_dep.z <= 0.2 ) //|| img_dep.z > 6)
                continue;
            float diff = img_dep.z - pt_cam_z;
            if (diff <= -trunc_margin)
                continue;
            uint32_t vox_index = pt_grid_x + pt_grid_y * CUBEVOXELSIZE + pt_grid_z * CUBEVOXELSIZE * CUBEVOXELSIZE;

            union voxel &p = pbox->pVoxel[vox_index];

            float dist = fmin(1.0f, __fdividef(diff, trunc_margin));
            // float weight_old = p.weight;//> 128 ? 128 : p.weight;
            // float weight_new = weight_old + 1.0f;
            float weight_old = (float)p.weight; //> 128 ? 128 : p.weight;
            p.weight = p.weight > 250 ? p.weight : p.weight + 1;
            float weight_new = __fdividef(1.0f, (float)p.weight);
            p.tsdf = __fmaf_rd(p.tsdf, weight_old, dist) * weight_new;

            uchar3 rgb_val = gpu_kpara.dev_rgbdata[pt_pix_y * im_width + pt_pix_x]; // pt->rgb[0];

            uint16_t mval = __fmaf_rd(p.color.x, weight_old, rgb_val.x) * weight_new;
            p.color.x = mval > 255 ? 255 : mval;
            mval = __fmaf_rd(p.color.y, weight_old, rgb_val.y) * weight_new;
            p.color.y = mval > 255 ? 255 : mval;
            mval = __fmaf_rd(p.color.z, weight_old, rgb_val.z) * weight_new;
            p.color.z = mval > 255 ? 255 : mval;
        }
    }

    __global__ void Integrate32F(struct Tnte fun, struct Voxel32 **dev_boxptr, struct kernelPara gpu_kpara,
                                 const Patch<PosType> depthScaled)
    {
        fun(dev_boxptr, gpu_kpara, depthScaled);
    }
}