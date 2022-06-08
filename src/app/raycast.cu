

#include <fstream>
#include "../cuda/datatype.cuh"
#include "../cuda/safe_call.hpp"
#include "tsdf.cuh"
#include <opencv2/viz.hpp>
#include "../cuda/device_array.hpp"
#define HASH_SPACE (0Xffffff)
#include <stack>
#include "cuVector.cuh"
typedef struct Voxel32 *PVOXEL;
struct MVhash
{
    PVOXEL *g_hashmap;
    PVOXEL *c_hashmap;
    PVOXEL g_space;
    std::stack<PVOXEL> free_space;
    std::stack<PVOXEL> use_space;
    uint cnt = 0;
    MVhash()
    {
        cudaMalloc(&g_hashmap, sizeof(struct Voxel32 *) * HASH_SPACE);
        cudaMemset(g_hashmap, 0, sizeof(struct Voxel32 *) * HASH_SPACE);
        cudaMallocHost(&c_hashmap, sizeof(struct Voxel32 *) * HASH_SPACE);
        cudaMemset(c_hashmap, 0, sizeof(struct Voxel32 *) * HASH_SPACE);
        cudaMalloc(&g_space, sizeof(struct Voxel32) * 512);
        for (int i = 0; i < 512; i++)
        {
            free_space.push(&g_space[i]);
            // std::cout << &g_space[i] << std::endl;
        }
    }
    void reset()
    {
        cnt = 0;
    }
    PVOXEL getcnt()
    {
        use_space.push(g_space + cnt);
        return g_space + cnt++;
    }
    __host__ PVOXEL &operator[](u32B4 key)
    {
        return c_hashmap[key.u32 & 0xffffff];
    }
    const __device__ PVOXEL &get(u32B4 key) const
    {
        return g_hashmap[key.u32 & 0xffffff];
    }
    void CPU2GPU_updateHash()
    {
        ck(cudaMemcpy(g_hashmap, c_hashmap, sizeof(struct Voxel32 *) * HASH_SPACE, cudaMemcpyHostToDevice));
    }

    __device__ PVOXEL word2get(float3 pos)
    {
        u32B4 _pos;
        _pos.x = pos.x * 3.125f;
        _pos.y = pos.y * 3.125f;
        _pos.z = pos.z * 3.125f;

        return g_hashmap[_pos.u32 & 0xffffff];
    }
    __device__ union voxel *word2getvol(float3 pos, u32B4 &uid, u32B4 &_pos2)
    {
        u32B4 _pos;
        _pos.x = __float2int_rd(pos.x * 3.125f);
        _pos.y = __float2int_rd(pos.y * 3.125f);
        _pos.z = __float2int_rd(pos.z * 3.125f);
        PVOXEL pv = g_hashmap[_pos.u32 & 0xffffff];
        if (pv == 0)
            return 0;
        uid = pv->index;

        _pos2.x = (pos.x - _pos.x * 0.32f) * 100;
        _pos2.y = (pos.y - _pos.y * 0.32f) * 100;
        _pos2.z = (pos.z - _pos.z * 0.32f) * 100;

        // printf("%d,%d,%d\n", _pos2.x, _pos2.y, _pos2.z);
        return &pv->pVoxel[_pos2.x + _pos2.y * 32 + _pos2.z * 32 * 32];
    }
};
__device__ unsigned int cnt = 0;
__global__ void ker(struct MVhash mhash, CUVector<float3> *out, CUVector<uchar3> *rgb, Intr intr)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    // printf("a%ld\n", val);
    for (int z = 0; z < 100; z++)
    {
        float3 pos = make_float3(x * 0.01 - 1, y * 0.01 - 1, z * 0.01 - 1);
        // pos.x = (x - intr.cx) / intr.fx;
        // pos.y = (y - intr.cy) / intr.fy;
        // printf("%f\n",(x - intr.cx) / intr.fx);
        pos.z = z * 0.01;

        //

        PVOXEL ret = mhash.word2get(pos);
        u32B4 uid, _pos2;
        auto pv = mhash.word2getvol(pos, uid, _pos2);
        if (pv != 0)
        {
            if (pv->weight > 0.2 && fabs(pv->tsdf) < 0.2)
            {
                unsigned int val = atomicInc(&out->cnt, 0xffffff);
                (*out)[val] =pos;// make_float3(uid.x * 0.32 + _pos2.x * 0.01, uid.y * 0.32 + _pos2.y * 0.01, uid.z * 0.32 + _pos2.z * 0.01);
            }

            // (*rgb)[val] = make_uchar3(x, y, z);
        }
        if (ret != 0)
        {
            // u32B4 uid = ret->index;
            // for (int i = 0; i < 32; i++)
            //     for (int j = 0; j < 32; j++)
            //         for (int k = 0; k < 32; k++)
            //         {
            //             // float3 pos2;

            //             // pos.x = (x - intr.cx) / intr.fx;
            //             // pos.y = (y - intr.cy) / intr.fy;
            //             // // printf("%f\n",(x - intr.cx) / intr.fx);
            //             // pos.z = z * 0.2;

            //             union voxel &pv = ret->pVoxel[k + j * 32 + i * 32 * 32];
            //             if (pv.weight > 0.2 && fabs(pv.tsdf) < 0.2)
            //             {
            //                 unsigned int val = atomicInc(&out->cnt, 0xffffff);
            //                 if (val == out->len)
            //                     return;
            //                 (*out)[val] = make_float3(uid.x * 0.32 + k * 0.01, uid.y * 0.32 + j * 0.01, uid.z * 0.32 + i * 0.01);
            //                 (*rgb)[val] = make_uchar3(pv.rgb[0], pv.rgb[1], pv.rgb[2]);
            //             }
            //         }
        }
        else
        {
            // out(y, x) = make_float3(0, 0, 0);
        }
    }

    // u32B4 mu32 = mhash.g_space[x].index; // g_space[x].index.u32
    // const PVOXEL &m2 = mhash.get(mu32);  // mhash.g_hashmap[mu32 & 0xffffff];
    // printf("id:%d mapid=%x %p add= %p\n", x, mu32.u32, m2, &mhash.g_space[x]);
}
int main()
{
    std::fstream file("abgrids.bin", std::ios::in | std::ios::binary); // | ios::app
    assert(file.is_open());
    struct Voxel32 pboxs;
    size_t len;
    file.read(reinterpret_cast<char *>(&len), sizeof(size_t));
    std::cout << len << std::endl;
    u64B4 cen;
    cv::Mat point, color;

    auto window = new cv::viz::Viz3d("map");
    window->showWidget("Coordinate", cv::viz::WCoordinateSystem());

    MVhash mhsh;
    Intr intr(550, 550, 320, 240);
    for (int i = 0; i < 295; i++)
    {
        file.read(reinterpret_cast<char *>(&pboxs), sizeof(struct Voxel32));
        auto p = mhsh.getcnt();

        ck(cudaMemcpy(p, &pboxs, sizeof(struct Voxel32), cudaMemcpyHostToDevice));
        // file.write(reinterpret_cast<char *>(&pboxs), sizeof(struct Voxel32));
        pboxs.index.cnt = 0xff;
        // printf("%x\n", pboxs.index.u32);
        // std::cout << p << std::endl;

        mhsh[pboxs.index] = p;

        pboxs.tobuff(point, color, cen);
        mhsh.free_space.pop();
        // std::cout << pboxs.index.u32 << " " << &g_space[i] << std::endl;
    }
    mhsh.CPU2GPU_updateHash();
    CUVector<float3> *gout;
    CUVector<uchar3> *img;
    cudaMallocManaged(&img, sizeof(CUVector<uchar3>));
    cudaMallocManaged(&gout, sizeof(CUVector<float3>));
    gout->creat(300 * 300 * 90);
    img->creat(300 * 300 * 90);

    // cudaMemcpy(gout, &out, sizeof(CUVector<float3>), cudaMemcpyHostToDevice);
    ker<<<640, 480>>>(mhsh, gout, img, intr);
    ck(cudaDeviceSynchronize());
    ck(cudaGetLastError());

    std::cout << gout->cnt << " " << gout->len << std::endl;
    cv::Mat dnp(gout->cnt, 1, CV_32FC3);
    cv::Mat col(gout->cnt, 1, CV_8UC3);

    gout->download(dnp.ptr<float3>(), gout->cnt * sizeof(float3));
    img->download(col.ptr<uchar3>(), gout->cnt * sizeof(uchar3));

    // cv::Mat ppo = dnp.reshape(3, dnp.rows * dnp.cols);
    cv::Affine3f pose = cv::Affine3f().translate(cv::Vec3f(4, 0, 0));
    window->showWidget("w2", cv::viz::WCloud(point, color), pose);
    window->showWidget("wc", cv::viz::WCloud(dnp, col));
    // std::cout << col << std::endl;
    window->spin();

    file.close();
}