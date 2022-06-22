
#include <cuda.h>
#include <iostream>
#include <math_constants.h>
#include "main.hpp"
#include <opencv2/viz/vizcore.hpp>
#include "../cuda/safe_call.hpp"

#pragma pack(push, 1)

#define PosType float3

#define CUBEVOXELSIZE (32)
#define DIV_CUBEVOXELSIZE (1.0f / CUBEVOXELSIZE)
#define VOXELSIZE_PCUBE (0.01f)
#define ACTIVATE_VOXNUM (1024)
#define DEPTHFACTOR (0.001f)

#define VOXELSIZE CUBEVOXELSIZE * 0.01f

typedef float4 Point;
typedef float4 Normal;
typedef uchar4 uchar4;
#pragma pack(pop)

#define COLS480 480
#define ROWS640 640

#include <fstream>
// #include "tsdf.cuh"
#include <opencv2/viz.hpp>
// #include "../cuda/device_array.hpp"
#define HASH_SPACE (0Xffffff)
#include <stack>
#include "cuVector.cuh"
#include "cuda/vector_math.hpp"

#define __kf_hdevice__ __host__ __device__ __forceinline__
#define __kf_device__ __device__ __forceinline__

__kf_device__ float3 normalized(const float3 &v)
{
    return v * rsqrt(dot(v, v));
}
__kf_hdevice__ float3 operator*(const float3 &v1, const int3 &v2)
{
    return make_float3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}
__kf_hdevice__ float3 operator/(const float &v, const float3 &vec)
{
    return make_float3(v / vec.x, v / vec.y, v / vec.z);
}

namespace device
{
    typedef int3 Vec3i;
    typedef float3 Vec3f;
    struct Mat3f
    {
        float3 data[3];
    };
    struct Aff3f
    {
        Mat3f R;
        Vec3f t;
    };
    __kf_device__ Vec3f operator*(const Mat3f &m, const Vec3f &v)
    {
        return make_float3(dot(m.data[0], v), dot(m.data[1], v), dot(m.data[2], v));
    }

    __kf_device__ Vec3f operator*(const Aff3f &a, const Vec3f &v) { return a.R * v + a.t; }

    __kf_device__ Vec3f tr(const float4 &v) { return make_float3(v.x, v.y, v.z); }

}

namespace device
{
    struct MGTsdfVolume
    {
    public:
        typedef union voxel elem_type;

        const int3 dims;
        const float3 voxel_size;
        const float trunc_dist;
        const int max_weight;

        PVOXEL *g_hashmap;
        uint cnt = 0;
        // __host__ PVOXEL &operator[](u32B4 key)
        // {
        //     return c_hashmap[key.u32 & 0xffffff];
        // }
        void CPU2GPU_updateHash()
        {
            // ck(cudaMemcpy(g_hashmap, c_hashmap, sizeof(struct Voxel32 *) * HASH_SPACE, cudaMemcpyHostToDevice));
        }
        MGTsdfVolume(PVOXEL *&g_hashmap, int3 _dims, float3 _voxel_size, float _trunc_dist, int _max_weight)
            : g_hashmap(g_hashmap), dims(_dims), voxel_size(_voxel_size), trunc_dist(_trunc_dist), max_weight(_max_weight){
                                                                                                       // cudaMalloc(&data, sizeof(voxel) * 512 * 512 * 512);
                                                                                                   };

        __device__ inline float get(int x, int y, int z) const
        {
            u32B4 upos;
            upos.x = __float2int_rd(x / 32);
            upos.y = __float2int_rd(y / 32);
            upos.z = __float2int_rd(z / 32);
            upos.cnt = 0;
            PVOXEL pv = g_hashmap[upos.u32 & 0xffffff];
            if (pv == 0)
            {
                return 0.0f;
            }
            u32B4 upos2;
            upos2.x = x % 32;
            upos2.y = y % 32;
            upos2.z = z % 32;
            return pv->pVoxel[upos2.z * 32 * 32 + 32 * upos2.y + upos2.x].tsdf;
        }
        __device__  float get(float3 p) const //inline
        {
            u32B4 upos;
            upos.x = __float2int_rd(p.x * 3.125f);
            upos.y = __float2int_rd(p.y * 3.125f);
            upos.z = __float2int_rd(p.z * 3.125f);
            upos.cnt = 0;
            PVOXEL pv = g_hashmap[upos.u32 & 0xffffff]; //
            if (pv == 0)
            {
                return 0.0f;
            }
            u32B4 upos2;
            upos2.x = __float2int_rd(p.x * 100) % 32;
            upos2.y = __float2int_rd(p.y * 100) % 32;
            upos2.z = __float2int_rd(p.z * 100) % 32;
            if (upos2.z < 0 || upos2.y < 0 || upos2.z < 0)
                return 1.0f;
            if (upos2.z >= 32 || upos2.y >= 32 || upos2.z >= 32)
                return 1.0f;
            uint16_t index = upos2.z * 32 * 32 + 32 * upos2.y + upos2.x;
            printf("%d \n",index);
            return pv->pVoxel[index].tsdf;
        }
        // __device__ inline const elem_type *operator()(int x, int y, int z) const {}
        // __device__ inline elem_type *beg(int x, int y) const {}
        // __device__ inline elem_type *zstep(elem_type *const ptr) const {}
        __device__ float interpolate(float3 p) const
        {
            p *= 100.0f;
            float3 cf = p;
            // TsdfVolume &volume = *(this);
            // rounding to negative infinity
            int3 g = make_int3(__float2int_rd(cf.x), __float2int_rd(cf.y), __float2int_rd(cf.z));
            // auto pv = this->get(p);
            // if (pv == 0)
            //     return CUDART_NAN_F;

            // if (g.x < 0 || g.x >= this->dims.x - 1 || g.y < 0 || g.y >= this->dims.y - 1 || g.z < 0 || g.z >= this->dims.z - 1)
            // return pv;
            float a = (cf.x - g.x);
            float b = (cf.y - g.y);
            float c = (cf.z - g.z);

            float tsdf = 0.f;
            tsdf += this->get(g.x + 0, g.y + 0, g.z + 0); // * (1 - a) * (1 - b) * (1 - c);
            tsdf += this->get(g.x + 0, g.y + 0, g.z + 1) * (1 - a) * (1 - b) * c;
            tsdf += this->get(g.x + 0, g.y + 1, g.z + 0) * (1 - a) * b * (1 - c);
            tsdf += this->get(g.x + 0, g.y + 1, g.z + 1) * (1 - a) * b * c;
            tsdf += this->get(g.x + 1, g.y + 0, g.z + 0) * a * (1 - b) * (1 - c);
            tsdf += this->get(g.x + 1, g.y + 0, g.z + 1) * a * (1 - b) * c;
            tsdf += this->get(g.x + 1, g.y + 1, g.z + 0) * a * b * (1 - c);
            tsdf += this->get(g.x + 1, g.y + 1, g.z + 1) * a * b * c;
            return tsdf;
        }

    private:
        MGTsdfVolume &operator=(const MGTsdfVolume &);
    };
    __device__ inline void intersect(float3 ray_org, float3 ray_dir, /*float3 box_min,*/ float3 box_max, float &tnear, float &tfar)
    {
        const float3 box_min = make_float3(0.f, 0.f, 0.f);

        // compute intersection of ray with all six bbox planes
        float3 invR = make_float3(1.f / ray_dir.x, 1.f / ray_dir.y, 1.f / ray_dir.z);
        float3 tbot = invR * (box_min - ray_org);
        float3 ttop = invR * (box_max - ray_org);

        // re-order intersections to find smallest and largest on each axis
        float3 tmin = make_float3(fminf(ttop.x, tbot.x), fminf(ttop.y, tbot.y), fminf(ttop.z, tbot.z));
        float3 tmax = make_float3(fmaxf(ttop.x, tbot.x), fmaxf(ttop.y, tbot.y), fmaxf(ttop.z, tbot.z));

        // find the largest tmin and the smallest tmax
        tnear = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
        tfar = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
    }
    struct TsdfRaycaster
    {
        device::MGTsdfVolume volume;

        Aff3f aff;
        Mat3f Rinv;
        Intr reproj;
        Vec3f volume_size;
        float time_step;
        float3 gradient_delta;
        float3 voxel_size_inv;

        TsdfRaycaster(const MGTsdfVolume &volume, const Aff3f &aff, const Mat3f &Rinv, const Intr &_reproj);
        __device__ inline void operator()(Patch<float4> points, Patch<float4> normals) const
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= points.cols || y >= points.rows)
                return;
            const float qnan = CUDART_NAN_F;

            points(y, x) = normals(y, x) = make_float4(qnan, qnan, qnan, qnan);

            float3 ray_org = aff.t;
            float3 ray_dir = normalized(aff.R * reproj.reprojector(x, y, 1.f));
            // printf("%f,%f,%f,%f,%f,%f\n", ray_org.x, ray_org.y, ray_org.z, ray_dir.x, ray_dir.y, ray_dir.z);

            float3 box_max = volume_size - volume.voxel_size;

            float tmin, tmax;
            intersect(ray_org, ray_dir, box_max, tmin, tmax);

            const float min_dist = 0.f;
            tmin = fmax(min_dist, tmin);
            if (tmin >= tmax)
                return;

            tmax -= time_step;
            float3 vstep = ray_dir * time_step;
            float3 next = ray_org + ray_dir * tmin;

            float tsdf_next = volume.get(next);
            // // printf("%f ",tsdf_next);
            for (float tcurr = tmin; tcurr < tmax; tcurr += time_step)
            {
                float tsdf_curr = tsdf_next;
                float3 curr = next;
                next += vstep;

                u32B4 uid, _pos2;
                tsdf_next = volume.get(next);
                if (tsdf_curr < 0.f && tsdf_next > 0.f)
                    break;

                if (tsdf_curr > 0.f && tsdf_next < 0.f)
                {

                    // points(y, x) = make_float4(next.x, next.y, next.z, 0); // make_float4(normal.x, normal.y, normal.z, 0.f);
                    // points(y, x) = make_float4(vertex.x, vertex.y, vertex.z, 0.f);
                    float Ft = volume.interpolate(curr);
                    float Ftdt = volume.interpolate(next);

                    float Ts = tcurr - __fdividef(time_step * Ft, Ftdt - Ft);
                    // printf("%f %f", Ft, Ts);
                    float3 vertex = ray_org + ray_dir * Ts;
                    float3 normal = compute_normal(vertex);

                    if (!isnan(normal.x * normal.y * normal.z))
                    {
                        normal = Rinv * normal;
                        vertex = Rinv * (vertex - aff.t);

                        normals(y, x) = make_float4(normal.x, normal.y, normal.z, 0.f);
                        points(y, x) = make_float4(vertex.x, vertex.y, vertex.z, 0.f);
                    }

                    break;
                }
            } /* for (;;) */
        }

        __device__ inline float3 compute_normal(const float3 &p) const
        {
            float3 n;

            float Fx1 = volume.interpolate(make_float3(p.x + gradient_delta.x, p.y, p.z));
            float Fx2 = volume.interpolate(make_float3(p.x - gradient_delta.x, p.y, p.z));
            n.x = __fdividef(Fx1 - Fx2, gradient_delta.x);

            float Fy1 = volume.interpolate(make_float3(p.x, p.y + gradient_delta.y, p.z));
            float Fy2 = volume.interpolate(make_float3(p.x, p.y - gradient_delta.y, p.z));
            n.y = __fdividef(Fy1 - Fy2, gradient_delta.y);

            float Fz1 = volume.interpolate(make_float3(p.x, p.y, p.z + gradient_delta.z));
            float Fz2 = volume.interpolate(make_float3(p.x, p.y, p.z - gradient_delta.z));
            n.z = __fdividef(Fz1 - Fz2, gradient_delta.z);

            return normalized(n);
        }
    };

    inline TsdfRaycaster::TsdfRaycaster(const MGTsdfVolume &_volume, const Aff3f &_aff, const Mat3f &_Rinv, const Intr &_reproj)
        : volume(_volume), aff(_aff), Rinv(_Rinv), reproj(_reproj) {}

    __global__ void raycast_kernel(const TsdfRaycaster raycaster, Patch<float4> points, Patch<float4> normals)
    {
        raycaster(points, normals);
    };

}
// }

__global__ void render_image_kernel(const Patch<float4> points, const Patch<float4> normals,
                                    const float3 light_pose, Patch<uchar4> dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= points.cols || y >= points.rows)
        return;

    float3 color;

    float3 p = device::tr(points(y, x));

    if (isnan(p.x))
    {
        const float3 bgr1 = make_float3(4.f / 255.f, 2.f / 255.f, 2.f / 255.f);
        const float3 bgr2 = make_float3(236.f / 255.f, 120.f / 255.f, 120.f / 255.f);

        float w = static_cast<float>(y) / dst.rows;
        color = bgr1 * (1 - w) + bgr2 * w;
    }
    else
    {
        float3 P = p;
        float3 N = device::tr(normals(y, x));

        const float Ka = 0.3f; // ambient coeff
        const float Kd = 0.5f; // diffuse coeff
        const float Ks = 0.2f; // specular coeff
        const float n = 20.f;  // specular power

        const float Ax = 1.f; // ambient color,  can be uchar4
        const float Dx = 1.f; // diffuse color,  can be uchar4
        const float Sx = 1.f; // specular color, can be uchar4
        const float Lx = 1.f; // light color

        // Ix = Ax*Ka*Dx + Att*Lx [Kd*Dx*(N dot L) + Ks*Sx*(R dot V)^n]

        float3 L = normalized(light_pose - P);
        float3 V = normalized(make_float3(0.f, 0.f, 0.f) - P);
        float3 R = normalized(N * dot(N, L) * 2.0f - L);

        float Ix = Ax * Ka * Dx + Lx * Kd * Dx * fmax(0.f, dot(N, L)) + Lx * Ks * Sx * __powf(fmax(0.f, dot(R, V)), n);
        color = make_float3(Ix, Ix, Ix);
    }

    uchar4 out;
    out.x = static_cast<unsigned char>(__saturatef(color.x) * 255.f);
    out.y = static_cast<unsigned char>(__saturatef(color.y) * 255.f);
    out.z = static_cast<unsigned char>(__saturatef(color.z) * 255.f);
    dst(y, x) = out;
}

template <typename D, typename S>
inline D device_cast(const S &source)
{
    return *reinterpret_cast<const D *>(source.val);
}
inline device::Aff3f device_castA(const cv::Affine3f &source)
{
    device::Aff3f aff;
    cv::Matx<float, 3, 3> R = source.rotation();
    cv::Vec3f t = source.translation();
    aff.R = device_cast<device::Mat3f>(R);
    aff.t = device_cast<device::Vec3f>(t);
    return aff;
}
#include "iostream"
#include "opencv2/opencv.hpp"
#include "ammap.h"
#include <unordered_map>

void cube2mgfusion()
{
    mmaplib::mmap fm("/home/u20/dataset/code/paper/KinectFusion_2022/build/t.bin");

    const union voxel *pv = (union voxel *)fm.data();
    assert(fm.is_open());
    // pv = reinterpret_cast<>(fm.data());
    std::unordered_map<uint32_t, struct Voxel32 *> map;
    // map[3] = 3;
    for (int i = 0; i < 512; i++)
        for (int j = 0; j < 512; j++)
            for (int k = 0; k < 512; k++)
            {
                const union voxel vb = pv[k + j * 512 + i * 512 * 512];
                if (vb.tsdf > 0.90)
                    continue;
                // printf("%f\n", vb.tsdf);
                u32B4 pos;
                pos.x = k / 32,
                pos.y = j / 32,
                pos.z = i / 32;
                pos.cnt = 0;
                // map[pos.u32] = new struct Voxel32;
                // printf("%d\n",pos.u32);
                if (!map.count(pos.u32))
                {
                    map[pos.u32] = new struct Voxel32;
                    // map.insert(std::pair<uint32_t, struct Voxel32 *>(pos.u32, nullptr)); // new struct Voxel32;
                }

                u32B4 pos2;
                pos2.x = k % 32,
                pos2.y = j % 32,
                pos2.z = i % 32;
                map[pos.u32]->index = pos;
                map[pos.u32]->pVoxel[pos2.z * 32 * 32 + 32 * pos2.y + pos2.x] = vb;
            }

    std::fstream file2("new.bin", std::ios::out | std::ios::binary); // | ios::app
    size_t len2 = map.size();
    std::cout << len2 << " " << fm.size() / sizeof(union voxel) / 512 / 512 / 512 << std::endl;
    file2.write(reinterpret_cast<char *>(&len2), sizeof(size_t));
    for (auto &it : map)
    {
        file2.write(reinterpret_cast<char *>(it.second), sizeof(struct Voxel32));
        // printf("%x \n", it.second->index.u32);
    }
    file2.close();
}

raycast::raycast()
{
    points.create(COLS480, ROWS640);
    norm.create(COLS480, ROWS640);
    dst.create(COLS480, ROWS640);

    raycastimg.create(COLS480, ROWS640, CV_8UC4);
    point_.create(COLS480, ROWS640, CV_32FC4);
}
raycast::~raycast()
{
    dst.release();
    points.release();
    norm.release();
}
int raycast::mgraycast_test(float4 cam, PVOXEL *&g_hashmap, cv::Affine3f camera_pose)
{

    struct Voxel32 pboxs;
    u64B4 cen;

    Intr intr(cam.x, cam.y, cam.z, cam.w);

    auto dims = make_int3(512, 512, 512);
    auto vsz = make_float3(0.01, 0.01, 0.01);
    float trunc_dist_ = 0.01f;
    int max_weight_ = 256;
    float raycaster_step_factor = 0.01;

    float gradient_delta_factor = 0.01;

    // cv::Affine3f pose = cv::Affine3f().translate(cv::Vec3f(-5.0 / 2, -5.0 / 2, -5.0 / 2));
    cv::Affine3f pose = cv::Affine3f::Identity(); //.translate(cv::Vec3f(-5.0 / 2, -5.0 / 2, -5.0 / 2));

    device::MGTsdfVolume volume(g_hashmap, dims, vsz, trunc_dist_, max_weight_);

    cv::Affine3f cam2vol = pose.inv() * camera_pose;
    device::Aff3f aff = device_castA(cam2vol);
    device::Mat3f Rinv = device_cast<device::Mat3f>(cam2vol.rotation().t()); //旋转矩阵视为单位矩阵 逆矩阵等于转置矩阵

    device::TsdfRaycaster rc(volume, aff, Rinv, intr);

    rc.volume_size = volume.voxel_size * volume.dims;
    rc.time_step = volume.trunc_dist * raycaster_step_factor;
    rc.gradient_delta = volume.voxel_size * gradient_delta_factor;
    rc.voxel_size_inv = 1.f / volume.voxel_size;

    dim3 block(32, 32);
    dim3 grid(divUp(points.cols, block.x), divUp(points.rows, block.y));

    raycast_kernel<<<grid, block>>>(rc, points, norm);
    ck(cudaDeviceSynchronize());
    ck(cudaGetLastError());

    float3 lp = make_float3(0, 0, 0);
    render_image_kernel<<<grid, block>>>(points, norm, lp, dst);
    ck(cudaDeviceSynchronize());
    ck(cudaGetLastError());

    dst.download(raycastimg.ptr<uchar4>(), raycastimg.step);
    ck(cudaGetLastError());

    cv::imshow("raycastimg", raycastimg);
    cv::waitKey(10);
    points.download(point_.ptr<float4>(), point_.step);
    ck(cudaGetLastError());
    std::cout << camera_pose.matrix << std::endl;

    return 0;
}