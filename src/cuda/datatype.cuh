#pragma once
#include <cuda_runtime_api.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#define CUBEVOXELSIZE (32)
#define DIV_CUBEVOXELSIZE (1.0f / CUBEVOXELSIZE)
#define VOXELSIZE_PCUBE (0.01f)
#define ACTIVATE_VOXNUM (1024)
#define DEPTHFACTOR (0.001f)

#define VOXELSIZE CUBEVOXELSIZE * 0.01f
#define VOXELSIZE_INV 3.125f

/** \brief Camera intrinsics structure
 */
struct Intr
{
    union
    {
        float4 cam;
        struct
        {
            float fx, fy, cx, cy;
        };
    };

    float2 finv;
    float sca;
    Intr() : fx(0), fy(0), cx(0), cy(0) {}
    Intr(float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_)
    {
        finv = make_float2(1.0f / fx_, 1.0f / fy_);
    }
    Intr(cv::Mat &k)
    {
        fx = k.at<float>(0, 0);
        fy = k.at<float>(1, 1);
        cx = k.at<float>(0, 2);
        cy = k.at<float>(1, 2);
        finv = make_float2(1.0f / fx, 1.0f / fy);
    }
    // Reprojector::Reprojector(float fx, float fy, float cx, float cy)
    inline __device__ float3 reprojector(int u, int v, float z) const
    {
        float x = z * (u - cx) * finv.x;
        float y = z * (v - cy) * finv.y;
        return make_float3(x, y, z);
    }
    __device__ __forceinline__ uint2 Projector(const float3 &p) const
    {
        uint2 coo;
        coo.x = __fmaf_rd(fx, __fdividef(p.x, p.z), cx);
        coo.y = __fmaf_rd(fy, __fdividef(p.y, p.z), cy);
        return coo;
    }

    void print()
    {
        printf("%f,%f,%f,%f sca:%f,%f\n", fx, fy, cx, cy, sca, finv.x);
    }
};

static inline int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

#pragma pack(push, 1)

enum BOX_TYPE
{
    GLOAB = 0,
    CURR = 1
};
struct u32B4;

struct ex_buf
{
    uchar3 color[32 * 32 * 32 * 128];
    float3 pose[32 * 32 * 32 * 128];
};
union Point3dim
{
    uint8_t data[15];
    struct
    {
        float x, y, z;
        uint8_t rgb[3];
    };
};
union voxel
{
    struct
    {
        float tsdf;
        //  float weight;
        uint8_t weight;
        union
        {
            uchar3 color;
            uint8_t rgb[3];
        };
    };
};

struct UPoints
{
    union
    {
        float3 pos;
        float xyz[3];
    };
    union
    {
        uchar3 color;
        uint8_t rgb[3];
    };
    void print()
    {
        printf("%f %f %f %d %d %d\n", pos.x, pos.y, pos.z, color.x, color.y, color.z);
    }
};
#define PosType float3
struct u64B4;
struct u32B4
{
    union
    {
        uint32_t u32 = 0x00000000;
        struct
        {
            int8_t x;
            int8_t y;
            int8_t z;
            uint8_t cnt;
            // uint8_t type : 2;
            // uint8_t cnt : 6;
        };
    };

    __host__ void print()
    {
        printf("%d %d %d\n", x, y, z);
    }
    __device__ __host__ u32B4(uint32_t _u = 0) : u32(_u){};
    __host__ __device__ friend u64B4 operator+(const u32B4 &u32, u64B4 u64);
};
struct u64B4
{
public:
    union
    {
        uint64_t u64;
        struct
        {
            int16_t x;
            int16_t y;
            int16_t z;
            int16_t _rev;
        };
    };
    __host__ __device__ u64B4(int16_t x = 0, int16_t y = 0, int16_t z = 0) : x(x), y(y), z(z)
    {
        _rev = 0;
    }
    __host__ __device__ u64B4(u32B4 u32)
    {
        x = (u32.x), y = (u32.y), z = (u32.z);
        _rev = 0;
    }
    __host__ void print()
    {
        printf("%d %d %d\n", x, y, z);
    }
};

// #define MY_COM_OPERA (
// __host__ __device__ inline u64B4 operator+(const u32B4 &u32, u64B4 u64)
// {
//     u64B4 ans;
//     ans.x = u32.x + u64.x;
//     ans.y = u32.y + u64.y;
//     ans.z = u32.z + u64.z;
//     ans._rev = 0;
//     return ans;
// }
__host__ __device__ inline u64B4 operator-(const u64B4 &u32, const u64B4 &u64)
{
    u64B4 ans;
    ans.x = u32.x - u64.x;
    ans.y = u32.y - u64.y;
    ans.z = u32.z - u64.z;
    return ans;
}

struct Voxel32
{
    u32B4 index;
    bool tobuff(cv::Mat &points, cv::Mat &color, const u64B4 &center)
    {
        // if (index.cnt > 0)
        //     index.cnt--;
        // else
        //     return false;
        for (int8_t pt_grid_z = 0; pt_grid_z < CUBEVOXELSIZE; pt_grid_z++)
        {
            for (int8_t pt_grid_y = 0; pt_grid_y < CUBEVOXELSIZE; pt_grid_y++)
            {
                for (int8_t pt_grid_x = 0; pt_grid_x < CUBEVOXELSIZE; pt_grid_x++)
                {
                    int volume_idx = pt_grid_z * CUBEVOXELSIZE * CUBEVOXELSIZE + pt_grid_y * CUBEVOXELSIZE + pt_grid_x;
                    union voxel &voxel = pVoxel[volume_idx];
                    // if (voxel.weight < 0.001f)
                    // {
                    //     continue;
                    // }
                    if (std::abs(voxel.tsdf) < 0.2f && voxel.weight > 0.0f) // || (pt_grid_x == 0 && pt_grid_y == 0 && pt_grid_z == 0))
                                                                            // if (pt_grid_x % 5 == 0 && pt_grid_y % 5 == 0 && pt_grid_z % 5 == 0)
                    {
                        cv::Vec3f vec;
                        vec[0] = (index.x + 1 * center.x) * VOXELSIZE + pt_grid_x * VOXELSIZE_PCUBE;
                        vec[1] = (index.y + 1 * center.y) * VOXELSIZE + pt_grid_y * VOXELSIZE_PCUBE;
                        vec[2] = (index.z + 1 * center.z) * VOXELSIZE + pt_grid_z * VOXELSIZE_PCUBE;
                        points.push_back(vec);
                        color.push_back(cv::Vec3b(voxel.rgb[0], voxel.rgb[1], voxel.rgb[2]));
                    }
                    // else if (pt_grid_x % 5 == 0 && pt_grid_y % 5 == 0 && pt_grid_z % 5 == 0)
                    // {
                    //     cv::Vec3f vec;
                    //     vec[0] = index.x * VOXELSIZE + pt_grid_x * VOXELSIZE_PCUBE;
                    //     vec[1] = index.y * VOXELSIZE + pt_grid_y * VOXELSIZE_PCUBE;
                    //     vec[2] = index.z * VOXELSIZE + pt_grid_z * VOXELSIZE_PCUBE;
                    //     points.push_back(vec);
                    //     color.push_back(cv::Vec3b(voxel.weight, voxel.weight, voxel.weight));
                    // }
                }
            }
        }
        return true;
    }
    void tobuff_all_space(cv::Mat &points, cv::Mat &color, const u64B4 &center);

    union voxel pVoxel[CUBEVOXELSIZE * CUBEVOXELSIZE * CUBEVOXELSIZE];

    Voxel32()
    {
        memset(pVoxel, 0x00, sizeof(union voxel) * CUBEVOXELSIZE * CUBEVOXELSIZE * CUBEVOXELSIZE);
    }

    // void tobox32_pionts()
    // {
    //     struct CloudBox pb
    // }
};
struct CpuVoxel32
{
    Voxel32 *pvoxel32;
    u64B4 wordPos;
    CpuVoxel32()
    {
        pvoxel32 = new Voxel32;
    };
};
struct CloudBox
{
    u32B4 index;
    u64B4 wordPos;
    std::vector<UPoints> points;

    CloudBox(const u32B4 &index, const u64B4 &_center) : index(index), wordPos(_center)
    {
    }
    CloudBox(const u64B4 &_center) : wordPos(_center)
    {
    }
    CloudBox()
    {
    }
    void copyTobox32(struct Voxel32 *&pcpu_box32)
    {
        pcpu_box32 = new struct Voxel32;
        for (std::size_t i = 0; i < points.size(); i++) //
        {
            u32B4 u64;
            UPoints &ptf = points[i];
            u64.x = std::floor(3.125f * ptf.pos.x);
            u64.y = std::floor(3.125f * ptf.pos.y);
            u64.z = std::floor(3.125f * ptf.pos.z);
            u32B4 u32;
            u32.x = (ptf.xyz[0] - u64.x * 0.32f) * 100;
            u32.y = (ptf.xyz[1] - u64.y * 0.32f) * 100;
            u32.z = (ptf.xyz[2] - u64.z * 0.32f) * 100;
            assert(u32.x < 32);
            assert(u32.y < 32);
            assert(u32.z < 32);
            int index = u32.x + u32.y * 32 + u32.z * 32 * 32;
            pcpu_box32->index = u64;
            pcpu_box32->pVoxel[index].tsdf = 0.10f;
            pcpu_box32->pVoxel[index].weight = 1.0f;
            pcpu_box32->pVoxel[index].rgb[0] = ptf.rgb[0];
            pcpu_box32->pVoxel[index].rgb[1] = ptf.rgb[1];
            pcpu_box32->pVoxel[index].rgb[2] = ptf.rgb[2];
        }
    }
};
struct kernelPara
{
    u64B4 center;
    union
    {
        float cam2base[16];
        float4 pose[4];
    };

    uchar3 *dev_rgbdata; //[480 * 640];

    // uint16_t dev_depthdata[480 * 640];
};
struct exmatcloud_para
{

    u64B4 center;
    unsigned int dev_points_num = 0;
    bool extall;
    // uint8_t buf[512-8-4];
};

struct ex_buf_
{
    UPoints up[32 * 32 * 32];
    // struct posevoxel pose[32 * 32 * 32];

    u64B4 center;
    unsigned int dev_points_num = 0;
};

#pragma pack(pop)
