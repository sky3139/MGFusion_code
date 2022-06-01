#pragma once
#include <cuda_runtime_api.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#define CUBEVOXELSIZE (32)
#define VOXELSIZE_PCUBE (0.01f)
#define ACTIVATE_VOXNUM (1024)
#define DEPTHFACTOR (0.001f)

#define VOXELSIZE CUBEVOXELSIZE * 0.01f


/** \brief Camera intrinsics structure
  */
struct Intr
{
    float fx, fy, cx, cy;
    Intr() : fx(0), fy(0), cx(0), cy(0) {}
    Intr(float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

    Intr operator()(int level_index) const
    {
        int div = 1 << level_index;
        return (Intr(fx / div, fy / div, cx / div, cy / div));
    }
    Intr(cv::Mat &k)  {

        fx=k.at<float>(0,0);
        fy=k.at<float>(1,1);

        cx=k.at<float>(0,2);
        cy=k.at<float>(1,2);

    }
    void print()
    {
        printf("%f,%f,%f,%f\n",fx,fy,cx,cy);
    }
};
struct cam_pose_str
{
    float val[16];

};
static inline int divUp(int total, int grain) {
    return (total + grain - 1) / grain;
}

#pragma pack(push, 1)




enum BOX_TYPE {
    GLOAB=0,CURR=1
};
union u64_4byte
{
    uint64_t u64 = 0x00;
    struct
    {
        int16_t x;
        int16_t y;
        int16_t z;
        int16_t _rev;
    };
    // struct
    // {
    //   int64_t x : 24;
    //   int64_t y : 24;
    //   int64_t z : 16;
    // };

    // int8_t byte4[4];
};
struct colorvoxel
{
    uint8_t rgb[3];
};
struct posevoxel
{
    float x, y, z;
    // float _recv=1.0f;
};
struct ex_buf
{
   struct colorvoxel color[32 * 32 * 32 * 256];
   struct posevoxel pose[32 * 32 * 32 * 256];
};

// struct Point3dim
// {
union Point3dim
{
    uint8_t data[15];
    struct
    {
        float x, y, z;
        uint8_t rgb[3];
    };
    // struct colorvoxel;
};
// };

union voxel
{
    struct
    {

        float tsdf;
        //  float weight;
        uint8_t weight;
        uint8_t rgb[3];
    };
};
union UPoints
{
    struct
    {
        float xyz[3];
        uint8_t rgb[3];
    };
};
union xyzPoints
{
    struct
    {
        float x;
        float y;
        float z;
    };
};


union u32_4byte
{
    uint32_t u32 = 0x00000000;
    // struct
    // {
    //     int32_t x:9;
    //     int32_t y:9;
    //     int32_t z:8;
    //     uint32_t type:2;
    //     uint32_t cnt:4;
    // };

        struct
    {
        int8_t x;
        int8_t y;
        int8_t z;
        int8_t type:2;
        uint8_t cnt:4;
    };

    // int8_t byte4[4];
};
struct box32
{
    u32_4byte index;
    void tobuff(cv::Mat &points, cv::Mat &color, const u64_4byte &center)
    {
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
    }
    void tobuff_all_space(cv::Mat &points, cv::Mat &color, const u64_4byte &center);
   
    union voxel pVoxel[CUBEVOXELSIZE * CUBEVOXELSIZE*CUBEVOXELSIZE];

    void init()
    {
        for (size_t i = 0; i < CUBEVOXELSIZE * CUBEVOXELSIZE * CUBEVOXELSIZE; i++)
        {
            pVoxel[i].tsdf = 1.0;
            pVoxel[i].weight = 0.0;
        }
    }

    // void tobox32_pionts()
    // {
    //     struct box32_points pb
    // }
};
struct box32_points
{
    u32_4byte index;
    u64_4byte center;
    std::vector<UPoints> points;

    box32_points(u64_4byte &_center)
    {
        center=_center;
    }
    box32_points()
    {
    }
};
struct kernelPara
{
    u64_4byte center;
    float cam2base[16];
    uint8_t dev_rgbdata[480 * 640][3];
    // uint16_t dev_depthdata[480 * 640];
};
struct exmatcloud_para
{
    u64_4byte center;
    unsigned int dev_points_num = 0;
    // uint8_t buf[512-8-4];
};


struct ex_buf_
{
    union UPoints up[32 * 32 * 32];
    // struct posevoxel pose[32 * 32 * 32];
 
    u64_4byte center;
    unsigned int dev_points_num = 0;
};

#pragma pack(pop)
