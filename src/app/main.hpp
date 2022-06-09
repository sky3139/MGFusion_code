#pragma once
#include <cuda.h>
#include <iostream>
#include <math_constants.h>
// #include "math.hpp"
#include <opencv2/viz/vizcore.hpp>
#include "../cuda/datatype.cuh"
#include "./cuVector.cuh"

typedef float4 POINT;
typedef struct Voxel32 *PVOXEL;
class raycast
{
public:
    Patch<POINT> points, norm;
    Patch<uchar4> dst;
    cv::Mat raycastimg; cv::Mat point_; cv::Mat normal_img;

    ~raycast();
    raycast();
    int mgraycast_test(float4 cam, PVOXEL *&g_hashmap, cv::Affine3f camera_pose);
};
