#pragma once
#include <cuda.h>
#include <iostream>
#include <math_constants.h>
// #include "math.hpp"
#include <opencv2/viz/vizcore.hpp>
#include "../cuda/datatype.cuh"
#include "../cuda/datatype.cuh"
// typedef uchar4 RGB;
typedef float4 POINT;
typedef struct Voxel32 *PVOXEL;
class raycast
{
public:
    Patch<POINT> points, norm;
    Patch<RGB> dst;
};

int mgraycast_test(cv::Mat &raycastimg, cv::Mat &point_, cv::Mat &normal_img, float4 cam, PVOXEL *&g_hashmap, cv::Affine3f camera_pose);
