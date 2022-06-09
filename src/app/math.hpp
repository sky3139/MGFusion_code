#pragma once

#define __kf_hdevice__ __host__ __device__ __forceinline__
#define __kf_device__ __device__ __forceinline__

// __kf_device__ float dot(const float3 &v1, const float3 &v2)
// {
//     return __fmaf_rn(v1.x, v2.x, __fmaf_rn(v1.y, v2.y, v1.z * v2.z));
// }

// __kf_device__ float3 &operator+=(float3 &vec, const float &v)
// {
//     vec.x += v;
//     vec.y += v;
//     vec.z += v;
//     return vec;
// }

// __kf_device__ float3 &operator+=(float3 &v1, const float3 &v2)
// {
//     v1.x += v2.x;
//     v1.y += v2.y;
//     v1.z += v2.z;
//     return v1;
// }

// __kf_device__ float3 operator+(const float3 &v1, const float3 &v2)
// {
//     return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
// }

// __kf_device__ float3 operator*(const float3 &v1, const float3 &v2)
// {
//     return make_float3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
// }



// __kf_device__ float3 operator/(const float3 &v1, const float3 &v2)
// {
//     return make_float3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
// }


// __kf_device__ float3 &operator*=(float3 &vec, const float &v)
// {
//     vec.x *= v;
//     vec.y *= v;
//     vec.z *= v;
//     return vec;
// }

__kf_device__ float3 operator-(const float3 &v1, const float3 &v2)
{
    return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__kf_hdevice__ float3 operator*(const float3 &v1, const float &v)
{
    return make_float3(v1.x * v, v1.y * v, v1.z * v);
}

__kf_hdevice__ float3 operator*(const float &v, const float3 &v1)
{
    return make_float3(v1.x * v, v1.y * v, v1.z * v);
}
__kf_device__ float3 normalized(const float3 &v)
{
    return v * rsqrt(dot(v, v));
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
