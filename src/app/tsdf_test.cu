#include "tsdf.cuh"
#include <set>
#include <vector>
#include "dataset.h"
#include "dataset_tum.h"
#include "dataset_ic.h"

#include "cuda/imgproc.cuh"
#include "cuda/datatype.cuh"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include "mapmanages.cuh"
#include "viewer.h"
#include "tool/Timer.hpp"
#include "cuda/device_array.hpp"
#include <map>
#include "cuda/temp_utils.hpp"
#include <vector_functions.hpp>
#include "cuda/vector_math.hpp"

#include "act.h"

using namespace std;
// cv::Mat getCload(cv::Mat &outdepth, cv::Mat &point_tsdf, std::vector<uint32_t> &cubexs,
//                  cv::Mat_<float> &cam_pose,PosType *host_points, struct kernelPara &para)
// {
//     static bool b = false;
//     cv::Mat wdpoint ;
//     cv::Mat point_cloud; // = Mat::zeros(height, width, CV_32FC3);
//     // = Mat::zeros(height, width, CV_32FC3);
//     static cv::Vec3s maxvec;
//     //point cloud 赋值，其中 fx,fy,cx,cy 为Kinect2 的内参
//     // float fx = cam_K.at<float>(0, 0), fy = cam_K.at<float>(1, 1), cx = cam_K.at<float>(0, 2), cy = cam_K.at<float>(1, 2);

//     for (int row = 0; row < 480; row++)
//         for (int col = 0; col <  640; col++)
//         {
//             //float dz = outdepth.at<float>(row, col);
//             cv::Vec4f vec;
//             // if (dz > 7.0)
//             //     continue;
//             // if (dz < 0.32)
//             // continue;
//             {
//                 vec[0] = host_points[row*480+col].x;
//                 vec[1] = host_points[row*480+col].y;
//                 vec[2] = host_points[row*480+col].z;
//                 vec[3] = 1.0f;

//                 wdpoint.push_back(vec);
//             }
//         }

//     // assert(dep1dm.rows);
//     //  std:: cout<<dep1dm<< std::endl;

//     for (int row = 0; row < wdpoint.rows; row++)
//     {

//         cv::Vec4f &vec = wdpoint.at<cv::Vec4f>(row, 0);
//         // std::cout<<vec[0]<<" "<<vec[1]<<" "<<vec[2]<<std::endl;
//         cv::Vec4s vecs;
//         vecs[0] = std::floor(vec[0] / (float)(VOXELSIZE)); //向下取整
//         vecs[1] = std::floor(vec[1] / (float)(VOXELSIZE)); //向下取整
//         vecs[2] = std::floor(vec[2] / (float)(VOXELSIZE)); //向下取整

//         if (std::abs(vecs[0]) > 128 || std::abs(vecs[1]) > 128 || std::abs(vecs[2]) > 128)
//         {
//             std::cout << vec << std::endl;
//             assert(0);
//         }
//         u32_4byte u64;
//         u64.x = vecs[0] - para.center.x; // - 10;
//         u64.y = vecs[1] - para.center.y; // - para.center.y; // - 10;
//         u64.z = vecs[2] - para.center.z;

//         cubexs.push_back(u64.u32);
//         point_tsdf.push_back(vecs);
//         // if (dz < 1.6 && vec[1] > 0.3)
//     }
//     // assert(0);
//     // std::cout << "max::" << maxvec << " " << para.center.x << " " << para.center.y << std::endl;
//     std::sort(cubexs.begin(), cubexs.end());
//     cubexs.erase(unique(cubexs.begin(), cubexs.end()), cubexs.end());

//     return point_cloud;
// }

// namespace device
// {
// __device__ float dot(const float3& v1, const float3& v2)
// {
//     return __fmaf_rn(v1.x, v2.x, __fmaf_rn(v1.y, v2.y, v1.z*v2.z));
// }

// __device__ Vec3f operator*(const Mat3f& m, const Vec3f& v)
// {
//     return make_float3(dot(m.data[0], v), dot (m.data[1], v), dot (m.data[2], v));
// }

// __device__ Vec3f operator*(const Aff3f& a, const Vec3f& v) {
//     return a.R * v + a.t;
// }

// __device__ Vec3f tr(const float4& v) {
//     return make_float3(v.x, v.y, v.z);
// }

// struct plus
// {
//     __device__ float operator () (float l, float r) const  {
//         return l + r;
//     }
//     __device__ double operator () (double l, double r) const  {
//         return l + r;
//     }
// };

// struct gmem
// {
//     template<typename T> __kf_device__ static T LdCs(T *ptr);
//     template<typename T> __kf_device__ static void StCs(const T& val, T *ptr);
// };

// template<> __kf_device__ ushort2 gmem::LdCs(ushort2* ptr);
// template<> __kf_device__ void gmem::StCs(const ushort2& val, ushort2* ptr);
// }
struct Reprojector
{
    Reprojector() {}
    Reprojector(float fx, float fy, float cx, float cy) : finv(make_float2(1.f / fx, 1.f / fy)), c(make_float2(cx, cy)){};
    float2 finv, c;
    __device__ float3 operator()(int x, int y, float z) const;
};

// Reprojector::Reprojector(float fx, float fy, float cx, float cy)
__device__ float3 Reprojector::operator()(int u, int v, float z) const
{
    float x = z * (u - c.x) * finv.x;
    float y = z * (v - c.y) * finv.y;
    return make_float3(x, y, z);
}

struct Pointcuda
{
    union
    {
        float data[4];
        struct
        {
            float x, y, z;
        };
    };
};

typedef Pointcuda Normal;

// kfusion::device::Reprojector::Reprojector(float fx, float fy, float cx, float cy) : finv(make_float2(1.f/fx, 1.f/fy)), c(make_float2(cx, cy)) {}
__global__ void points_normals_kernel(const Reprojector reproj, const PtrStepSz<ushort> depth, PtrStep<float4> points, PtrStep<float4> normals)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= depth.cols || y >= depth.rows)
        return;

    const float qnan = __int_as_float(0x7fffffff);
    points.ptr(y)[x] = make_float4(qnan, qnan, qnan, qnan);
    normals.ptr(y)[x] = make_float4(qnan, qnan, qnan, qnan);

    if (x >= depth.cols - 1 || y >= depth.rows - 1)
        return;

    // // //mm -> meters
    float z00 = depth.ptr(y)[x] * 0.0002f;
    float z01 = depth.ptr(y)[x + 1] * 0.0002f;
    float z10 = depth.ptr(y + 1)[x] * 0.0002f;

    if (z00 * z01 * z10 != 0)
    {
        float3 v00 = reproj(x, y, z00);
        float3 v01 = reproj(x + 1, y, z01);
        float3 v10 = reproj(x, y + 1, z10);

        float3 n = normalized(cross(v01 - v00, v10 - v00));
        normals.ptr(y)[x] = make_float4(-n.x, -n.y, -n.z, 0.f);
        points.ptr(y)[x] = make_float4(v00.x, v00.y, v00.z, 0.f);
    }
}
// kfusion::device::Reprojector::Reprojector(float fx, float fy, float cx, float cy) : finv(make_float2(1.f/fx, 1.f/fy)), c(make_float2(cx, cy)) {}
// __kf_device__ Vec3f tr(const float4& v) { return ; }
void computePointNormals(const Intr &intr, const DeviceArray2D<unsigned short> &depth, DeviceArray2D<float4> &points, DeviceArray2D<float4> &normals)
{
    points.create(depth.rows(), depth.cols());
    normals.create(depth.rows(), depth.cols());

    dim3 block(32, 8);
    dim3 grid(divUp(depth.cols(), block.x), divUp(depth.rows(), block.y));
    Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    points_normals_kernel<<<grid, block>>>(reproj, depth, points, normals);
    ck(cudaGetLastError());
}

struct RGB
{
    union
    {
        struct
        {
            unsigned char b, g, r;
        };
        int bgra;
    };
};

__global__ void render_image_kernel(const PtrStep<ushort> depth, const PtrStep<float4> normals,
                                    const Reprojector reproj, PtrStepSz<uchar4> dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    float3 color;

    int d = depth.ptr(y)[x];

    if (d == 0)
    {
        const float3 bgr1 = make_float3(4.f / 255.f, 2.f / 255.f, 2.f / 255.f);
        const float3 bgr2 = make_float3(236.f / 255.f, 120.f / 255.f, 120.f / 255.f);

        float w = static_cast<float>(y) / dst.rows;
        color = bgr1 * (1 - w) + bgr2 * w;
    }
    else
    {
        float3 P = reproj(x, y, d * 0.001f);
        float4 v4 = normals.ptr(y)[x];

        float3 N = make_float3(v4.x, v4.y, v4.z);

        const float Ka = 0.3f; // ambient coeff
        const float Kd = 0.5f; // diffuse coeff
        const float Ks = 0.2f; // specular coeff
        const float n = 20.f;  // specular power

        const float Ax = 1.f; // ambient color,  can be RGB
        const float Dx = 1.f; // diffuse color,  can be RGB
        const float Sx = 1.f; // specular color, can be RGB
        const float Lx = 1.f; // light color

        // Ix = Ax*Ka*Dx + Att*Lx [Kd*Dx*(N dot L) + Ks*Sx*(R dot V)^n]

        float3 L = normalized(make_float3(0, 0, 0) - P);
        float3 V = normalized(make_float3(0.f, 0.f, 0.f) - P);
        float3 R = normalized(make_float3(N.x * 2.0f, N.y * 2.0f, N.z * 2.0f) * dot(N, L) - L);

        float Ix = Ax * Ka * Dx + Lx * Kd * Dx * fmax(0.f, dot(N, L)) + Lx * Ks * Sx * __powf(fmax(0.f, dot(R, V)), n);
        color = make_float3(Ix, Ix, Ix);
    }

    uchar4 out;
    out.x = static_cast<unsigned char>(__saturatef(color.x) * 255.f);
    out.y = static_cast<unsigned char>(__saturatef(color.y) * 255.f);
    out.z = static_cast<unsigned char>(__saturatef(color.z) * 255.f);
    out.w = 0;
    dst.ptr(y)[x] = out;
}
__global__ void render_image_kernel(const PtrStep<Point> points, const PtrStep<Normal> normals,
                                    const Reprojector reproj, float3 light_pose, PtrStepSz<uchar4> dst)
{
    // int x = threadIdx.x + blockIdx.x * blockDim.x;
    // int y = threadIdx.y + blockIdx.y * blockDim.y;

    // if (x >= dst.cols || y >= dst.rows)
    //     return;

    // float3 color;

    // float3 p = tr(points.ptr(y)[x]);
    // light_pose=make_float3(0,0,0);
    // if (isnan(p.x))
    // {
    //     const float3 bgr1 = make_float3(4.f/255.f, 2.f/255.f, 2.f/255.f);
    //     const float3 bgr2 = make_float3(236.f/255.f, 120.f/255.f, 120.f/255.f);

    //     float w = static_cast<float>(y) / dst.rows;
    //     color = bgr1 * (1 - w) + bgr2 * w;
    // }
    // else
    // {
    //     float3 P = p;

    //           float4 v4=normals.ptr(y)[x];

    //     float3 N = make_float3(v4.x, v4.y, v4.z);

    //     const float Ka = 0.3f;  //ambient coeff
    //     const float Kd = 0.5f;  //diffuse coeff
    //     const float Ks = 0.2f;  //specular coeff
    //     const float n = 20.f;  //specular power

    //     const float Ax = 1.f;   //ambient color,  can be RGB
    //     const float Dx = 1.f;   //diffuse color,  can be RGB
    //     const float Sx = 1.f;   //specular color, can be RGB
    //     const float Lx = 1.f;   //light color

    //     //Ix = Ax*Ka*Dx + Att*Lx [Kd*Dx*(N dot L) + Ks*Sx*(R dot V)^n]

    //     float3 L = normalized(light_pose - P);
    //     float3 V = normalized(make_float3(0.f, 0.f, 0.f) - P);
    //     float3 R = normalized(2 * N * dot(N, L) - L);

    //     float Ix = Ax*Ka*Dx + Lx * Kd * Dx * fmax(0.f, dot(N, L)) + Lx * Ks * Sx * __powf(fmax(0.f, dot(R, V)), n);
    //     color = make_float3(Ix, Ix, Ix);
    // }

    // uchar4 out;
    // out.x = static_cast<unsigned char>(__saturatef(color.x) * 255.f);
    // out.y = static_cast<unsigned char>(__saturatef(color.y) * 255.f);
    // out.z = static_cast<unsigned char>(__saturatef(color.z) * 255.f);
    // out.w = 0;
    // dst.ptr(y)[x]= out;
}
void bilateralFilter2(const DeviceArray2D<unsigned short> &src, const DeviceArray2D<unsigned short> &dst, int kernel_size,
                      float sigma_spatial, float sigma_depth)
{
    sigma_depth *= 1000; // meters -> mm

    // points.create(depth.rows(), depth.cols());
    // normals.create(depth.rows(), depth.cols());

    dim3 block(32, 8);
    dim3 grid(divUp(src.cols(), block.x), divUp(src.rows(), block.y));
    // dim3 grid (divUp (depth.cols(), block.x), divUp (depth.rows (), block.y));
    // Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    // points_normals_kernel<<<grid, block>>>(reproj, depth, points, normals);
    // ck ( cudaGetLastError () );

    // dim3 block (32, 8);

    // cudaSafeCall( cudaFuncSetCacheConfig (bilateral_kernel, cudaFuncCachePreferL1) );
    device::bilateral_kernel<<<grid, block>>>(src, dst, kernel_size, 0.5f / (sigma_spatial * sigma_spatial), 0.5f / (sigma_depth * sigma_depth));
    ck(cudaGetLastError());
};
void renderImage(const Intr &intr, const DeviceArray2D<unsigned short> &depth,
                 DeviceArray2D<float4> &points, DeviceArray2D<float4> &normals,
                 DeviceArray2D<RGB> &image, RGB *_32buf)
{
    // const device::Depth& d = (const device::Depth&)depth;
    // const device::Normals& n = (const device::Normals&)normals;
    Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);
    // device::Vec3f light = device_cast<device::Vec3f>(light_pose);

    // device::Image& i = (device::Image&)image;
    // device::renderImage(d, n, reproj, light, i);

    // auto light_pose = Vec3f::all(0.f); //meters
    // device::Vec3f light = device_cast<device::Vec3f>(light_pose);

    // , const float3& light_pose
    dim3 block(32, 8);
    dim3 grid(divUp(depth.cols(), block.x), divUp(depth.rows(), block.y));

    render_image_kernel<<<grid, block>>>(depth, normals, reproj, image);
    // cudaSafeCall ( cudaGetLastError () );

    // p_int.download(_32buf,sizeof(u32_4byte)*640);
    // ck(cudaMallocHost((void **)&host_points, sizeof( PosType)*480*640));
    image.download(_32buf, 4 * 640);
    cv::Mat asdsa(480, 640, CV_8UC4, _32buf);
    cv::imshow("a", asdsa);
    cv::waitKey(1);
    // waitAllDefaultStream();
}
struct TSDF
{
    float *gpu_cam_K;
    int first_frame_idx = 60;
    float num_frames = 3010;

    float base2world[4 * 4];
    int im_width = 640;  // 743;// 640;
    int im_height = 480; // 465;//480;
    float voxel_size = 0.01f;
    float trunc_margin;

    float cam_K[3 * 3];
    cv::Mat_<float> cam_K_vec;
    viewer *mp_v;
    TSDF(viewer *v)
    {
        mp_v = v;
        trunc_margin = voxel_size * 5;
    }
    dataset *parser;
    // dataset_tum *parser;

    void loop()
    {
        mapmanages mm;
        cv::FileStorage fs("../config.yaml", cv::FileStorage::READ);
        fs["cam_K_vec"] >> cam_K_vec;
        std::cout << cam_K_vec << std::endl;
        cudaMalloc(&gpu_cam_K, 3 * 3 * sizeof(float));
        cudaMemcpy(gpu_cam_K, cam_K_vec.data, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDA(cudaGetLastError());

        struct Voxel32 **host_boxptr = new struct Voxel32 *[ACTIVATE_VOXNUM];
        // int **p = new int* [5];
        struct Voxel32 **dev_boxptr;
        cudaMalloc((void **)&dev_boxptr, sizeof(struct Voxel32 *) * ACTIVATE_VOXNUM);
        cv::Mat points, color;

        struct kernelPara *gpu_kpara;
        cudaMalloc((void **)&gpu_kpara, sizeof(struct kernelPara));
        Timer tm;
        DeviceArray2D<PosType> depthScaled(480, 640);
        DeviceArray2D<PosType> gocloud(480, 640);

        PosType *host_points; //=new PosType;
        ck(cudaMallocHost((void **)&host_points, sizeof(PosType) * 480 * 640));
        // std::cout<<sizeof( PosType)<<std::endl;

        //
        DeviceArray2D<unsigned short> depth_device_img(480, 640);
        DeviceArray2D<unsigned short> device_depth_src(480, 640);

        DeviceArray2D<float4> points_pyr, normals_pyr;
        DeviceArray2D<u32_4byte> p_int(480, 640);
        RGB *host_32buf;
        DeviceArray2D<RGB> imgcuda(480, 640);
        ck(cudaMallocHost((void **)&host_32buf, sizeof(RGB) * 480 * 640));

        Intr intr(cam_K_vec);
        // cam_pose_str _cam;
        intr.print();
        int mode = 0;
        int save_skip_list_num;
        fs["mode"] >> mode;
        fs["save_skip_list_num"] >> save_skip_list_num;
        int show_cloud;
        fs["show_cloud"] >> show_cloud;

        float *g_cam;
        cudaMalloc((void **)&g_cam, sizeof(float) * 16);

        uint32_t *_32buf;
        ck(cudaMallocHost((void **)&_32buf, sizeof(u32_4byte) * 480 * 640));

        // std::cout<<show_cloud<<std::endl;
        // dataset_tum *parser = new dataset(0,-1,1,fs["matpose"]);
        parser = new dataset_ic(1, -1, 1, fs["matpose"]);
        std::map<uint32_t, bool> grids_count;
        // bool tf = false;
        for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; frame_idx += 1)
        { // ++frame_idx
            // tm.Start();

            std::cout << "frame_idx:" << frame_idx << std::endl;
            bool over_ = parser->ReadNextTUM(frame_idx);
            if (!over_)
            {
                mm.exmatcloud(mm.cpu_kpara.center);
                exmatcloudply222("out.ply", mm.curr_point, mm.curr_color);
                // mm.savetoply("out.ply");
                // parser->Mat_save_by_binary(points, cv::format("pc/%04d.point", frame_idx));
                // parser->Mat_save_by_binary(color, cv::format("pc/%04d.color", frame_idx));
                // if (points.rows > 0)
                //     mp_v->inset_cloud("curr", cv::viz::WCloud(points, color));
                // cv::Affine3f affpose(cam_pose);
                break;
            }

            // while  (1);
            memcpy(mm.cpu_kpara.dev_rgbdata, parser->rgb_.data, parser->rgb_.rows * parser->rgb_.cols * 3);
            device_depth_src.upload(parser->depth_src.data, parser->depth_src.step, parser->depth_src.rows, parser->depth_src.cols);
            //双边滤波
            bilateralFilter2(device_depth_src, depth_device_img, 7, 4.5f, 0.04f);
            cv::Mat point_tsdf;
            std::vector<uint32_t> cubexs;
            cv::Mat_<float> cam_pose(4, 4, &parser->m_pose.val[0]);
            cam_pose.reshape(1, 4);
            memcpy(mm.cpu_kpara.cam2base, &parser->m_pose.val[0], 4 * 4 * sizeof(float));

            cudaMemcpy((void *)g_cam, (void *)(&parser->m_pose.val[0]), sizeof(float) * 16, cudaMemcpyHostToDevice);

            // std::cout <<std::endl<<cam_pose<<endl;
            checkCUDA(cudaGetLastError());

            dim3 block_scale(32, 8);
            dim3 grid_scale(divUp(parser->depth_src.cols, block_scale.x), divUp(parser->depth_src.rows, block_scale.y));
            // scales depth along ray and converts mm -> meters.

            //
            //
            // depthScaled.download(pnormal,sizeof(PosType)*640);
            //
            device::scaleDepth<<<grid_scale, block_scale>>>(depth_device_img, depthScaled, gocloud, p_int, g_cam, intr);
            checkCUDA(cudaGetLastError());
            {
                // computePointNormals(intr, depth_device_img, points_pyr, normals_pyr);
                // renderImage(intr, depth_device_img, points_pyr, normals_pyr, imgcuda, host_32buf);
            }
            gocloud.download(host_points, 12 * 640);
            p_int.download(_32buf, sizeof(u32_4byte) * 640);

            std::set<uint32_t> set32(_32buf, _32buf + 480 * 640);
            // cout<<set32.size()<<endl;

            //当前深度图 点云
            cv::Mat asdp(480 * 640, 1, CV_32FC3, &host_points[0].x);
            mp_v->inset_depth2(asdp, cv::Affine3f::Identity());
            cv::waitKey(1);

            cudaMemcpy((void *)gpu_kpara, (void *)(&mm.cpu_kpara), sizeof(struct kernelPara), cudaMemcpyHostToDevice);
            checkCUDA(cudaGetLastError());
            int i = 0;
            int cntt = 0;
            for (std::set<uint32_t>::iterator it = set32.begin(); it != set32.end(); ++it)
            {
                uint32_t indexa = *it & 0xffffff; // box相对坐标
                if ((mm.pboxs)[indexa] == 0)
                { //
                    host_boxptr[i] = mm.getidlebox(indexa);
                    (mm.pboxs)[indexa] = host_boxptr[i];
                    cntt++;
                }
                else
                {
                    host_boxptr[i] = (mm.pboxs)[indexa];
                }
                u32_4byte u32;
                u32.u32 = indexa;
                // u32.type = 0x1;
                u32.cnt = 0xf;
                ck(cudaMemcpy((void *)&host_boxptr[i]->index, (void *)(&u32), sizeof(uint32_t), cudaMemcpyHostToDevice));
                i++;
                grids_count[indexa] = true;
            }
            // std::cout << mm.cpu_box.size() << std::endl;
            // if(mm.pskipList->size()>save_skip_list_num)
            // {
            //     mm.savefile();
            //     mp_v->is_exit.store(false);
            //     mp_v->m_cond.notify_all();
            //     break;
            // }
            assert(i != 0);
            //将需要处理的box地址拷贝到GPU
            cudaMemcpy((void *)dev_boxptr, (void *)host_boxptr, (i) * sizeof(struct Voxel32 *), cudaMemcpyHostToDevice);
            checkCUDA(cudaGetLastError());
            // std::cout << "frame_idx: " << frame_idx << " cnt:i=" << i << " " <<  << std::endl;
            // assert(num != 0);
            // printf("num=%d\n", num);

            dim3 grid(i - 1, 1, 1), block(32, 32, 1); // 设置参数
            device::Integrate32<<<grid, block>>>(gpu_cam_K,
                                                 im_height, im_width, voxel_size, trunc_margin,
                                                 dev_boxptr, gpu_kpara, depthScaled);
            // cudaDeviceSynchronize();
            checkCUDA(cudaGetLastError());
            // float atime=tm.ElapsedMicroSeconds()*0.001f;

            // atime22=tm.ElapsedMicroSeconds()*0.001f;
            float atime[5] = {0};
            tm.Start();

            //显示当前点云 true false
            if (show_cloud == 1)
            {
                points = cv::Mat();
                color = cv::Mat();
                mm.exmatcloud(mm.cpu_kpara.center);
                // parser->Mat_save_by_binary(points, cv::format("pc/%04d.point", frame_idx));
                // parser->Mat_save_by_binary(color, cv::format("pc/%04d.color", frame_idx));
                if (points.rows > 0)
                    mp_v->inset_cloud("curr1", cv::viz::WCloud(mm.curr_point, mm.curr_color));
                // cv::Affine3f affpose(cam_pose);
                // mp_v->inset_depth(dep, cv::Affine3f::Identity());//affpose);//cv::Affine3f::Identity());
                // cv::waitKey(1);
            }
            atime[2] = tm.ElapsedMicroSeconds() * 0.001f;

            // cudaStreamSynchronize();

            //移除
            if (mm.gpu_pbox_free.size() < 100) // frame_idx % 100 == 15 ||
            {
                tm.Start();
                mm.savenode_cube_();
                atime[0] = tm.ElapsedMicroSeconds() * 0.001f;
                mm.cpu_kpara.center.x = std::floor(parser->m_pose.val[3] * 3.125f);
                mm.cpu_kpara.center.y = std::floor(parser->m_pose.val[7] * 3.125f);
                mm.cpu_kpara.center.z = std::floor(parser->m_pose.val[11] * 3.125f);
                // mm.movenode(mm.cpu_kpara.center);
                tm.Start();
                mm.resetnode();
                atime[1] = tm.ElapsedMicroSeconds() * 0.001f;

                // std::cout<<""<<atime[1]<<","<<atime[0]<<std::endl;
            }
            char key_val = cv::waitKey(1);
            if (key_val == 's')
            {
                cv::waitKey();
            }
            // cv::Mat shot =mp_v->getScreenshot();
            // cv::imshow("shot",shot);
            // cv::imwrite(cv::format("shot%d.png",frame_idx),shot);
            //显示轨迹 debug信息
            if (true)
            {
                mp_v->inset_traj(parser->m_pose);
                //  while(1);
            }
            std::string debugtext = cv::format("Frame_id:%d remain box:%ld period:%.4f ms", frame_idx, mm.gpu_pbox_free.size(),
                                               tm.ElapsedMicroSeconds() * 0.001f);
            debugtext += cv::format(" slist:%d", mm.pskipList->size());
            mp_v->setstring(debugtext);
            // std::cout<<frame_idx<<","<<mm.gpu_pbox_use.size()+mm.pskipList->size()<<","<<atime[2]<<","<<tm.ElapsedMicroSeconds()*0.001f<<","<<cntt<<std::endl;
            // std::cout<<frame_idx<<","<<mm.gpu_pbox_use.size()<<","<<atime<<","<<atime2<<std::endl;
        }
        fs.release();

        mp_v->pthd.join();
        // exmatcloudply(points, color);
    }
    void exmat_img(cv::Mat &points, cv::Mat &color, cv::Mat &rgb, cv::Mat &depth)
    {
        rgb = cv::Mat::zeros(im_height, im_width, CV_8UC3);
        cv::Mat _dep = cv::Mat_<float>::zeros(im_height, im_width);

        for (int i = 0; i < points.rows; i++)
        {
            cv::Vec3f pt = points.at<cv::Vec3f>(i, 0);

            // 计算小体素的世界坐标
            float pt_base_x = pt[0];
            float pt_base_y = pt[1];
            float pt_base_z = pt[2];

            float *cam2base = parser->m_pose.val;

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
            cv::Vec3b rg = color.at<cv::Vec3b>(i, 0);
            rgb.at<cv::Vec3b>(pt_pix_y, pt_pix_x) = rg;
            _dep.at<float>(pt_pix_y, pt_pix_x) = pt_cam_z;
        }
        _dep.convertTo(depth, CV_16U, 1000);
    }
    void excloud()
    {
    }
};
int main()
{
    cudaDeviceReset();
    viewer v;
    struct TSDF tsdf(&v);
    tsdf.loop();
    // tsdf.excloud();
    return 0;
}