#include "tsdf.cuh"
#include <set>
#include <vector>

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
#include "app/cuVector.cuh"
#include "../../read.hpp"
#include "app/main.hpp"

#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include "../tool/tree.h"
using namespace std;

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
__global__ void points_normals_kernel(const Reprojector reproj, const Patch<ushort> depth, Patch<float4> points, Patch<float4> normals)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= depth.cols || y >= depth.rows)
        return;

    const float qnan = __int_as_float(0x7fffffff);
    points(y, x) = make_float4(qnan, qnan, qnan, qnan);
    normals(y, x) = make_float4(qnan, qnan, qnan, qnan);

    if (x >= depth.cols - 1 || y >= depth.rows - 1)
        return;

    // // //mm -> meters
    float z00 = depth(y, x) * 0.0002f;
    float z01 = depth(y, x + 1) * 0.0002f;
    float z10 = depth(y + 1, x) * 0.0002f;

    if (z00 * z01 * z10 != 0)
    {
        float3 v00 = reproj(x, y, z00);
        float3 v01 = reproj(x + 1, y, z01);
        float3 v10 = reproj(x, y + 1, z10);

        float3 n = normalized(cross(v01 - v00, v10 - v00));
        normals(y, x) = make_float4(-n.x, -n.y, -n.z, 0.f);
        points(y, x) = make_float4(v00.x, v00.y, v00.z, 0.f);
    }
}
// kfusion::device::Reprojector::Reprojector(float fx, float fy, float cx, float cy) : finv(make_float2(1.f/fx, 1.f/fy)), c(make_float2(cx, cy)) {}
// __kf_device__ Vec3f tr(const float4& v) { return ; }
void computePointNormals(const Intr &intr, const Patch<unsigned short> &depth, Patch<float4> &points, Patch<float4> &normals)
{
    points.create(depth.rows, depth.cols);
    normals.create(depth.rows, depth.cols);

    dim3 block(32, 8);
    dim3 grid(divUp(depth.cols, block.x), divUp(depth.rows, block.y));
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

__global__ void render_image_kernel(const Patch<ushort> depth, const Patch<float4> normals,
                                    const Reprojector reproj, Patch<RGB> dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    float3 color;

    const ushort d = depth(y, x);

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
        float4 v4 = normals(y, x);

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

    RGB out;
    out.b = static_cast<unsigned char>(__saturatef(color.x) * 255.f);
    out.g = static_cast<unsigned char>(__saturatef(color.y) * 255.f);
    out.r = static_cast<unsigned char>(__saturatef(color.z) * 255.f);
    // out.w = 0;
    dst(y, x) = out;
}

void bilateralFilter2(const Patch<unsigned short> &src, const Patch<unsigned short> &dst, int kernel_size,
                      float sigma_spatial, float sigma_depth)
{
    sigma_depth *= 1000; // meters -> mm

    // points.create(depth.rows, depth.cols);
    // normals.create(depth.rows, depth.cols);

    dim3 block(32, 8);
    dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));
    // dim3 grid (divUp (depth.cols, block.x), divUp (depth.rows (), block.y));
    // Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    // points_normals_kernel<<<grid, block>>>(reproj, depth, points, normals);
    // ck ( cudaGetLastError () );
    // dim3 block (32, 8);
    device::bilateral_kernel<<<grid, block>>>(src, dst, kernel_size, 0.5f / (sigma_spatial * sigma_spatial), 0.5f / (sigma_depth * sigma_depth));
    ck(cudaGetLastError());
};
void renderImage(const Intr &intr, const Patch<unsigned short> &depth,
                 Patch<float4> &points, Patch<float4> &normals,
                 Patch<RGB> &image, RGB *_32buf)
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
    dim3 grid(divUp(depth.cols, block.x), divUp(depth.rows, block.y));

    render_image_kernel<<<grid, block>>>(depth, normals, reproj, image);
    // cudaSafeCall ( cudaGetLastError () );

    // p_int.download(_32buf,sizeof(u32B4)*640);
    // ck(cudaMallocHost((void **)&host_points, sizeof( float3)*480*640));
    image.download(_32buf, 4 * 640);
    cv::Mat asdsa(480, 640, CV_8UC4, _32buf);
    cv::imshow("a", asdsa);
    cv::waitKey(1);
    // waitAllDefaultStream();
}
struct kEqual
{
    __host__ __device__ bool operator()(uint32_t x, uint32_t y)
    {
        return x > y; //&& (x.get<1>() == y.get<1>())
    }
};
__global__ void seta(uint32_t *ptr, uint32_t *len)
{
    uint32_t last = ptr[0];
    int tx = threadIdx.x;
    // int tx = threadIdx.x;
    int j = 0;
    int sum = 48 * 6400;
    int base = tx * 48 * 6400;
    for (int i = 1 + base; i < sum + base; i++)
    {
        if (last == ptr[i])
        {
            continue;
        }
        unsigned int val = atomicInc(&len[2000], 0xffffff);
        // len[val] = last;
        len[val] = last;
        last = ptr[i];
    }
    // printf("%d %d\n", j, len[val]);

    // printf("%d %d\n", j, *len);
}
#define BITS_PER_WORD 32
#define MASK 0x1f
#define SHIFT 5
// #define BITS_PER_WORD 8
// #define MASK 0x07
// #define SHIFT 3
// #define BITS_PER_WORD 64
// #define MASK 0x3f
// #define SHIFT 6

// BITS_PER_WORD 与 MASK、SHIFT 是相匹配的，
// 如果 BITS_PER_WORD 为 8，则 SHIFT 为 3，MASK 为 0x07
// 如果 BITS_PER_WORD 为 64，则 SHIFT 为 6，MASK 为 0x3f
// 同样的存储位图的数组的元素类型也要发生相应的改变，BITS_PER_WORD == 8，char
// BITS_PER_WORD == 64, ⇒ long long
#define N 256 * 256 * 256
// int a[1 + N / BITS_PER_WORD];

__global__ void read_k(uint32_t *out, uint32_t *bitmap)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    for (size_t tar = 0; tar < 256; tar++)
    {
        uint32_t tarr = tar + 256 * 256 * bx + tx * 256;
        bool ret = bitmap[tarr >> SHIFT] & (1 << (tarr & MASK));
        if (ret)
        {
            unsigned int val = atomicInc(&out[2000], 0xffffff);
            out[val] = tarr;
            printf("%d\n", 1);
        }
    }
}
struct bit
{

    typedef uint32_t maptype;
    maptype *d_bitmap;
    // maptype *h_bitmap;
    maptype *curr_bitmap;
    //[1 + N / BITS_PER_WORD];
    bit()
    {
        ck(cudaMalloc((void **)&d_bitmap, (1 + N / BITS_PER_WORD) * sizeof(maptype)));
        // ck(cudaMallocHost((void **)&h_bitmap, (1 + N / BITS_PER_WORD) * sizeof(maptype)));
        curr_bitmap = d_bitmap;
        clr();
    }
    inline __device__ __host__ void set(const uint32_t i)
    {
        curr_bitmap[i >> SHIFT] |= (1 << (i & MASK));
    }
    // a[i >> SHIFT] ⇒ 返回的是int整型，也是长度为 32 的 bit 比特串；
    inline __device__ __host__ void clr(const uint32_t i)
    {
        curr_bitmap[i >> SHIFT] &= ~(1 << (i & MASK));
    }

    inline __device__ __host__ bool read(const uint32_t i) const
    {
        return curr_bitmap[i >> SHIFT] & (1 << (i & MASK));
    }
    __host__ void clr()
    {
        cudaMemset(d_bitmap, 0, (1 + N / BITS_PER_WORD) * sizeof(maptype));
        // memset(h_bitmap, 0, (1 + N / BITS_PER_WORD) * sizeof(maptype));
        // curr_bitmap = h_bitmap;
    }
    // __host__ void tocpu()
    // {
    //     ck(cudaMemcpy((void *)d_bitmap, h_bitmap, (1 + N / BITS_PER_WORD) * sizeof(maptype), cudaMemcpyDeviceToHost)); //上传位姿
    // }
    // __host__ void togpu()
    // {
    //     ck(cudaMemcpy((void *)d_bitmap, h_bitmap, (1 + N / BITS_PER_WORD) * sizeof(maptype), cudaMemcpyHostToDevice)); //上传位姿
    //     curr_bitmap = d_bitmap;
    // }
};
__global__ void readk(uint32_t *out, struct bit gb)
{
    int bx = 256 * 256 * blockIdx.x;
    int tx = threadIdx.x * 256;
    for (size_t tar = 0; tar < 256; tar++)
    {
        uint32_t tarr = tar + bx + tx;
        bool ret = gb.read(tarr); // bitmap[tarr >> SHIFT] & (1 << (tarr & MASK));

        if (ret)
        {
            unsigned int val = atomicInc(&out[2000], 0xffffff);
            out[val] = tarr;
        }
    }
}
class TSDF
{
public:
    float4 gpu_cam_K;
    int first_frame_idx = 1;
    float num_frames = 3010;

    float base2world[4 * 4];
    int im_width = 640;  // 743;// 640;
    int im_height = 480; // 465;//480;
    float voxel_size = 0.01f;
    float trunc_margin;

    float cam_K[3 * 3];
    viewer *mp_v;
    TSDF(viewer *v)
    {
        mp_v = v;
        trunc_margin = voxel_size * 5;
    }
    DataSet<float> *parser;
    // dataset_tum *parser;

    void loop()
    {
        mapmanages mm;
        cv::FileStorage fs("../config.yaml", cv::FileStorage::READ);
        parser = new DataSet<float>(fs["matpose"]);
        gpu_cam_K = make_float4(parser->fx, parser->fy, parser->cx, parser->cy);
        checkCUDA(cudaGetLastError());

        vector<struct Voxel32 *> host_boxptr(ACTIVATE_VOXNUM); // struct Voxel32 *[ACTIVATE_VOXNUM];
        struct Voxel32 **dev_boxptr;
        cudaMalloc((void **)&dev_boxptr, sizeof(struct Voxel32 *) * ACTIVATE_VOXNUM);
        cv::Mat points, color;

        struct kernelPara *gpu_kpara;
        cudaMalloc((void **)&gpu_kpara, sizeof(struct kernelPara));
        Timer tm;
        Patch<float3> depthScaled(480, 640);
        Patch<float3> gocloud(480, 640);

        float3 *host_points; //=new float3;
        ck(cudaMallocHost((void **)&host_points, sizeof(float3) * 480 * 640));
        // std::cout<<sizeof( float3)<<std::endl;

        //
        Patch<unsigned short> depth_device_img(480, 640);
        Patch<unsigned short> device_depth_src(480, 640);

        Patch<float4> points_pyr, normals_pyr;
        Patch<uint32_t> p_int(480, 640);
        RGB *host_32buf;
        Patch<RGB> imgcuda(480, 640);
        ck(cudaMallocHost((void **)&host_32buf, sizeof(RGB) * 480 * 640));

        Intr intr(parser->cam_K);
        intr.sca = 1.0f / parser->depth_factor;
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
        ck(cudaMallocHost((void **)&_32buf, sizeof(u32B4) * 480 * 640));
        raycast ray;

        struct device::Tnte fun;
        fun.intr = gpu_cam_K;
        fun.im_height = im_height;
        fun.im_width = im_width;
        fun.voxel_size = voxel_size;
        fun.trunc_margin = trunc_margin;
        thrust::device_vector<uint32_t> kset(640 * 480, 0);
        uint32_t *hv_ptr; // = thrust::raw_pointer_cast(kset.data());

        thrust::device_vector<uint32_t> out(2048, 0);
        uint32_t *outprt = thrust::raw_pointer_cast(out.data());
        // uint32_t *zin_number;
        (cudaMallocHost((void **)&hv_ptr, 640 * 480 * sizeof(uint32_t)));

        CBTInserter ci(ACTIVATE_VOXNUM);
        struct bit b;
        for (int frame_idx = 0; frame_idx < parser->pose.frames - 1; frame_idx += 1)
        {
            // std::cout << "frame_idx:" << frame_idx << std::endl;
            bool over_ = parser->ReadNextTUM(frame_idx);
            // parser->m_pose.val[3] -= 15.997225f;
            // parser->m_pose.val[7] -= -1.722280;
            // parser->m_pose.val[11] -= 8.929637;
            if (!over_)
            {
                // cout << "over" << endl;
                // mm.save_tsdf_mode_grids(fs["matpose"]);
                // mm.exmatcloud(mm.cpu_kpara.center);
                // string savename = string(fs["matpose"]) + "ours_new.ply";
                // mm.savetoply("out.ply");
                // parser->Mat_save_by_binary(points, cv::format("pc/%04d.point", frame_idx));
                // parser->Mat_save_by_binary(color, cv::format("pc/%04d.color", frame_idx));
                // if (points.rows > 0)
                //     mp_v->inset_cloud("curr", cv::viz::WCloud(points, color));
                // cv::Affine3f affpose(cam_pose);
                //           mm.gpu_para->extall = true;
                // mm.exmatcloud(mm.cpu_kpara.center);
                mm.saveallnode();
                mm.SaveVoxelGrid2SurfacePointCloud(cv::format("pc/over%d", frame_idx));
                cout << cv::format("pc/over%d", frame_idx) << endl;
                assert(0);
                break;
            }
            tm.Start();
            b.clr();
            ck(cudaMemcpy(mm.cpu_kpara.dev_rgbdata, parser->rgb_.data, parser->rgb_.rows * parser->rgb_.cols * 3, cudaMemcpyHostToDevice)); //上传彩色图像到GPU
            device_depth_src.upload(parser->depth_src.ptr<uint16_t>(), parser->depth_src.step);                                             //上传深度图
            bilateralFilter2(device_depth_src, depth_device_img, 7, 4.5f, 0.04f);                                                           //双边滤波
            memcpy(mm.cpu_kpara.cam2base, &parser->m_pose.val[0], 4 * 4 * sizeof(float));                                                   //上传位姿
            ck(cudaMemcpy((void *)g_cam, (void *)(&parser->m_pose.val[0]), sizeof(float) * 16, cudaMemcpyHostToDevice));                    //上传位姿
            dim3 block_scale(32, 32);
            dim3 grid_scale(divUp(parser->depth_src.cols, block_scale.x), divUp(parser->depth_src.rows, block_scale.y));
            // depthScaled.download(pnormal,sizeof(float3)*640);
            {
                // Timer seta2("gpuread");
                // Timer seta2("scaleDepth");
                device::scaleDepth<<<grid_scale, block_scale>>>(hv_ptr, depth_device_img, depthScaled, gocloud, intr, mm.cpu_kpara, b.d_bitmap); //深度图预处理
                ck(cudaDeviceSynchronize());
                {
                    hv_ptr[2000] = 0;
                    // b.togpu();
                    readk<<<256, 256>>>(hv_ptr, b);
                    ck(cudaDeviceSynchronize());
                }
            }
            // cout << hv_ptr[2000] << endl;
            // computePointNormals(intr, depth_device_img, points_pyr, normals_pyr);
            // renderImage(intr, depth_device_img, points_pyr, normals_pyr, imgcuda, host_32buf);

            // for (k = 0; k < 640 * 480; k++)
            //     ci.insert(_32buf[k]);
            // tm.PrintSeconds("a");
            // {
            //     Timer t("sett");
            //     std::set<uint32_t> set32(_32buf, _32buf + 640 * 480);
            // }
            // for (thrust::device_vector<uint32_t>::iterator it = kset.begin(); it != kvend; it++)
            // {
            //     set3222.insert(*it);
            // }
            // // 当前深度图 点云
            //  gocloud.download(host_points, 12 * 640); //当前帧的点云
            // cv::Mat asdp(480 * 640, 1, CV_32FC3, &host_points[0].x);
            // mp_v->inset_depth2(asdp, cv::Affine3f::Identity());
            // cv::waitKey(1);
            int i = 0;
            for (int k = 0; k < hv_ptr[2000]; k++)
            {
                uint32_t indexa = hv_ptr[k] & 0xffffff; // box相对坐标 取前24位
                if (indexa == 0)                        // || last == aaaaa[i])   //相机原点和无效深度点忽略，
                    continue;
                // last = aaaaa[i];
                // cout << aaaaa[i] << " " << it << endl;

                if ((mm.pboxs)[indexa] == 0) //此空间未初始化，从记忆库拿
                {
                    host_boxptr[i] = mm.getidlebox(indexa);
                    (mm.pboxs)[indexa] = host_boxptr[i];

                    u32B4 u32;
                    u32.u32 = indexa;
                    // u32.type = 0x1;

                    ck(cudaMemcpyAsync((void *)&host_boxptr[i]->index, (void *)(&u32), sizeof(uint32_t), cudaMemcpyHostToDevice));
                }
                else //已经重建过
                {
                    host_boxptr[i] = (mm.pboxs)[indexa];
                }
                i++;
                if (i >= ACTIVATE_VOXNUM - 2)
                    break;
            }
            // cout << "i=" << i << " " << 0 << " " << endl;
            assert(i != 0), assert(i < ACTIVATE_VOXNUM);

            //将需要处理的box地址拷贝到GPU
            ck(cudaMemcpy((void *)dev_boxptr, (void *)&host_boxptr[0], (i) * sizeof(struct Voxel32 *), cudaMemcpyHostToDevice));
            {

                dim3 grid(i, 1, 1), block(32, 32, 1); // 设置参数
                device::Integrate32F<<<grid, block>>>(fun,
                                                      dev_boxptr, mm.cpu_kpara, depthScaled);
                ck(cudaDeviceSynchronize());
            }
            // ci.reset();

            std::string debugtext = cv::format("Frame_id:%d remain box:%ld period:%.4f ms", frame_idx, mm.gpu_pbox_free.size(),
                                               tm.ElapsedMicroSeconds() * 0.001f) +
                                    cv::format(" cloudBoxs:%ld", mm.mcps.mp_cpuVoxel32.size());
            // mp_v->setstring(debugtext);
            cout << debugtext << endl;
            //显示当前点云 true false
            if (show_cloud == 1 || frame_idx % 50 == 30)
            {
                // {
                // mm.gpu_para->extall = true;
                // mm.exmatcloud(mm.cpu_kpara.center);
                // }
                // // if (mm.gpu_pbox_free.size() < 1500 || frame_idx % 50 == 30)
                // if (mm.curr_point.rows > 0)
                //     mp_v->inset_cloud("curr1", cv::viz::WCloud(mm.curr_point, mm.curr_color)); // cv::viz::Color::red())); // color

                // parser->Mat_save_by_binary(points, cv::format("pc/%04d.point", frame_idx));
                // parser->Mat_save_by_binary(color, cv::format("pc/%04d.color", frame_idx));

                // if (mm.all_point[2].rows > 0)
                //     mp_v->inset_cloud("all2", cv::viz::WCloud(mm.all_point[2], cv::viz::Color::blue())); // color
                // cv::Affine3f affpose(cam_pose);
                // mp_v->inset_depth(dep, cv::Affine3f::Identity());//affpose);//cv::Affine3f::Identity());
                // mp_v->inset_traj(parser->m_pose);
                // char key = cv::waitKey(0);
                // cv::Mat bt = mp_v->getScreenshot();
                // cv::imwrite("bt.png", bt);
                // string savename = string(fs["matpose"]) + "ours_new22.ply";
            }
            if (mm.gpu_pbox_free.size() < 500 || frame_idx % 50 == 30)
            {
                     mm.gpu_para->extall = true;
                mm.exmatcloud(mm.cpu_kpara.center);

                u64B4 src_center = mm.cpu_kpara.center;
                mm.cpu_kpara.center.x = std::floor(parser->m_pose.val[3] * 3.125f);
                mm.cpu_kpara.center.y = std::floor(parser->m_pose.val[7] * 3.125f);
                mm.cpu_kpara.center.z = std::floor(parser->m_pose.val[11] * 3.125f);
                u64B4 DS_N = src_center - mm.cpu_kpara.center;
                mm.movenode_6_28(DS_N, src_center, mm.cpu_kpara.center); //转换坐标后在范围内就保留 不在范围内就移除，超过次数也移除

                mm.move2center(DS_N, src_center, mm.cpu_kpara.center);

                // if (mm.all_point[0].rows > 0)
                //     // mm.SaveVoxelGrid2SurfacePointCloud(cv::format("pc/%d", frame_idx));
                //     mp_v->inset_cloud("all", cv::viz::WCloud(mm.all_point[0], mm.all_point[1])); // cv::viz::Color::blue())); //  color
            }

            // atime[2] = ;
            // std::cout << tm.ElapsedMicroSeconds() << std::endl;
            // cudaStreamSynchronize();
            // cudaFreeHost(aaaaa);
            // 移除
            // if (mm.gpu_pbox_free.size() < 1500 || frame_idx % 50 == 30)
            // {
            //     // mm.gpu_para->extall = true;
            //     // mm.exmatcloud(mm.cpu_kpara.center);
            //     // mm.gpu_para->extall = false;
            //     // if (mm.curr_point.rows > 0)
            //     //     mp_v->inset_cloud(cv::format("curr1%d", frame_idx), cv::viz::WCloud(mm.curr_point, mm.curr_color));

            //     u64B4 src_center = mm.cpu_kpara.center;
            //     mm.cpu_kpara.center.x =0;// std::floor(parser->m_pose.val[3] * 3.125f);
            //     mm.cpu_kpara.center.y =0;//  std::floor(parser->m_pose.val[7] * 3.125f);
            //     mm.cpu_kpara.center.z = 0;// std::floor(parser->m_pose.val[11] * 3.125f);
            //     u64B4 DS_N = src_center - mm.cpu_kpara.center;
            //     src_center.print();
            //     mm.cpu_kpara.center.print();
            //     DS_N.print();
            //     mm.movenode_6_28(DS_N, src_center, mm.cpu_kpara.center); //转换坐标后在范围内就保留 不在范围内就移除，超过次数也移除
            //     // mm.exmatcloud(mm.cpu_kpara.center);
            //     // if (mm.curr_point.rows > 0)
            //     //     mp_v->inset_cloud(cv::format("cu22%d", frame_idx), cv::viz::WCloud(mm.curr_point));//, mm.curr_color

            //     // mm.resetnode();
            // }
            //     // std::cout<<""<<atime[1]<<","<<atime[0]<<std::endl;
            // }

            // cv::Mat shot =mp_v->getScreenshot();
            // cv::imshow("shot",shot);
            // cv::imwrite(cv::format("shot%d.png",frame_idx),shot);
            //显示轨迹 debug信息
            // if (true)
            // {
            //
            //     //  while(1);
            // }

            // ray.mgraycast_test(intr.cam,dev_boxptr, mp_v->getpose());
            // mm.save_tsdf_mode_grids("ab");
            // assert(0);
            // Mat po_int, col_or;
            // mm.mcps.margCpuVoxel32Tocloud(po_int, col_or);
            // if (po_int.rows > 0)
            //     mp_v->inset_cloud("curr22", cv::viz::WCloud(po_int)); // col_or
            // cv::waitKey(1);
            // if (mm.gpu_pbox_use.size() > 500)
            // {

            //     std::fstream file("temp", std::ios::out | std::ios::binary); // | ios::app
            //     struct Voxel32 pboxs;
            //     for (int i = 0; i < 500; i++)
            //     {
            //         ck(cudaMemcpy((void *)&pboxs, (void *)(mm.gpu_pbox_use[i]), sizeof(struct Voxel32), cudaMemcpyDeviceToHost));
            //         file.write(reinterpret_cast<char *>(&pboxs), sizeof(struct Voxel32));
            //     }
            //     file.close();
            //     assert(0);
            // }
        }
        mm.saveallnode();
        string save_p=fs["matpose"];
        mm.SaveVoxelGrid2SurfacePointCloud(save_p+cv::format("/mgfusion"));
        fs.release();
        assert(0);
        // mp_v->pthd.join();
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
            float pt_cam_y = cam2base[ // cv::Affine3f affpose(cam_pose);
                                 0 * 4 + 1] *
                                 tmp_pt[0] +
                             cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
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
int main(int argc, char **argv)
{
    // cudaDeviceReset();
    viewer v(argc, (char **)argv);
    // // while (1)
    // // {
    // //     /* code */
    // // }

    class TSDF tsdf(&v);
    auto pthd = std::thread(&TSDF::loop, tsdf);
    v.loop();
    pthd.join();
    return 0;
}