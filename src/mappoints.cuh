#pragma once
#include "cuda/datatype.cuh"
#include "cuda/skiplist.h"
#include <vector>
#include "cuda/imgproc.cuh"

// __device__ __host__ inline float3 operator*(float3 v1, float a)
// {
//     return make_float3(v1.x * a, v1.y * a, v1.z * a);
// }

class mappoints
{
public:
    SkipList<uint32_t, struct CloudBox *> *pskipList;
    std::vector<struct CloudBox *> cloudBoxs;
    std::vector<struct CpuVoxel32 *> mp_cpuVoxel32;
    mappoints()
    {
        pskipList = new SkipList<uint32_t, struct CloudBox *>(6);
    }
    // TODO 高效搜索
    bool getCube32(uint64_t idx, struct Voxel32 *&cpu_pbox)
    {
        for (int i = 0; i < mp_cpuVoxel32.size(); i++)
        {
            if (mp_cpuVoxel32[i]->wordPos.u64 == idx)
            {
                cpu_pbox = mp_cpuVoxel32[i]->pvoxel32; //->copyTobox32(cpu_pbox);
                delete mp_cpuVoxel32[i];
                mp_cpuVoxel32[i] = mp_cpuVoxel32.back();
                mp_cpuVoxel32.pop_back();
                // assert(0);
                return true;
            }
        }
        return false;
    }
    void addpoint(struct CpuVoxel32 &cpu_pbox, const u64B4 &center) //转换成点云
    {
        mp_cpuVoxel32.push_back(&cpu_pbox);
    }

    // box32转为 pointbox32
    // void addpoint(struct CpuVoxel32 &cpu_pbox, const u64B4 &center) //转换成点云
    // {
    //     cv::Mat _points, color;
    //     u64B4 u64wd;

    //     u64wd.x = cpu_pbox.pvoxel32->index.x + center.x;
    //     u64wd.y = cpu_pbox.pvoxel32->index.y + center.y;
    //     u64wd.z = cpu_pbox.pvoxel32->index.z + center.z;

    //     cpu_pbox.pvoxel32->tobuff(_points, color, center);
    //     CloudBox *pCloudBox = new CloudBox(cpu_pbox.pvoxel32->index, u64wd);
    //     for (std::size_t i = 0; i < _points.rows; i++) //
    //     {
    //         UPoints up;
    //         const cv::Vec3f &ptf = _points.at<cv::Vec3f>(i, 0);
    //         const cv::Vec3b &cob = color.at<cv::Vec3b>(i, 0);
    //         memcpy((void *)up.xyz, ptf.val, sizeof(float) * 3);
    //         memcpy((void *)up.rgb, cob.val, sizeof(uint8_t) * 3);
    //         pCloudBox->points.push_back(up);
    //     }
    //     cloudBoxs.push_back(pCloudBox);
    // }

    void addpoint_gpu(struct Voxel32 &cpu_pbox, u64B4 &center)
    {
        // cv::Mat _points, color;
        // u64B4 u64wd;
        // u64wd.x=cpu_pbox.index.x+center.x;
        // u64wd.y=cpu_pbox.index.y+center.y;
        // u64wd.z=cpu_pbox.index.z+center.z;

        struct Voxel32 *gpu_pbox;
        ck(cudaMalloc((void **)&gpu_pbox, sizeof(struct Voxel32)));
        ck(cudaMemcpy(gpu_pbox, (void *)(&cpu_pbox), sizeof(struct Voxel32), cudaMemcpyHostToDevice));

        struct ex_buf_ *point_buf;
        ck(cudaMallocManaged((void **)&point_buf, sizeof(ex_buf_)));
        point_buf->center = center;
        dim3 grid(32, 1, 1), block(32, 32, 1); // 设置参数
        device::mappoints_cube2tsdf<<<grid, block>>>(gpu_pbox, point_buf);

        cudaDeviceSynchronize();
        ck(cudaGetLastError());

        // std::cout << point_buf->dev_points_num << std::endl;
        // cpu_pbox.tobuff(_points, color, center);
        CloudBox *points = new CloudBox(cpu_pbox.index, center);
        points->index.cnt = 0;
        // points->index.type = 0;

        points->points.resize(point_buf->dev_points_num);
        memcpy(&points->points[0], (void *)&point_buf->up, sizeof(UPoints) * point_buf->dev_points_num);
        // points->points.insert(points->points.begin(), point_buf->up, point_buf->up + point_buf->dev_points_num);

        // std::copy(start, end, std::back_inserter(container));

        // for (std::size_t i = 0; i < point_buf->dev_points_num; i++) //
        // {
        //     UPoints up;
        //     memcpy((void *)&up, &point_buf->up[i], sizeof(UPoints) * 3);
        //     points->points.push_back(up);
        // }
        cloudBoxs.push_back(points);
    }

    void addpoint_gpu_batch(struct Voxel32 *gpu_pbox, u64B4 &center, int num)
    {
        // cv::Mat _points, color;
        // u64B4 u64wd;
        // u64wd.x=cpu_pbox.index.x+center.x;
        // u64wd.y=cpu_pbox.index.y+center.y;
        // u64wd.z=cpu_pbox.index.z+center.z;

        // struct Voxel32 *gpu_pbox;
        // ck(cudaMalloc((void **)&gpu_pbox, sizeof(struct Voxel32) * num));

        // ck(cudaMemcpy(gpu_pbox, (void *)(cpu_pbox), sizeof(struct Voxel32) * num, cudaMemcpyHostToDevice));

        struct ex_buf_ *point_buf;
        ck(cudaMallocManaged((void **)&point_buf, sizeof(ex_buf_) * num));
        for (int i = 0; i < num; i++)
            point_buf[i].center = center;
        dim3 grid(num, 1, 1), block(32, 32, 1); // 设置参数
        device::mappoints_cube2tsdf_batch<<<grid, block>>>(gpu_pbox, point_buf);

        cudaDeviceSynchronize();
        ck(cudaGetLastError());

        // // std::cout << point_buf->dev_points_num << std::endl;
        // // cpu_pbox.tobuff(_points, color, center);
        //
        // points->index = cpu_pbox->index;
        // points->index.cnt = 0;
        // points->index.type = 0;
        for (int i = 0; i < num; i++)
        {
            CloudBox *points = new CloudBox(center);
            points->points.resize(point_buf[i].dev_points_num);
            memcpy(&points->points[0], (void *)&point_buf[i].up, sizeof(UPoints) * point_buf[i].dev_points_num);
            cloudBoxs.push_back(points);
        }
        ck(cudaFree(point_buf));
        ck(cudaFree(gpu_pbox));

        // // points->points.insert(points->points.begin(), point_buf->up, point_buf->up + point_buf->dev_points_num);

        // // std::copy(start, end, std::back_inserter(container));

        // // for (std::size_t i = 0; i < point_buf->dev_points_num; i++) //
        // // {
        // //     UPoints up;
        // //     memcpy((void *)&up, &point_buf->up[i], sizeof(UPoints) * 3);
        // //     points->points.push_back(up);
        // // }
    }
    //多个BOX合并为一个点云
    void marg(cv::Mat &_points, cv::Mat &color)
    {
        // std::cout << "cloudBoxs:" << cloudBoxs.size() << std::endl;
        for (auto &it : cloudBoxs)
        {
            cv::Vec3f ptf;
            cv::Vec3b cob;
            // std::cout<<"cloudBoxs:"<<cloudBoxs.size()<<std::endl;
            for (int i = 0; i < it->points.size(); i++)
            {
                UPoints &up = it->points[i];
                memcpy(ptf.val, (void *)up.xyz, sizeof(float) * 3);
                memcpy(cob.val, (void *)up.rgb, sizeof(uint8_t) * 3);
                _points.push_back(ptf);
                color.push_back(cob);
            }
        }
    }
    void margCpuVoxel32Tocloud(cv::Mat &_points, cv::Mat &color)
    {
        // std::cout << "cloudBoxs:" << cloudBoxs.size() << std::endl;
        for (auto &it : mp_cpuVoxel32)
        {
            cv::Vec3f ptf;
            cv::Vec3b cob;
            it->pvoxel32->tobuff(_points, color, it->wordPos);
        }
    }
    void test(cv::Mat &pt, cv::Mat &color, cv::Mat &expoints, cv::Mat &excolor)
    {
        load("maptest.cube.cloud");

        std::map<uint32_t, struct Voxel32 *> boxmap;
        std::cout << "cloudBoxs:" << cloudBoxs.size() << std::endl;
        for (auto &it : cloudBoxs)
        {
            cv::Vec3f ptf;
            ;
            cv::Vec3b cob;
            // std::cout<<"cloudBoxs:"<<cloudBoxs.size()<<std::endl;
            for (int i = 0; i < it->points.size(); i++)
            {
                UPoints &up = it->points[i];
                memcpy(ptf.val, (void *)up.xyz, sizeof(float) * 3);
                memcpy(cob.val, (void *)up.rgb, sizeof(uint8_t) * 3);

                u32_4byte u64;
                cv::Vec3f retf = 3.125f * ptf;
                // cv::Vec3s rets;
                u64.x = std::floor(retf[0]);
                u64.y = std::floor(retf[1]);
                u64.z = std::floor(retf[2]);

                std::map<uint32_t, struct Voxel32 *>::iterator iter = boxmap.find(u64.u32);
                if (iter != boxmap.end())
                {
                }
                else
                {
                    boxmap[u64.u32] = new struct Voxel32();
                }
                u32_4byte u32;
                u32.x = (ptf[0] - u64.x * 0.32f) * 100;
                u32.y = (ptf[1] - u64.y * 0.32f) * 100;
                u32.z = (ptf[2] - u64.z * 0.32f) * 100;
                assert(u32.x < 32);
                assert(u32.y < 32);
                assert(u32.z < 32);
                int index = u32.x + u32.y * 32 + u32.z * 32 * 32;
                boxmap[u64.u32]->index = u64;
                boxmap[u64.u32]->pVoxel[index].tsdf = 0.0f;
                boxmap[u64.u32]->pVoxel[index].weight = 50.0f;
                boxmap[u64.u32]->pVoxel[index].rgb[0] = cob[0];
                boxmap[u64.u32]->pVoxel[index].rgb[1] = cob[1];
                boxmap[u64.u32]->pVoxel[index].rgb[2] = cob[2];
            }
        }
        u64B4 u640;

        std::cout << boxmap.size() << std::endl;

        for (auto &kv : boxmap)
        {
            struct Voxel32 *p = kv.second;
            p->tobuff(expoints, excolor, u640);
        }
    }
    //保存box
    void save(std::string fname = "map.cube.cloud")
    {
        std::fstream file(fname, std::ios::out | std::ios::binary); // | ios::app
        size_t num = cloudBoxs.size();
        file.write(reinterpret_cast<char *>(&num), sizeof(size_t));
        std::cout << "cloudBoxs:" << cloudBoxs.size() << std::endl;
        for (auto &it : cloudBoxs)
        {
            cv::Vec3f ptf;
            ;
            cv::Vec3b cob;
            // std::cout<<"cloudBoxs:"<<cloudBoxs.size()<<std::endl;
            size_t point_num = it->points.size();
            uint32_t index_ = it->index.u32;
            file.write(reinterpret_cast<char *>(&index_), sizeof(uint32_t));
            // UPoints *p=
            file.write(reinterpret_cast<char *>(&point_num), sizeof(size_t));
            file.write(reinterpret_cast<char *>(&it->points[0]), sizeof(UPoints) * point_num);

            // for(int i=0; i<it->points.size(); i++)
            // {
            //     UPoints &up=it->points[i];
            //     memcpy(ptf.val,(void*)up.xyz,sizeof(float)*3);
            //     memcpy(cob.val,(void*)up.rgb,sizeof(uint8_t)*3);
            //     _points.push_back(ptf);
            //     color.push_back(cob);
            // }
        }
        file.close();

        // std::vector<struct CloudBox *> cloudBoxs;
        // std::vector<uint32_t> pkey;
        // pskipList->display_list(pkey,cloudBoxs);

        // pskipList
    };
    void load(std::string fname = "map.cube.cloud")
    {
        std::fstream file(fname, std::ios::in | std::ios::binary); // | ios::app
        size_t num;
        file.read((char *)&num, sizeof(size_t));
        for (int i = 0; i < num; i++)
        {
            uint32_t index_;

            file.read(reinterpret_cast<char *>(&index_), sizeof(uint32_t));
            struct CloudBox *pboxp = new struct CloudBox();
            size_t point_num; //=it->points.size();
            file.read(reinterpret_cast<char *>(&point_num), sizeof(size_t));
            pboxp->points.resize(point_num);
            file.read(reinterpret_cast<char *>(&pboxp->points[0]), sizeof(UPoints) * point_num);

            // for(int j=0; j<point_num; j++)
            // {

            //     pboxp->push_back();
            // }
            cloudBoxs.push_back(pboxp);
        }
        file.close();
    };

    // private:

    void hdtest_gpu(int index_, struct Voxel32 *&pboxmap)
    {
        auto it = cloudBoxs[index_];
        size_t psize = it->points.size();
        if (psize == 0)
        {
            return;
        }
        // std::cout << it->points.size() << std::endl;
        // load("maptest.cube.cloud");
        // hdtest(num, pboxmap);
        ck(cudaMalloc((void **)&pboxmap, sizeof(Voxel32)));
        dim3 grid0(CUBEVOXELSIZE, 1, 1), block0(CUBEVOXELSIZE, CUBEVOXELSIZE, 1); // 设置参数
        device::cloud2grids_init<<<grid0, block0>>>(pboxmap);
        // cudaDeviceSynchronize();
        ck(cudaGetLastError());
        UPoints *pup;

        ck(cudaMalloc((void **)&pup, sizeof(UPoints) * psize));
        ck(cudaGetLastError());

        cudaMemcpy((void *)(pup), (void *)(&it->points[0]), sizeof(UPoints) * psize, cudaMemcpyHostToDevice);
        ck(cudaGetLastError());
        dim3 grid(psize, 1, 1), block(1, 1, 1); // 设置参数
        device::cloud2grids<<<grid, block>>>(pboxmap, pup, psize);
        //  cudaDeviceSynchronize();
        // cudaFree(pup);
        // cudaFree(pboxmap);
        //
        ck(cudaGetLastError());
        // pboxmap = new struct Voxel32();

        // std::cout<<"cloudBoxs:"<<cloudBoxs.size()<<std::endl;
        //
        // {
        //     cv::Vec3f ptf;

        //     cv::Vec3b cob;
        //     // std::cout<<"cloudBoxs:"<<cloudBoxs.size()<<std::endl;
        //     for (int i = 0; i < it->points.size(); i++)
        //     {
        //         UPoints &up = it->points[i];
        //         memcpy(ptf.val, (void *)up.xyz, sizeof(float) * 3);
        //         memcpy(cob.val, (void *)up.rgb, sizeof(uint8_t) * 3);

        //         u32_4byte u64;
        //         cv::Vec3f retf = 3.125f * ptf;
        //         // cv::Vec3s rets;
        //         u64.x = std::floor(retf[0]);
        //         u64.y = std::floor(retf[1]);
        //         u64.z = std::floor(retf[2]);

        //         u32_4byte u32;
        //         u32.x = (ptf[0] - u64.x * 0.32f) * 100;
        //         u32.y = (ptf[1] - u64.y * 0.32f) * 100;
        //         u32.z = (ptf[2] - u64.z * 0.32f) * 100;
        //         assert(u32.x < 32);
        //         assert(u32.y < 32);
        //         assert(u32.z < 32);
        //         int index = u32.x + u32.y * 32 + u32.z * 32 * 32;
        //         pboxmap->index = u64;
        //         pboxmap->pVoxel[index].tsdf = 0.0f;
        //         pboxmap->pVoxel[index].weight = 50.0f;
        //         pboxmap->pVoxel[index].rgb[0] = cob[0];
        //         pboxmap->pVoxel[index].rgb[1] = cob[1];
        //         pboxmap->pVoxel[index].rgb[2] = cob[2];
        //     }
        // }
    }
};