#pragma once
#include "cuda/datatype.cuh"
#include "cuda/skiplist.h"
#include <vector>
#include "cuda/imgproc.cuh"

class mappoints
{
public:
    SkipList<uint32_t, struct box32_points *> *pskipList;
    std::vector<struct box32_points *> pboxs;
    mappoints()
    {
        pskipList = new SkipList<uint32_t, struct box32_points *>(6);
    }
    // TODO 高效搜索
    bool getCube32(uint64_t idx, struct box32 *&cpu_pbox)
    {
        u32_4byte u32;
        u64_4byte u64;
        u64.u64 = idx;
        u32.x = u64.x;
        u32.y = u64.y;
        u32.z = u64.z;

        for (int i = 0; i < pboxs.size(); i++)
        {
            if (pboxs[i]->index.u32 == u32.u32)
            {
                hdtest(i, cpu_pbox);
                return true;
            }
        }
        return false;
    }
    // box32转为 pointbox32
    void addpoint(struct box32 &cpu_pbox, u64_4byte &center)
    {
        cv::Mat _points, color;
        // u64_4byte u64wd;
        // u64wd.x=cpu_pbox.index.x+center.x;
        // u64wd.y=cpu_pbox.index.y+center.y;
        // u64wd.z=cpu_pbox.index.z+center.z;

        cpu_pbox.tobuff(_points, color, center);
        box32_points *points = new box32_points(center);
        points->index = cpu_pbox.index;
        points->index.cnt = 0;
        // points->index.type = 0;
        for (std::size_t i = 0; i < _points.rows; i++) //
        {
            UPoints up;
            const cv::Vec3f &ptf = _points.at<cv::Vec3f>(i, 0);
            const cv::Vec3b &cob = color.at<cv::Vec3b>(i, 0);
            memcpy((void *)up.xyz, ptf.val, sizeof(float) * 3);
            memcpy((void *)up.rgb, cob.val, sizeof(uint8_t) * 3);
            points->points.push_back(up);
        }
        pboxs.push_back(points);
    }

    void addpoint_gpu(struct box32 &cpu_pbox, u64_4byte &center)
    {
        // cv::Mat _points, color;
        // u64_4byte u64wd;
        // u64wd.x=cpu_pbox.index.x+center.x;
        // u64wd.y=cpu_pbox.index.y+center.y;
        // u64wd.z=cpu_pbox.index.z+center.z;

        struct box32 *gpu_pbox;
        ck(cudaMalloc((void **)&gpu_pbox, sizeof(struct box32)));
        ck(cudaMemcpy(gpu_pbox, (void *)(&cpu_pbox), sizeof(struct box32), cudaMemcpyHostToDevice));

        struct ex_buf_ *point_buf;
        ck(cudaMallocManaged((void **)&point_buf, sizeof(ex_buf_)));
        point_buf->center = center;
        dim3 grid(32, 1, 1), block(32, 32, 1); // 设置参数
        device::mappoints_cube2tsdf<<<grid, block>>>(gpu_pbox, point_buf);

        cudaDeviceSynchronize();
        ck(cudaGetLastError());

        // std::cout << point_buf->dev_points_num << std::endl;
        // cpu_pbox.tobuff(_points, color, center);
        box32_points *points = new box32_points(center);
        points->index = cpu_pbox.index;
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
        pboxs.push_back(points);
    }

    void addpoint_gpu_batch(struct box32 *gpu_pbox, u64_4byte &center, int num)
    {
        // cv::Mat _points, color;
        // u64_4byte u64wd;
        // u64wd.x=cpu_pbox.index.x+center.x;
        // u64wd.y=cpu_pbox.index.y+center.y;
        // u64wd.z=cpu_pbox.index.z+center.z;

        // struct box32 *gpu_pbox;
        // ck(cudaMalloc((void **)&gpu_pbox, sizeof(struct box32) * num));

        // ck(cudaMemcpy(gpu_pbox, (void *)(cpu_pbox), sizeof(struct box32) * num, cudaMemcpyHostToDevice));

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
            box32_points *points = new box32_points(center);
            points->points.resize(point_buf[i].dev_points_num);
            memcpy(&points->points[0], (void *)&point_buf[i].up, sizeof(UPoints) * point_buf[i].dev_points_num);
            pboxs.push_back(points);
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

    struct box32 *expoint(struct box32_points *&cpu_pbox)
    {
        struct box32 *pcpu_box32 = new struct box32;
        for (std::size_t i = 0; i < cpu_pbox->points.size(); i++) //
        {
            u32_4byte u64;
            const UPoints &ptf = cpu_pbox->points[i];
            u64.x = (ptf.xyz[0] * 3.125f); // std::floor
            u64.y = (ptf.xyz[1] * 3.125f);
            u64.z = (ptf.xyz[2] * 3.125f);

            u32_4byte u32;
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
        // std::cout<<__LINE__<<std::endl;
        // u64_4byte u640;
        // std::cout<<boxmap.size()<<std::endl;

        // for (auto &kv : boxmap) {
        //     struct box32 *p=kv.second;
        //     // u64_4byte u64;
        //     // u64.u64=kv.first;
        //     // p->index.u32.x=u64.u32.x;
        //     // p->index.u32.y=u64.u32.y;
        //     // p->index.u32.z=u64.u32.z;

        //     // std::cout<<u64.u64<<std::endl;

        //     p->tobuff(expoints,excolor,u640);
        // }

        // }
        return pcpu_box32;
    }
    //多个BOX合并为一个点云
    void marg(cv::Mat &_points, cv::Mat &color)
    {
        // std::vector<struct box32_points *> pboxs;
        // std::vector<uint32_t> pkey;
        // pskipList->display_list(pkey,pboxs);

        std::cout << "pboxs:" << pboxs.size() << std::endl;
        for (auto &it : pboxs)
        {
            cv::Vec3f ptf;
            ;
            cv::Vec3b cob;
            // std::cout<<"pboxs:"<<pboxs.size()<<std::endl;

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

    void test(cv::Mat &pt, cv::Mat &color, cv::Mat &expoints, cv::Mat &excolor)
    {
        load("maptest.cube.cloud");

        std::map<uint32_t, struct box32 *> boxmap;
        std::cout << "pboxs:" << pboxs.size() << std::endl;
        for (auto &it : pboxs)
        {
            cv::Vec3f ptf;
            ;
            cv::Vec3b cob;
            // std::cout<<"pboxs:"<<pboxs.size()<<std::endl;
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

                std::map<uint32_t, struct box32 *>::iterator iter = boxmap.find(u64.u32);
                if (iter != boxmap.end())
                {
                }
                else
                {
                    boxmap[u64.u32] = new struct box32();
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
        u64_4byte u640;

        std::cout << boxmap.size() << std::endl;

        for (auto &kv : boxmap)
        {
            struct box32 *p = kv.second;
            p->tobuff(expoints, excolor, u640);
        }
    }
    //保存box
    void save(std::string fname = "map.cube.cloud")
    {
        std::fstream file(fname, std::ios::out | std::ios::binary); // | ios::app
        size_t num = pboxs.size();
        file.write(reinterpret_cast<char *>(&num), sizeof(size_t));
        std::cout << "pboxs:" << pboxs.size() << std::endl;
        for (auto &it : pboxs)
        {
            cv::Vec3f ptf;
            ;
            cv::Vec3b cob;
            // std::cout<<"pboxs:"<<pboxs.size()<<std::endl;
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

        // std::vector<struct box32_points *> pboxs;
        // std::vector<uint32_t> pkey;
        // pskipList->display_list(pkey,pboxs);

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
            struct box32_points *pboxp = new struct box32_points();
            size_t point_num; //=it->points.size();
            file.read(reinterpret_cast<char *>(&point_num), sizeof(size_t));
            pboxp->points.resize(point_num);
            file.read(reinterpret_cast<char *>(&pboxp->points[0]), sizeof(UPoints) * point_num);

            // for(int j=0; j<point_num; j++)
            // {

            //     pboxp->push_back();
            // }
            pboxs.push_back(pboxp);
        }
        file.close();
    };

    // private:
    void hdtest(int index_, struct box32 *&pboxmap)
    {
        // load("maptest.cube.cloud");
        pboxmap = new struct box32();
        pboxmap->init();
        // std::cout<<"pboxs:"<<pboxs.size()<<std::endl;
        auto it = pboxs[index_];
        {
            cv::Vec3f ptf;

            cv::Vec3b cob;
            // std::cout<<"pboxs:"<<pboxs.size()<<std::endl;
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

                u32_4byte u32;
                u32.x = (ptf[0] - u64.x * 0.32f) * 100;
                u32.y = (ptf[1] - u64.y * 0.32f) * 100;
                u32.z = (ptf[2] - u64.z * 0.32f) * 100;
                assert(u32.x < 32);
                assert(u32.y < 32);
                assert(u32.z < 32);
                int index = u32.x + u32.y * 32 + u32.z * 32 * 32;
                pboxmap->index = u64;
                pboxmap->pVoxel[index].tsdf = 0.0f;
                pboxmap->pVoxel[index].weight = 50.0f;
                pboxmap->pVoxel[index].rgb[0] = cob[0];
                pboxmap->pVoxel[index].rgb[1] = cob[1];
                pboxmap->pVoxel[index].rgb[2] = cob[2];
            }
        }
    }
    void hdtest_gpu(int index_, struct box32 *&pboxmap)
    {
        auto it = pboxs[index_];
        size_t psize = it->points.size();
        if (psize == 0)
        {
            return;
        }
        // std::cout << it->points.size() << std::endl;
        // load("maptest.cube.cloud");
        // hdtest(num, pboxmap);
        ck(cudaMalloc((void **)&pboxmap, sizeof(box32)));
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
        device::cloud2grids<<<grid, block>>>(pboxmap, pup,psize);
        //  cudaDeviceSynchronize();
        // cudaFree(pup);
        // cudaFree(pboxmap);
        //
        ck(cudaGetLastError());
        // pboxmap = new struct box32();

        // std::cout<<"pboxs:"<<pboxs.size()<<std::endl;
        //
        // {
        //     cv::Vec3f ptf;

        //     cv::Vec3b cob;
        //     // std::cout<<"pboxs:"<<pboxs.size()<<std::endl;
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