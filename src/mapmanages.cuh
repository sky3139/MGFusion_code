#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include "cuda/device_array.hpp"
#include "cuda/datatype.cuh"
#include "cuda/imgproc.cuh"
#include "cuda/skiplist.h"
#include <functional>

#include "tsdf.cuh"
#include <queue>
#include <stack>
#include <fstream>
#include <iostream>
#include "lz4.h"
#include "mappoints.cuh"
#include "tool/Timer.hpp"

#define CURR_BOX_NUM (0xffffff)
// 2048
#define ALLL_NUM 2048 * 2

class mapmanages
{

public:
    // config
    bool use_skip_list = false;

    cv::Mat curr_point, curr_color;
    // std::vector<struct Voxel32 *> pboxs;
    struct Voxel32 **pboxs; //已申请的CURR_BOX_NUM个地址
    SkipList<uint64_t, struct Voxel32 *> *pskipList;

    mappoints mcps;
    struct Voxel32 *dev_boxpool;                //已申请的ALLL_NUM个box空间的首地址
    std::stack<struct Voxel32 *> gpu_pbox_free; //记录空闲的box在GPU中的地址
    std::vector<struct Voxel32 *> gpu_pbox_use; //记录已经使用的box在GPU中的地址

    cv::Mat points, color;
    struct kernelPara cpu_kpara;
    uint8_t tfidex = 0;

    mapmanages();
    // bool find_in_cpu_pbox_use(uint64_t idx, struct Voxel32 *&cpu_pbox)
    // {
    //     for (auto &it : cpu_pbox_use)
    //     {
    //         if (idx == it->wordPos)
    //             return true;
    //     }
    //     return false;
    // }
    struct Voxel32 *getidlebox(uint32_t val)
    {
        assert(gpu_pbox_free.size() != 0);
        struct Voxel32 *cpu_pbox = nullptr;
        u32_4byte u32;
        u32.u32 = val;

        u64B4 u64; //计算绝对坐标
        u64.x = u32.x + cpu_kpara.center.x;
        u64.y = u32.y + cpu_kpara.center.y;
        u64.z = u32.z + cpu_kpara.center.z;
        // u32.type = 0x1;
        //得到一个空闲的GPU空间
        struct Voxel32 *gpu_pbox = gpu_pbox_free.top();

        // if (exist) //如果有记录,就给当前记录赋值
        // {
        //     cudaMemcpy((void *)gpu_pbox, (void *)(cpu_pbox), sizeof(struct Voxel32), cudaMemcpyHostToDevice);
        //     delete cpu_pbox;
        // }

        bool exist = mcps.getCube32(u64.u64, cpu_pbox);
        if (exist) //如果有记录,就给当前记录赋值
        {
            cudaMemcpy((void *)gpu_pbox, (void *)(cpu_pbox), sizeof(struct Voxel32), cudaMemcpyHostToDevice);
            delete cpu_pbox;
        }

        gpu_pbox_free.pop();
        gpu_pbox_use.push_back(gpu_pbox);
        return gpu_pbox;
    }
    void savenode_cube_(u64B4 &center)
    {
        std::vector<struct CpuVoxel32 *> cpu_pbox_use; //记录已经使用的box在CPU中的地址

        int start_id = cpu_pbox_use.size();
        // u64B4 u64wd;
        // u64wd.x = cpu_pbox.index.x + center.x;
        // u64wd.y = cpu_pbox.index.y + center.y;
        // u64wd.z = cpu_pbox.index.z + center.z;
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        for (int i = 0; i < gpu_pbox_use.size(); i++)
        {
            struct CpuVoxel32 *pbox = new struct CpuVoxel32();
            pbox->wordPos = center;
            ck(cudaMemcpyAsync((void *)(pbox->pvoxel32), (void *)(gpu_pbox_use[i]), sizeof(struct Voxel32), cudaMemcpyDeviceToHost, stream)); //拷贝到CPU
            cpu_pbox_use.push_back(pbox);
        }
        ck(cudaStreamSynchronize(stream));
        // // tm.PrintSeconds(" a");
        // // tm.Start();
        for (int i = 0; i < gpu_pbox_use.size(); i++) //转换成全局坐标
        {
            struct CpuVoxel32 *pbox = cpu_pbox_use[i + start_id];
            // pbox->index.x += cpu_kpara.center.x;
            // pbox->index.y += cpu_kpara.center.y;
            // pbox->index.z += cpu_kpara.center.z;
            // if (use_skip_list)
            //     pskipList->insert_element(u64.u64, pbox);
            mcps.addpoint(*pbox, cpu_kpara.center);
        }
        // // tm.PrintSeconds(" b");
        cudaStreamDestroy(stream);
        // std::vector<struct Voxel32 *>().swap(cpu_pbox_use);
    }
    void savenode_only_mini_memory()
    {
        // for (int i = 0; i < gpu_pbox_use.size(); i++)
        // {
        //     struct Voxel32 *pbox = new struct Voxel32();
        //     cudaMemcpy((void *)(pbox), (void *)(gpu_pbox_use[i]), sizeof(struct Voxel32), cudaMemcpyDeviceToHost);
        //     checkCUDA(cudaGetLastError());
        //     u64B4 u64;
        //     u64.x = pbox->index.x + cpu_kpara.center.x;
        //     u64.y = pbox->index.y + cpu_kpara.center.y;
        //     u64.z = pbox->index.z + cpu_kpara.center.z;
        //     pbox->index.x += cpu_kpara.center.x;
        //     pbox->index.y += cpu_kpara.center.y;
        //     pbox->index.z += cpu_kpara.center.z;
        //     if (use_skip_list)
        //         pskipList->insert_element(u64.u64, pbox);
        //     mcps.addpoint(*pbox, cpu_kpara.center);
        //     delete pbox;
        // }
    }
    void resetnode()
    {
        struct Voxel32 srcbox;

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        // std::cout << "size=" << gpu_pbox_use.size() << " " << 0 << std::endl;
        for (int i = 0; i < gpu_pbox_use.size(); i++)
        {
            cudaMemcpyAsync((void *)(gpu_pbox_use[i]), (void *)(&srcbox), sizeof(struct Voxel32), cudaMemcpyHostToDevice, stream);
            checkCUDA(cudaGetLastError());
            gpu_pbox_free.push(gpu_pbox_use[i]);
        }
        std::vector<struct Voxel32 *>().swap(gpu_pbox_use);
        memset(pboxs, 0, sizeof(struct Voxel32 *) * CURR_BOX_NUM);
        ck(cudaStreamSynchronize(stream));
        cudaStreamDestroy(stream);
    }
    void movenode(u64B4 &center)
    {
        struct Voxel32 srcbox;
        u32_4byte u32;
        // std::cout << "size=" << gpu_pbox_use.size() << " " << 0 << std::endl;
        struct Voxel32 **cpu_pbox = (struct Voxel32 **)calloc(CURR_BOX_NUM, sizeof(struct Voxel32 *));

        for (uint32_t i = 0; i < CURR_BOX_NUM; i++)
        {
            if (pboxs[i] == 0)
            {
                continue;
            }
            u32.u32 = i;

            u32.x -= center.x;
            u32.y -= center.y;
            u32.z -= center.z;

            cpu_pbox[u32.u32] = pboxs[i];
            cudaMemcpy((void *)(&cpu_pbox[u32.u32]->index.u32), (void *)(&u32.u32), sizeof(uint32_t), cudaMemcpyHostToDevice);

            checkCUDA(cudaGetLastError());
        }
        free(pboxs);
        pboxs = cpu_pbox;
        // std::cout << "size=" << gpu_pbox_use.size() << " " << 0 << std::endl;
        // std::vector<struct Voxel32 *>().swap(gpu_pbox_use);
        // std::cout << "size=" << gpu_pbox_use.size() << " " << 0 << std::endl;
    }
    void exidlebx(cv::Mat &_points, cv::Mat &color, u64B4 &center)
    {
        // struct Voxel32 *gpu_pbox;
        // int num = cpu_pbox_use.size();
        // ck(cudaMalloc((void **)&gpu_pbox, sizeof(struct Voxel32) * num));
        // checkCUDA(cudaGetLastError());

        // for (int i = 0; i < num; i++)
        // {
        //     cpu_pbox_use[i]->index.cnt = 0;
        //     cudaMemcpy((void *)(&(gpu_pbox[i])), (void *)(cpu_pbox_use[i]), sizeof(struct Voxel32), cudaMemcpyHostToDevice);
        //     checkCUDA(cudaGetLastError());
        //     delete cpu_pbox_use[i];
        // }
        // u64B4 u64;
        // exmatcloud_bynum(_points, color, u64, gpu_pbox, num);
        // cudaFree(gpu_pbox);
        // std::vector<struct Voxel32 *>().swap(cpu_pbox_use);
    }
    void skiplistbox(cv::Mat &_points, cv::Mat &color, u64B4 &center);
    void savefile(std::string fname = "map.bin")
    {
        std::vector<struct Voxel32 *> pboxs;
        std::vector<uint64_t> pkey;
        pskipList->display_list(pkey, pboxs);
        std::size_t num = pboxs.size();
        std::fstream file(fname, std::ios::out | std::ios::binary); // | ios::app
        file.write(reinterpret_cast<char *>(&num), sizeof(size_t));
        char *pchCompressed = new char[sizeof(struct Voxel32)];
        std::size_t all_size = 0, de_size = 0;

        for (int i = 0; i < num; i++)
        {
            int nCompressedSize = LZ4_compress_default((const char *)(pboxs[i]), pchCompressed, sizeof(struct Voxel32), sizeof(struct Voxel32));
            all_size += sizeof(struct Voxel32) + sizeof(int);
            de_size += nCompressedSize;
            // pboxs[i]->index.x =u64.x ;
            // pboxs[i]->index.y =u64.y;
            // pboxs[i]->index.z =u64.z ;
            file.write(reinterpret_cast<char *>(&nCompressedSize), sizeof(int));
            file.write(reinterpret_cast<char *>(pchCompressed), nCompressedSize);
            // file.write(reinterpret_cast<char *>(pboxs[i]), sizeof(struct Voxel32));
        }
        delete[] pchCompressed;
        printf("SAVE:item=%ld,%ld MB,%ld MB\n", num, de_size >> 20, all_size >> 20);
        file.close();
    }
    void loadfile(cv::Mat &_points, cv::Mat &color)
    {
        std::ifstream fin("map.bin", std::ios::binary);
        std::size_t num;
        fin.read((char *)&num, sizeof(size_t));
        std::cout << num << std::endl;
        struct Voxel32 cpu_pbox;
        std::size_t gpu_box_cnt = std::min(num, 4000UL);
        struct Voxel32 *gpu_pbox;
        std::cout << "gpu_pbox:" << gpu_box_cnt << std::endl;

        ck(cudaMalloc((void **)&gpu_pbox, sizeof(struct Voxel32) * gpu_box_cnt));
        checkCUDA(cudaGetLastError());

        char *pchCompressedInput = new char[sizeof(struct Voxel32)];

        u64B4 u64;
        u64.u64 = 0;
        // std::size_t all_size=0,de_size=0;
        for (int i = 0; i < num; i++)
        {
            if (i >= 4000)
            {
                exmatcloud_bynum(_points, color, u64, gpu_pbox, 4000);
                i = 0;
                num -= 4000;
                continue;
            }
            int nInputSize;
            fin.read((char *)&nInputSize, sizeof(int));
            fin.read((char *)pchCompressedInput, nInputSize);
            // std::cout<<"nInputSize:"<<nInputSize<<"  "<<sizeof(  struct Voxel32)<< std::endl;

            LZ4_decompress_safe(pchCompressedInput, (char *)&cpu_pbox, sizeof(struct Voxel32), sizeof(struct Voxel32));
            // de_size
            cpu_pbox.index.cnt = 0;

            // mps.addpoint(cpu_pbox);
            // fin.read((char *)&cpu_pbox,  sizeof( struct Voxel32));
            // std::cout<<i<<" "<<cpu_pbox.index.u32 << std::endl;
            cudaMemcpy((void *)(&(gpu_pbox[i])), (void *)(&cpu_pbox), sizeof(struct Voxel32), cudaMemcpyHostToDevice);
            checkCUDA(cudaGetLastError());
        }
        mcps.load("maptest.cube.cloud");
        mcps.marg(_points, color);
        // mps.save();
        // exmatcloud_bynum(_points, color, u64, gpu_pbox, num);
        cudaFree(gpu_pbox);
        delete[] pchCompressedInput;
        fin.close();

        // dataset::Mat_save_by_binary(_points,"points.cvmat");
        // dataset::Mat_save_by_binary(color,"color.cvmat");
    }
    void loadfilecpu(cv::Mat &_points, cv::Mat &color)
    {
        std::ifstream fin("map.bin", std::ios::binary);
        std::size_t num;
        fin.read((char *)&num, sizeof(size_t));
        struct Voxel32 cpu_pbox;
        u64B4 u64;

        for (int i = 0; i < num; i++)
        {

            fin.read((char *)&cpu_pbox, sizeof(struct Voxel32));
            cpu_pbox.tobuff(_points, color, u64);
        }
        fin.close();
    }
    void cloud2tsdf(cv::Mat &_points, cv::Mat &color, cv::Mat &expoints, cv::Mat &excolor)
    {
        // std::cout<<_points.rows<<std::endl;
        std::map<uint32_t, struct Voxel32 *> boxmap;
        for (std::size_t i = 0; i < _points.rows; i++) //
        {
            u32_4byte u64;
            const cv::Vec3f &ptf = _points.at<cv::Vec3f>(i, 0);
            const cv::Vec3b &cob = color.at<cv::Vec3b>(i, 0);

            cv::Vec3f retf = 3.125f * ptf;
            // cv::Vec3s rets;
            u64.x = std::floor(retf[0]);
            u64.y = std::floor(retf[1]);
            u64.z = std::floor(retf[2]);

            std::map<uint32_t, struct Voxel32 *>::iterator iter = boxmap.find(u64.u32);
            if (iter != boxmap.end())
            {
                //找到了
            }
            else
            {
                //没找到
                boxmap[u64.u32] = new struct Voxel32();
            }
            // boxmap.is
            //             if()
            // std::cout<<__LINE__<<" "<<i <<" "<<(ptf[1]- u32.y*0.32f)*100<<std::endl;

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

            // std::cout<<<<std::endl;
            //             std::cout<<(retf[1]- u64.y)*0.01f<<std::endl;
            //             std::cout<<(retf[2]- u64.z)*0.01f<<std::endl;
            //             std::cout<<retf<<std::endl;
        }
        std::cout << __LINE__ << std::endl;
        u64B4 u640;
        std::cout << boxmap.size() << std::endl;

        for (auto &kv : boxmap)
        {
            struct Voxel32 *p = kv.second;
            // u64B4 u64;
            // u64.u64=kv.first;
            // p->index.u32.x=u64.u32.x;
            // p->index.u32.y=u64.u32.y;
            // p->index.u32.z=u64.u32.z;

            // std::cout<<u64.u64<<std::endl;

            p->tobuff(expoints, excolor, u640);
        }
    }
    void cloud2tsdftest(cv::Mat &pt, cv::Mat &color, cv::Mat &expoints, cv::Mat &excolor)
    {

        mcps.test(pt, color, expoints, excolor);
    }
    static void exmatcloud_bynum(cv::Mat &_points, cv::Mat &color, u64B4 center, struct Voxel32 *gpu, int number);
    void exmatcloud(u64B4 center);

    void save_tsdf_mode_grids(std::string name)
    {
        std::fstream file(cv::format("%sgrids.bin", name.c_str()), std::ios::out | std::ios::binary); // | ios::app
        struct Voxel32 pboxs;
        size_t len = gpu_pbox_use.size();
        file.write(reinterpret_cast<char *>(&len), sizeof(size_t));

        for (int i = 0; i < gpu_pbox_use.size(); i++)
        {
            ck(cudaMemcpy((void *)&pboxs, (void *)(gpu_pbox_use[i]), sizeof(struct Voxel32), cudaMemcpyDeviceToHost));
            file.write(reinterpret_cast<char *>(&pboxs), sizeof(struct Voxel32));
        }
        std::cout << gpu_pbox_use.size() << std::endl;
        file.close();
    }
};
//释放
// for (size_t k = 0; k < 0xffffff; k++)
// {
//     if ((*pboxs)[k] == NULL)
//         continue;
//     u32_4byte u32;
//     u32.u32 = k;

//     u32_4byte cam32;
//     cam32.x = std::floor(parser->m_pose.val[3] / (float)(VOXELSIZE));
//     cam32.y = std::floor(parser->m_pose.val[7] / (float)(VOXELSIZE));
//     cam32.z = std::floor(parser->m_pose.val[11] / (float)(VOXELSIZE));
//     int han_dis = std::abs(cam32.x - u32.x) + std::abs(cam32.y - u32.y) + std::abs(cam32.z - u32.z);
//     if (han_dis > valmax)
//     {
//         valmax = han_dis;
//         cv::viz::WCube mcube(cv::Vec3d::all(-0.32), cv::Vec3d::all(0.32));
//         cv::Affine3f ap = cv::Affine3f::Identity();
//         cv::Vec3f pt(u32.x * 0.32f, u32.y * 0.32f, u32.z * 0.32f);
//         cv::Affine3f a2 = ap.translate(pt);

//         // std::cout << k << "  " << a2.matrix << std::endl;
//         cudaFree((*pboxs)[k]);
//         checkCUDA(cudaGetLastError());
//         window.showWidget("maxcube", mcube, a2);

//         (*pboxs)[k] = NULL;
//         valmax = 30;
//     }
//     // valmax = han_dis > valmax ? han_dis : valmax;
//     // parser->m_pose.val[7] -= 21;
//     // parser->m_pose.val[7]+=6;
// }

//   // Compute surface points from TSDF voxel grid and save points to point cloud file
// void
// SaveVoxelGrid2SurfacePointCloud(const std::string &file_name)
// {
//     int num_pts = 0;
//     for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++)
//         if (std::abs(hp_tsdf[i].tsdf) < tsdf_thresh && hp_tsdf[i].weight > weight_thresh)
//             num_pts++;

//     // Create header for .ply file
//     FILE *fp = fopen(file_name.c_str(), "w");
//     fprintf(fp, "ply\n");
//     fprintf(fp, "format binary_little_endian 1.0\n");
//     fprintf(fp, "element vertex %d\n", num_pts);
//     fprintf(fp, "property float x\n");
//     fprintf(fp, "property float y\n");
//     fprintf(fp, "property float z\n");
//     fprintf(fp, "property uchar red\n");
//     fprintf(fp, "property uchar green\n");
//     fprintf(fp, "property uchar blue\n");
//     fprintf(fp, "end_header\n");
//     // Create point cloud content for ply file
//     for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++)
//     {

//         // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
//         if (std::abs(hp_tsdf[i].tsdf) < tsdf_thresh && hp_tsdf[i].weight > weight_thresh)
//         {

//             // Compute voxel indices in int for higher positive number range
//             int z = floor(i / (voxel_grid_dim_x * voxel_grid_dim_y));
//             int y = floor((i - (z * voxel_grid_dim_x * voxel_grid_dim_y)) / voxel_grid_dim_x);
//             int x = i - (z * voxel_grid_dim_x * voxel_grid_dim_y) - (y * voxel_grid_dim_x);

//             // Convert voxel indices to float, and save coordinates to ply file
//             float pt_base_x = voxel_grid_origin_x + (float)x * voxel_size;
//             float pt_base_y = voxel_grid_origin_y + (float)y * voxel_size;
//             float pt_base_z = voxel_grid_origin_z + (float)z * voxel_size;
//             fwrite(&pt_base_x, sizeof(float), 1, fp);
//             fwrite(&pt_base_y, sizeof(float), 1, fp);
//             fwrite(&pt_base_z, sizeof(float), 1, fp);
//             fwrite(&(hp_tsdf[i].rgb[0]), sizeof(uint8_t), 1, fp);
//             fwrite(&(hp_tsdf[i].rgb[1]), sizeof(uint8_t), 1, fp);
//             fwrite(&(hp_tsdf[i].rgb[2]), sizeof(uint8_t), 1, fp);
//         }
//     }
//     fclose(fp);
// }

/* std::cout << " Create header for .ply file " << 0 << std::endl;
        FILE *fp = fopen("m.ply", "w");
        fprintf(fp, "ply\n");
        fprintf(fp, "format   ascii   1.0\n");
        // fprintf(fp, "format binary_little_endian 1.0\n");
        fprintf(fp, "element vertex %d\n", 0);
        fprintf(fp, "property float x\n");
        fprintf(fp, "property float y\n");
        fprintf(fp, "property float z\n");
        fprintf(fp, "property uchar red\n");
        fprintf(fp, "property uchar green\n");
        fprintf(fp, "property uchar blue\n");
        fprintf(fp, "end_header\n");
        // Create point cloud content for ply file
        int num2 = 0;
        for (int i = 0; i < 0xffffff; i++)
        {
            if ((mm.pboxs)[i] == NULL)
                continue;
            cudaMemcpy((void *)&srcbox, (void *)((*pboxs)[i]), sizeof(Voxel32), cudaMemcpyDeviceToHost);
            checkCUDA(cudaGetLastError());
            u32_4byte u32 = srcbox.index;
            for (int8_t pt_grid_z = 0; pt_grid_z < 32; pt_grid_z++)
            {
                for (int8_t pt_grid_y = 0; pt_grid_y < 32; pt_grid_y++)
                {
                    for (int8_t pt_grid_x = 0; pt_grid_x < 32; pt_grid_x++)
                    {
                        int volume_idx = pt_grid_z * 32 * 32 + pt_grid_y * 32 + pt_grid_x;
                        union voxel &voxel = srcbox.pVoxel[volume_idx];

                        if (std::abs(voxel.tsdf) < 0.2f && voxel.weight > 0.0f)
                        // cout << tsdfwordpose << endl;
                        // if (pt_grid_x % 5 == 0 && pt_grid_y % 5 == 0 && pt_grid_z % 5 == 0)
                        {
                            cv::Vec3f vec;
                            vec[0] = u32.x * 0.32f + pt_grid_x * voxel_size;
                            vec[1] = u32.y * 0.32f + pt_grid_y * voxel_size;
                            vec[2] = u32.z * 0.32f + pt_grid_z * voxel_size;

                            // fwrite(&vec[0], sizeof(float), 1, fp);
                            // fwrite(&vec[1], sizeof(float), 1, fp);
                            // fwrite(&vec[2], sizeof(float), 1, fp);
                            // fwrite(&(voxel.rgb[0]), sizeof(uint8_t), 1, fp);
                            // fwrite(&(voxel.rgb[1]), sizeof(uint8_t), 1, fp);
                            // fwrite(&(voxel.rgb[2]), sizeof(uint8_t), 1, fp);

                            fprintf(fp, "%f %f %f %d %d %d\n", vec[0], vec[1], vec[2], voxel.rgb[0], voxel.rgb[1], voxel.rgb[2]);

                            num2++;
                        }

                        //  if (std::abs(voxel.tsdf) < 0.2f && voxel.weight > 0.0f)
                    }
                }
            }

            // // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
            // if (std::abs(srcbox.tsdf) < tsdf_thresh && hp_tsdf[i].weight > weight_thresh)
            // {
            // }
        }
        std::cout << num2 << std::endl;
        //    std:: string str=cv::format("element vertex %d\n",num2);
        //     ModifyLineData("m.ply",,str);
        fclose(fp);
        // TSDF 拷贝到CPU

        // cudaMemcpy(hp_tsdf, dp_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(voxel), cudaMemcpyDeviceToHost);
        // checkCUDA(cudaGetLastError());
        // std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;
        // SaveVoxelGrid2SurfacePointCloud("m.ply", voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
        //                                 voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
        //                                 0.2f, 0.0f, hp_tsdf);

        */
