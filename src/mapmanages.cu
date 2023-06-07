#include "mapmanages.cuh"
using namespace std;
// #include <opencv2/highgui/highgui.hpp>
mapmanages::mapmanages()
{
    // unsigned int value2 = 1;
    // cudaMemcpyToSymbol(device::pos_index, &value2, sizeof(unsigned int));
    // printf("Host: copy %d to the global variable\n", value2);
    // checkGlobalVariable<<<10, 1>>>();
    // cudaMemcpyFromSymbol(&value2, device::pos_index, sizeof(unsigned int));
    // printf("Host: the value changed by the kernel to %d \n", value2);
    // cudaDeviceReset();
    pskipList = new SkipList<uint64_t, struct Voxel32 *>(6);
    // pboxs =  std::vector<struct Voxel32 *>(CURR_BOX_NUM, NULL);
    pboxs = (struct Voxel32 **)calloc(CURR_BOX_NUM, sizeof(struct Voxel32 *));
    struct Voxel32 srcbox;
    ck(cudaMalloc((void **)&dev_boxpool, sizeof(struct Voxel32) * ALLL_NUM)); //申请GPU显存
    ck(cudaMemset(dev_boxpool, 0, sizeof(struct Voxel32) * ALLL_NUM));
    ck(cudaMallocManaged(&gpu_para, sizeof(struct exmatcloud_para)));
    ck(cudaMalloc(&cpu_kpara.dev_rgbdata, sizeof(uchar3) * 640 * 480));
    checkCUDA(cudaGetLastError());
    for (int i = 0; i < ALLL_NUM; i++)
    {
        gpu_pbox_free.push(&dev_boxpool[i]);
    }
}

void mapmanages::exmatcloud_bynum(cv::Mat &points, cv::Mat &color, u64B4 center)
{
    // static bool allo=false;
    //    Timer tm("a");
    // tm.Start();
    CUVector<struct Voxel32 *> g_use(gpu_pbox_use.size());
    g_use.upload(gpu_pbox_use.data(), gpu_pbox_use.size());
    gpu_para->dev_points_num = 0;
    gpu_para->center = center;
    struct ex_buf *gpu_buffer;
    ck(cudaMalloc((void **)&gpu_buffer, sizeof(ex_buf))); // 60 MB
    {
        dim3 grid(g_use.len, 1, 1), block(32, 32, 1); // 设置参数
        device::extract_kernel<<<grid, block>>>(gpu_buffer, g_use, gpu_para);
        ck(cudaDeviceSynchronize());
    }
    if (gpu_para->dev_points_num == 0)
    {
        ck(cudaFree(gpu_buffer));
        return;
    }
    points = cv::Mat(gpu_para->dev_points_num, 1, CV_32FC3); //, gpu_buffer->pose, gpu_buffer->color
    color = cv::Mat(gpu_para->dev_points_num, 1, CV_8UC3);
    ck(cudaMemcpy(points.ptr<float3>(), gpu_buffer->pose, sizeof(float3) * gpu_para->dev_points_num, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(color.ptr<uchar3>(), gpu_buffer->color, sizeof(uchar3) * gpu_para->dev_points_num, cudaMemcpyDeviceToHost));
    ck(cudaFree(gpu_buffer));
    g_use.release();
}

void Voxel32::tobuff_all_space(cv::Mat &points, cv::Mat &color, const u64B4 &center)
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
                    // color.push_back(cv::Vec3b(255, 0, 0));
                }
                else // if(pt_grid_x % 2 == 0 && pt_grid_y % 2 == 0 && pt_grid_z %2 == 0)
                    if (pt_grid_x == 0 && pt_grid_y == 0 && pt_grid_z != 0)
                    {
                        cv::Vec3f vec;
                        vec[0] = index.x * VOXELSIZE + pt_grid_x * VOXELSIZE_PCUBE;
                        vec[1] = index.y * VOXELSIZE + pt_grid_y * VOXELSIZE_PCUBE;
                        vec[2] = index.z * VOXELSIZE + pt_grid_z * VOXELSIZE_PCUBE;
                        points.push_back(vec);
                        // color.push_back(cv::Vec3b(vec[2] * 80, vec[2] * 80, vec[2] * 80));
                        color.push_back(cv::Vec3b(0, 0, 255));
                    }
                    else if (pt_grid_y == 0 && pt_grid_z == 0 && pt_grid_x != 0)
                    {
                        cv::Vec3f vec;
                        vec[0] = index.x * VOXELSIZE + pt_grid_x * VOXELSIZE_PCUBE;
                        vec[1] = index.y * VOXELSIZE + pt_grid_y * VOXELSIZE_PCUBE;
                        vec[2] = index.z * VOXELSIZE + pt_grid_z * VOXELSIZE_PCUBE;
                        points.push_back(vec);
                        color.push_back(cv::Vec3b(0, 0, 255));
                        // color.push_back(cv::Vec3b(vec[2] * 80, vec[2] * 80, vec[2] * 80));
                    }
                    else if (pt_grid_x == 0 && pt_grid_z == 0 && pt_grid_y != 0)
                    {
                        cv::Vec3f vec;
                        vec[0] = index.x * VOXELSIZE + pt_grid_x * VOXELSIZE_PCUBE;
                        vec[1] = index.y * VOXELSIZE + pt_grid_y * VOXELSIZE_PCUBE;
                        vec[2] = index.z * VOXELSIZE + pt_grid_z * VOXELSIZE_PCUBE;
                        points.push_back(vec);
                        color.push_back(cv::Vec3b(0, 0, 255));
                        // color.push_back(cv::Vec3b(vec[2] * 80, vec[2] * 80, vec[2] * 80));
                    }
            }
        }
    }
}
void mapmanages::exmatcloud(u64B4 center)
{
    // CPU exmatcloud true false
    if (false)
    {
        struct Voxel32 *newpasd = new Voxel32;
        std::cout << gpu_pbox_use.size() << std::endl;
        for (int i = 0; i < gpu_pbox_use.size(); i++)
        {
            cudaMemcpy((void *)newpasd, (void *)gpu_pbox_use[i], sizeof(struct Voxel32), cudaMemcpyDeviceToHost);
            checkCUDA(cudaGetLastError());
            // cudaMemcpy((void *)&host_para, (void *)(gpu_para), sizeof(exmatcloud_para), cudaMemcpyDeviceToHost);
            newpasd->tobuff_all_space(curr_point, curr_color, center);
        }
    }
    else
    {
        exmatcloud_bynum(curr_point, curr_color, center);
    }
}
void mapmanages::move2center(u64B4 &dst, u64B4 &src_center, u64B4 &now_center)
{
    struct Voxel32 **cpu_pbox = (struct Voxel32 **)calloc(CURR_BOX_NUM, sizeof(struct Voxel32 *));

    int number = gpu_pbox_use.size();
    cout << gpu_pbox_use.size() << endl;
    struct Voxel32 srcbox;
    for (int i = number - 1; i >= 0; i--)
    {
        // cudaMemcpy(&srcbox, gpu_pbox_use[i], sizeof(struct Voxel32), cudaMemcpyDeviceToHost);

        u64B4 now_local;
        u32B4 u32_src;
        ck(cudaMemcpy((void *)&u32_src, (void *)&gpu_pbox_use[i]->index, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        now_local.x = u32_src.x + dst.x;
        now_local.y = u32_src.y + dst.y;
        now_local.z = u32_src.z + dst.z;
        // if ((abs(now_local.x) > 60) || (abs(now_local.y) > 60) || (abs(now_local.z) > 60))
        // {
        //     SaveVoxelGrid2SurfacePointCloud(cv::format("pc/%d", 0));

        //     assert(0);
        // }
        u32B4 u32new = u32_src;
        u32new.x = now_local.x;
        u32new.y = now_local.y;
        u32new.z = now_local.z;
        u32new.cnt = u32_src.cnt;
        // u32_src.print();
        // u32new.print();
        cpu_pbox[u32new.u32 & 0xffffff] = gpu_pbox_use[i];
        ck(cudaMemcpy((void *)&gpu_pbox_use[i]->index, (void *)(&u32new), sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    free(pboxs), pboxs = cpu_pbox;
}
void mapmanages::movenode_6_28(u64B4 &dst, u64B4 &src_center, u64B4 &now_center)
{
    int number = gpu_pbox_use.size();
    // cout << gpu_pbox_use.size() << endl;
    struct Voxel32 srcbox;
    // std::vector<bool> guse_bak(gpu_pbox_use.size());
    // int cnnn = 0;
    for (int i = number - 1; i >= 0; i--)
    {
        if (gpu_para->mask[i] == true) //该移除的
        {
            // cnnn++;
            cudaMemcpy(&srcbox, gpu_pbox_use[i], sizeof(struct Voxel32), cudaMemcpyDeviceToHost);
            u32B4 u32_src = srcbox.index;
            srcbox.tobuff(all_point[0], all_point[1], src_center);
            cudaMemset(gpu_pbox_use[i], 0, sizeof(struct Voxel32));
            gpu_pbox_free.push(gpu_pbox_use[i]);
            gpu_pbox_use[i] = gpu_pbox_use.back();
            pboxs[u32_src.u32 & 0xffffff] = 0;
            gpu_pbox_use.pop_back();
            continue;
        } //该保留，移动中心
    }
    // cout << cnnn << " " << gpu_pbox_use.size() << endl;
    // free(pboxs), pboxs = cpu_pbox;
}
void mapmanages::movenode_62( u64B4 &dst, u64B4 &now_center)
{
    Timer t("movenode_62");
    struct Voxel32 srcbox;
    u32B4 u32new;
    // std::cout << "size=" << gpu_pbox_use.size() << " " << 0 << std::endl;
    struct Voxel32 **cpu_pbox = (struct Voxel32 **)calloc(CURR_BOX_NUM, sizeof(struct Voxel32 *));
    struct Voxel32 **dev_pbox_use;
    int number = gpu_pbox_use.size();
    cudaMalloc(&dev_pbox_use, sizeof(struct Voxel32 *) * number);
    ck(cudaMemcpy((void *)dev_pbox_use, (void *)&gpu_pbox_use[0], (number) * sizeof(struct Voxel32 *), cudaMemcpyHostToDevice));
    u32B4 *srcid, *nowid;
    cudaMallocManaged(&srcid, sizeof(u32B4) * number);
    cudaMallocManaged(&nowid, sizeof(u32B4) * number);
    bool *mask;
    cudaMallocManaged(&mask, sizeof(bool) * number);
    ck(cudaGetLastError());
    dim3 grid(number, 1, 1), block(1, 1, 1);
    device::update_loacl_index<<<grid, block>>>(dev_pbox_use, dst, now_center, srcid, nowid, mask);
    int cnt = 0, cnt2 = 0;
    ck(cudaDeviceSynchronize());
    for (int i = 0; i < number; i++)
    {
        // if (mask[i] == false) //该导出
        {
            gpu_pbox_free.push(gpu_pbox_use[i]);
            cnt++;
            cudaMemset(gpu_pbox_use[i], 0, sizeof(struct Voxel32));
            // cudaMemcpy(&srcbox, gpu_pbox_use[i], sizeof(struct Voxel32), cudaMemcpyDeviceToHost);
            gpu_pbox_use[i] = gpu_pbox_use.back();
            gpu_pbox_use.pop_back();

            continue;
        }
        cnt2++;
        if (srcid[i].u32 == 0)
            printf("%x ", srcid[i].u32);
        cpu_pbox[srcid[i].u32 & 0xffffff] = pboxs[nowid[i].u32 & 0xffffff];
    }
    free(pboxs);
    cudaFree(dev_pbox_use);
    cudaFree(srcid);
    cudaFree(nowid);
    cudaFree(mask);
    pboxs = cpu_pbox;
    std::cout << "sum:" << number << " export:" << cnt << " change:" << cnt2 << std::endl;
    // std::vector<struct Voxel32 *>().swap(gpu_pbox_use);
    // std::cout << "size=" << gpu_pbox_use.size() << " " << 0 << std::endl;
}
void mapmanages::skiplistbox(cv::Mat &_points, cv::Mat &color, u64B4 &center)
{
    std::vector<struct Voxel32 *> pboxs;
    std::vector<uint64_t> pkey;
    pskipList->display_list(pkey, pboxs);
    struct Voxel32 *gpu_pbox;
    std::size_t num = pboxs.size();
    ck(cudaMalloc((void **)&gpu_pbox, sizeof(struct Voxel32) * num));
    checkCUDA(cudaGetLastError());

    for (int i = 0; i < num; i++)
    {
        // pboxs[i]->index.cnt = 0;
        cudaMemcpy((void *)(&(gpu_pbox[i])), (void *)(pboxs[i]), sizeof(struct Voxel32), cudaMemcpyHostToDevice);
        checkCUDA(cudaGetLastError());
        // delete cpu_box[i];
    }
    u64B4 u64;
    exmatcloud_bynum(_points, color, u64);
    cudaFree(gpu_pbox);
    // std::vector<struct Voxel32 *>().swap(cpu_box);
    // checkCUDA(cudaGetLastError());

    // for (int i = 0; i < pkey; i++)
    // {
    //     // cpu_box[i]->index.cnt = 0;
    //     // cudaMemcpy((void *)(&(gpu_pbox[i])), (void *)(cpu_box[i]), sizeof(struct Voxel32), cudaMemcpyHostToDevice);
    //     // checkCUDA(cudaGetLastError());
    //     // delete cpu_box[i];
    // }
    // // u64B4 u64;
    // // exmatcloud_bynum(_points, color, u64, gpu_pbox, num);
    // cudaFree(gpu_pbox);
    // std::vector<struct Voxel32 *>().swap(cpu_box);
}
void mapmanages::resetnode()
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
void mapmanages::saveallnode()
{
    struct Voxel32 srcbox;
    for (int i = 0; i < gpu_pbox_use.size(); i++)
    {
        cudaMemcpy(&srcbox, gpu_pbox_use[i], sizeof(struct Voxel32), cudaMemcpyDeviceToHost);
        u32B4 u32_src = srcbox.index;
        srcbox.tobuff(all_point[0], all_point[1], cpu_kpara.center);
    }
    std::vector<struct Voxel32 *>().swap(gpu_pbox_use);
    memset(pboxs, 0, sizeof(struct Voxel32 *) * CURR_BOX_NUM);
    
}
void mapmanages::SaveVoxelGrid2SurfacePointCloud(const std::string &file_name)
{
    FILE *fp = fopen(cv::format("%s.ply", file_name.c_str()).c_str(), "w");
    fprintf(fp, "ply\n");
    fprintf(fp, "format binary_little_endian 1.0\n");
    fprintf(fp, "element vertex %ld\n", all_point[0].rows);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "end_header\n");
    cout << file_name << ":" << all_point[0].rows << endl;
    for (int i = 0; i < all_point[0].rows; i++)
    {
        fwrite(all_point[0].ptr<cv::Vec3f>(i, 0), sizeof(float), 3, fp);
        fwrite(all_point[1].ptr<cv::Vec3b>(i, 0), sizeof(uchar), 3, fp);

        //     }
    }
    fclose(fp);
}
// FILE *fp = fopen(cv::format("%s.ply", file_name.c_str()).c_str(), "w");
// fprintf(fp, "ply\n");
// fprintf(fp, "format binary_little_endian 1.0\n");
// fprintf(fp, "element vertex %ld\n", upints->size());
// fprintf(fp, "property float x\n");
// fprintf(fp, "property float y\n");
// fprintf(fp, "property float z\n");
// fprintf(fp, "property uchar red\n");
// fprintf(fp, "property uchar green\n");
// fprintf(fp, "property uchar blue\n");
// fprintf(fp, "end_header\n");
// // Create point cloud content for ply file
// // for (int i = 0; i < upints.size(); i++)
// // {
// fwrite(upints->data(), sizeof(UPoints), upints->size(), fp);

// //     }
// // }
// fclose(fp);
// }
// mapmanages::~mapmanages()
// {
// }
