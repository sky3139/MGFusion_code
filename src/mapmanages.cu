#include "mapmanages.cuh"
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
void mapmanages::movenode_62(struct Voxel32 **&dev_boxptr, u64B4 &dst, u64B4 &now_center)
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

// mapmanages::~mapmanages()
// {
// }
