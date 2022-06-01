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
    pskipList = new SkipList<uint64_t, struct box32 *>(6);
    // pboxs =  std::vector<struct box32 *>(CURR_BOX_NUM, NULL);
    pboxs = (struct box32 **)calloc(CURR_BOX_NUM, sizeof(struct box32 *));
    // cpu_kpara.rgbdata = new uint8_t[640 * 480 * 3];
    struct box32 srcbox;
    srcbox.init();
    cudaMalloc((void **)&dev_boxpool, sizeof(struct box32) * ALLL_NUM); //申请GPU显存
    checkCUDA(cudaGetLastError());
    for (int i = 0; i < ALLL_NUM; i++)
    {
        cudaMemcpy((void *)(&(dev_boxpool[i])), (void *)(&srcbox), sizeof(struct box32), cudaMemcpyHostToDevice);
        checkCUDA(cudaGetLastError());
        gpu_pbox_free.push(&dev_boxpool[i]);
    }

    checkCUDA(cudaGetLastError());
}

void mapmanages::exmatcloud_bynum(cv::Mat &points, cv::Mat &color, u64_4byte center, struct box32 *gpu_boxpool, int number)
{
    // static bool allo=false;
    // Timer tm;
    // tm.Start();
    struct exmatcloud_para *gpu_para;
    // struct exmatcloud_para *host_para;

    cudaMallocManaged(&gpu_para, sizeof(struct exmatcloud_para));
    gpu_para->center = center;

    // std::cout<<sizeof(exmatcloud_para)<<std::endl;
    // Point3dim *gpu_buffer;
    struct ex_buf *gpu_buffer;
    // if(gpu_buffer==nullptr)
    ck(cudaMallocManaged((void **)&gpu_buffer, sizeof(ex_buf))); // 120 MB
    checkCUDA(cudaGetLastError());

    // tm.Start();
    dim3 grid(number, 1, 1), block(32, 32, 1); // 设置参数
    device::extract_kernel<<<grid, block>>>(gpu_buffer, gpu_boxpool, gpu_para);
    cudaDeviceSynchronize();
    ck(cudaGetLastError());
    // tm.PrintSeconds(cv::format("ex  %d  , %d",__LINE__,gpu_para->dev_points_num ));
    // tm.Start();

    // ck(cudaMemcpy((void *)host_para, (void *)(gpu_para), sizeof(exmatcloud_para), cudaMemcpyDeviceToHost)) ;
    // tm.PrintSeconds(cv::format("%d",__LINE__));

    // cudaMemcpy((void *)(&value), (void *)&device::pos_index, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (gpu_para->dev_points_num == 0)
    {
        cudaFree(gpu_buffer);
        checkCUDA(cudaGetLastError());
        cudaFree(gpu_para);
        checkCUDA(cudaGetLastError());
        return;
    }
    // tm.Start();

    points = cv::Mat(gpu_para->dev_points_num, 1, CV_32FC3, &gpu_buffer->pose[0].x).clone();
    color = cv::Mat(gpu_para->dev_points_num, 1, CV_8UC3, &gpu_buffer->color[0].rgb[0]).clone();

    {
        // std::cout<<gpu_para->dev_points_num<<std::endl;
        //     struct ex_buf *gpu_out_buffer;
        //     // if(gpu_buffer==nullptr)
        //     ck(cudaMallocManaged((void **)&gpu_out_buffer, sizeof(ex_buf))) ;  //120 MB
        // dim3 grid2(number, 1, 1), block2(32, 32, 1); // 设置参数
        //     rander_cloud_kernel<<<grid2, block2>>>(gpu_buffer, gpu_para,gpu_out_buffer);
    }

    cudaFree(gpu_buffer);
    checkCUDA(cudaGetLastError());
    cudaFree(gpu_para);
    checkCUDA(cudaGetLastError());
}

void box32::tobuff_all_space(cv::Mat &points, cv::Mat &color, const u64_4byte &center)
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
void mapmanages::exmatcloud(u64_4byte center)
{
    // CPU exmatcloud true false
    if (false)
    {
        struct box32 *newpasd = new box32;
        newpasd->init();
        std::cout << gpu_pbox_use.size() << std::endl;
        for (int i = 0; i < gpu_pbox_use.size(); i++)
        {
            cudaMemcpy((void *)newpasd, (void *)gpu_pbox_use[i], sizeof(struct box32), cudaMemcpyDeviceToHost);
            checkCUDA(cudaGetLastError());
            // cudaMemcpy((void *)&host_para, (void *)(gpu_para), sizeof(exmatcloud_para), cudaMemcpyDeviceToHost);
            newpasd->tobuff_all_space(curr_point, curr_color, center);
        }
    }
    else
    {
        exmatcloud_bynum(curr_point, curr_color, center, dev_boxpool, ALLL_NUM);
    }
}

void mapmanages::skiplistbox(cv::Mat &_points, cv::Mat &color, u64_4byte &center)
{
    std::vector<struct box32 *> pboxs;
    std::vector<uint64_t> pkey;
    pskipList->display_list(pkey, pboxs);
    struct box32 *gpu_pbox;
    std::size_t num = pboxs.size();
    ck(cudaMalloc((void **)&gpu_pbox, sizeof(struct box32) * num));
    checkCUDA(cudaGetLastError());

    for (int i = 0; i < num; i++)
    {
        pboxs[i]->index.cnt = 0;
        cudaMemcpy((void *)(&(gpu_pbox[i])), (void *)(pboxs[i]), sizeof(struct box32), cudaMemcpyHostToDevice);
        checkCUDA(cudaGetLastError());
        // delete cpu_box[i];
    }
    u64_4byte u64;
    exmatcloud_bynum(_points, color, u64, gpu_pbox, num);
    cudaFree(gpu_pbox);
    // std::vector<struct box32 *>().swap(cpu_box);
    // checkCUDA(cudaGetLastError());

    // for (int i = 0; i < pkey; i++)
    // {
    //     // cpu_box[i]->index.cnt = 0;
    //     // cudaMemcpy((void *)(&(gpu_pbox[i])), (void *)(cpu_box[i]), sizeof(struct box32), cudaMemcpyHostToDevice);
    //     // checkCUDA(cudaGetLastError());
    //     // delete cpu_box[i];
    // }
    // // u64_4byte u64;
    // // exmatcloud_bynum(_points, color, u64, gpu_pbox, num);
    // cudaFree(gpu_pbox);
    // std::vector<struct box32 *>().swap(cpu_box);
}

// mapmanages::~mapmanages()
// {
// }
