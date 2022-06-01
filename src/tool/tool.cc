
#include <stdio.h>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
// #include "tsdf.cuh"
#include "../cuda/datatype.cuh"

using namespace std;
// dataset::dataset(std::string path)
// {
//     Mat_read_binary(posss, path);
//     posss.convertTo(posss, CV_32F);
// }

// std::string dataset::ReadNextTUM(int frame_idx)
// {
//     depth_path = cv::format("/home/u20/dataset/img/depth/%04d.png", frame_idx);
//     cv::Mat outdepth;
//     depth_src = cv::imread(depth_path, 2);

//     if (depth_src.empty())
//     {
//         std::string ss = "ls " + depth_path;
//         system(ss.c_str());
//         assert(0);
//     }
//     int g_ndValue = 10;
//     int g_nsigmaColorValue = 10;
//     int g_nsigmaSpaceValue = 10;

//     depth_src.convertTo(depth_, CV_32FC1, DEPTHFACTOR);
//     // cv::bilateralFilter(outdepth, depth_, g_ndValue, g_nsigmaColorValue, g_nsigmaSpaceValue);

//     m_pose = posss.at<cv::Vec<float, 16>>(frame_idx, 0);
//     // pose.val[3] += 5.0f;
//     // pose.val[7] += 5.0f;
//     // pose.val[11] += 5.0f;
//     color_path = cv::format("/home/u20/dataset/img/rgb/%04d.png", frame_idx);
//     // cout << depth_path << " " << color_path << endl;

//     rgb_ = cv::imread(color_path);
//     if (rgb_.empty())
//     {
//         std::string ss = "ls " + color_path;
//         system(ss.c_str());
//         assert(0);
//     }
//     // cv::imshow("_depth", depth_src); // depth_im_file = data_path + "/frame-" + .str() + ".color.jpg";
//     cv::imshow("_rgb_img", rgb_); // depth_im_file = data_path + "/frame-" + .str() + ".color.jpg";
//     cv::waitKey(1);
//     return depth_path;
// }
// bool dataset::Mat_read_binary(cv::Mat &img_vec, std::string filename) //整体读出
// {
//     int channl(0);
//     int rows(0);
//     int cols(0);
//     short type(0);
//     short em_size(0);
//     std::ifstream fin(filename, std::ios::binary);
//     fin.read((char *)&channl, 1);
//     fin.read((char *)&type, 1);
//     fin.read((char *)&em_size, 2);
//     fin.read((char *)&cols, 4);
//     fin.read((char *)&rows, 4);
//     printf("READ:cols=%d,type=%d,em_size=%d,rows=%d,channels=%d\n", cols, type, em_size, rows, channl);
//     img_vec = cv::Mat(rows, cols, type);
//     fin.read((char *)&img_vec.data[0], rows * cols * em_size);
//     fin.close();
//     return true;
// }
// bool dataset::Mat_save_by_binary(cv::Mat &image, std::string filename) //单个写入
// {
//     int channl = image.channels();
//     int rows = image.rows;
//     int cols = image.cols;
//     short em_size = image.elemSize();
//     short type = image.type();

//     std::fstream file(filename, std::ios::out | std::ios::binary); // | ios::app
//     file.write(reinterpret_cast<char *>(&channl), 1);
//     file.write(reinterpret_cast<char *>(&type), 1);
//     file.write(reinterpret_cast<char *>(&em_size), 2);
//     file.write(reinterpret_cast<char *>(&cols), 4);
//     file.write(reinterpret_cast<char *>(&rows), 4);
//     printf("SAVE:cols=%d,type=%d,em_size=%d,rows=%d,channels=%d\n", cols, type, em_size, rows, channl);
//     file.write(reinterpret_cast<char *>(image.data), em_size * cols * rows);
//     file.close();
//     return true;
// }
void checkGpuMem()
{
    size_t avail;
    size_t total;
    cudaMemGetInfo(&avail, &total);
    size_t used = total - avail;
    u_int32_t used_M = used >> 20;
    u_int32_t total_M = total >> 20;

    std::cout << total_M << "," << used_M << std::endl;
}

void ReadRGB(std::string filename, int H, int W, uint8_t *depth)
{
    cv::Mat depth_mat = cv::imread(filename, 0);
    // std::cout<<depth_mat<<std::endl;
    if (depth_mat.empty())
    {
        std::cout << "Error: rgb image  at:" << filename << std::endl;
        cv::waitKey(0);
    }
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
        {
            depth[r * W + c] = (uint8_t)(depth_mat.at<uint8_t>(r, c));
            // if (depth[r * W + c] > 6.0f) // Only consider depth < 6m
            // depth[r * W + c] = 0;
        }
}

void ReadDepth(std::string filename, int H, int W, float *depth)
{
    cv::Mat depth_mat = cv::imread(filename, -1);
    // std::cout<<depth_mat<<std::endl;
    if (depth_mat.empty())
    {
        std::cout << "Error: depth image file not read!" << std::endl;
        cv::waitKey(0);
    }
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
        {
            depth[r * W + c] = (float)(depth_mat.at<unsigned short>(r, c)) / 5000.0f;
            if (depth[r * W + c] > 6.0f) // Only consider depth < 6m
                depth[r * W + c] = 0;
        }
}

// Load an M x N matrix from a text file (numbers delimited by spaces/tabs)
// Return the matrix as a float vector of the matrix in row-major order
cv::Mat_<float> LoadMatrixFromFile(std::string filename, int M, int N, float *MAT)
{
    cv::Mat_<float> matrix;
    FILE *fp = fopen(filename.c_str(), "r");
    for (int i = 0; i < M * N; i++)
    {
        float tmp;
        int iret = fscanf(fp, "%f", &tmp);
        matrix.push_back(tmp);
        MAT[i] = tmp;
    }
    fclose(fp);
    return matrix;
}

// 4x4 matrix multiplication (matrices are stored as float arrays in row-major order)
void multiply_matrix(const float m1[16], const float m2[16], float mOut[16])
{
    mOut[0] = m1[0] * m2[0] + m1[1] * m2[4] + m1[2] * m2[8] + m1[3] * m2[12];
    mOut[1] = m1[0] * m2[1] + m1[1] * m2[5] + m1[2] * m2[9] + m1[3] * m2[13];
    mOut[2] = m1[0] * m2[2] + m1[1] * m2[6] + m1[2] * m2[10] + m1[3] * m2[14];
    mOut[3] = m1[0] * m2[3] + m1[1] * m2[7] + m1[2] * m2[11] + m1[3] * m2[15];

    mOut[4] = m1[4] * m2[0] + m1[5] * m2[4] + m1[6] * m2[8] + m1[7] * m2[12];
    mOut[5] = m1[4] * m2[1] + m1[5] * m2[5] + m1[6] * m2[9] + m1[7] * m2[13];
    mOut[6] = m1[4] * m2[2] + m1[5] * m2[6] + m1[6] * m2[10] + m1[7] * m2[14];
    mOut[7] = m1[4] * m2[3] + m1[5] * m2[7] + m1[6] * m2[11] + m1[7] * m2[15];

    mOut[8] = m1[8] * m2[0] + m1[9] * m2[4] + m1[10] * m2[8] + m1[11] * m2[12];
    mOut[9] = m1[8] * m2[1] + m1[9] * m2[5] + m1[10] * m2[9] + m1[11] * m2[13];
    mOut[10] = m1[8] * m2[2] + m1[9] * m2[6] + m1[10] * m2[10] + m1[11] * m2[14];
    mOut[11] = m1[8] * m2[3] + m1[9] * m2[7] + m1[10] * m2[11] + m1[11] * m2[15];

    mOut[12] = m1[12] * m2[0] + m1[13] * m2[4] + m1[14] * m2[8] + m1[15] * m2[12];
    mOut[13] = m1[12] * m2[1] + m1[13] * m2[5] + m1[14] * m2[9] + m1[15] * m2[13];
    mOut[14] = m1[12] * m2[2] + m1[13] * m2[6] + m1[14] * m2[10] + m1[15] * m2[14];
    mOut[15] = m1[12] * m2[3] + m1[13] * m2[7] + m1[14] * m2[11] + m1[15] * m2[15];
}

// 4x4 matrix inversion (matrices are stored as float arrays in row-major order)
bool invert_matrix(const float m[16], float invOut[16])
{
    float inv[16], det;
    int i;
    inv[0] = m[5] * m[10] * m[15] -
             m[5] * m[11] * m[14] -
             m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] +
             m[13] * m[6] * m[11] -
             m[13] * m[7] * m[10];

    inv[4] = -m[4] * m[10] * m[15] +
             m[4] * m[11] * m[14] +
             m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] -
             m[12] * m[6] * m[11] +
             m[12] * m[7] * m[10];

    inv[8] = m[4] * m[9] * m[15] -
             m[4] * m[11] * m[13] -
             m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    inv[12] = -m[4] * m[9] * m[14] +
              m[4] * m[10] * m[13] +
              m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] -
              m[12] * m[5] * m[10] +
              m[12] * m[6] * m[9];

    inv[1] = -m[1] * m[10] * m[15] +
             m[1] * m[11] * m[14] +
             m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] -
             m[13] * m[2] * m[11] +
             m[13] * m[3] * m[10];

    inv[5] = m[0] * m[10] * m[15] -
             m[0] * m[11] * m[14] -
             m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    inv[9] = -m[0] * m[9] * m[15] +
             m[0] * m[11] * m[13] +
             m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] -
             m[12] * m[1] * m[11] +
             m[12] * m[3] * m[9];

    inv[13] = m[0] * m[9] * m[14] -
              m[0] * m[10] * m[13] -
              m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    inv[2] = m[1] * m[6] * m[15] -
             m[1] * m[7] * m[14] -
             m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    inv[6] = -m[0] * m[6] * m[15] +
             m[0] * m[7] * m[14] +
             m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] -
             m[12] * m[2] * m[7] +
             m[12] * m[3] * m[6];

    inv[10] = m[0] * m[5] * m[15] -
              m[0] * m[7] * m[13] -
              m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    inv[14] = -m[0] * m[5] * m[14] +
              m[0] * m[6] * m[13] +
              m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] -
              m[12] * m[1] * m[6] +
              m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
             m[1] * m[7] * m[10] +
             m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] +
             m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
              m[0] * m[7] * m[9] +
              m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] +
              m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}

// g_srcImage = cv::imread("/home/u16/dataset/img/depth/0000.png");

// //判断图像是否加载成功
// if (g_srcImage.empty())
// {
//     std::cout << "图像加载失败!" << std::endl;
//     return ;
// }
// else
//     std::cout << "图像加载成功!" << std::endl
//               << std::endl;

// cv::imshow("原图像", g_srcImage);

// //定义输出图像窗口属性和轨迹条属性
// g_ndValue = 10;
// g_nsigmaColorValue = 10;
// g_nsigmaSpaceValue = 10;

// char dName[20];
// sprintf(dName, "邻域直径 %d", g_ndMaxValue);

// char sigmaColorName[20];
// sprintf(sigmaColorName, "sigmaColor %d", g_nsigmaColorMaxValue);

// char sigmaSpaceName[20];
// sprintf(sigmaSpaceName, "sigmaSpace %d", g_nsigmaSpaceMaxValue);

// //创建轨迹条
// cv::createTrackbar(dName, "双边滤波图像", &g_ndValue, g_ndMaxValue, on_bilateralFilterTrackbar);
// on_bilateralFilterTrackbar(g_ndValue, 0);

// cv::createTrackbar(sigmaColorName, "双边滤波图像", &g_nsigmaColorValue,
//                    g_nsigmaColorMaxValue, on_bilateralFilterTrackbar);
// on_bilateralFilterTrackbar(g_nsigmaColorValue, 0);

// cv::createTrackbar(sigmaSpaceName, "双边滤波图像", &g_nsigmaSpaceValue,
//                    g_nsigmaSpaceMaxValue, on_bilateralFilterTrackbar);
// on_bilateralFilterTrackbar(g_nsigmaSpaceValue, 0);
// cv::waitKey(0);