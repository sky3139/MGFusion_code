#pragma once
#include <stdio.h>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>


// class dataset
// {
// public:
//   cv::Mat posss;
//   std::string depth_path;
//   std::string color_path;
//   cv::Mat depth_src, _rgb_img, depth_, rgb_;
//   cv::Vec<float, 16> m_pose;
//   dataset(std::string path);
//   std::string ReadNextTUM(int frame_idx);
//   bool Mat_read_binary(cv::Mat &img_vec, std::string filename);
//   bool Mat_save_by_binary(cv::Mat &image, std::string filename); //单个写入
// };

cv::Mat_<float> LoadMatrixFromFile(std::string filename, int M, int N, float *);

void ReadDepth(std::string filename, int H, int W, float *depth);
void ReadDepth(cv::Mat &depth_mat, cv::Mat &srcImage, int H, int W, union points *point);
void multiply_matrix(const float m1[16], const float m2[16], float mOut[16]);
bool invert_matrix(const float m[16], float invOut[16]);
void ReadRGB(std::string filename, int H, int W, uint8_t *depth);
cv::Mat getCload(cv::Mat &depth, cv::Mat &point_tsdf, std::vector<uint32_t> &cubexs, \
cv::Mat_<float> &cam_pose, cv::Mat &p,struct kernelPara &centers);
