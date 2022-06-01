
#include <iostream>
#include <algorithm>
#include <fstream>
#include <map>
#include <iomanip>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>
using namespace std;
using namespace cv;

string PATH = "./infinitam/"; // "/home/lei/docker/res/InfiniTAM/Files/tum/Frames/";

map<string, string> rgbdepth;

int main()
{

    char *pname = new char[128];

    std::string data_path = PATH;

    int base_frame_idx = 0;
    while (true)
    {

        sprintf(pname, "/home/lei/docker/res/killingFusionCuda-master/data/hat/depth_%06d.png", base_frame_idx);
        Mat dept = cv::imread(string(pname), -1); //;cv::IMREAD_GRAYSCALE

        sprintf(pname, "/home/lei/docker/res/killingFusionCuda-master/data/hat/color_%06d.png", base_frame_idx);
        Mat rgb = cv::imread(string(pname)); //;cv::IMREAD_GRAYSCALE

        // Mat rgb = cv::imread(folder_name + rgb_path);        //;cv::IMREAD_GRAYSCALE
        // Mat dept = cv::imread(folder_name + depth_path, -1); //;cv::IMREAD_GRAYSCALE
        cv::imshow("rgb", rgb);
        cv::imshow("dept", dept);
        sprintf(pname, "%s%04d.ppm", PATH.c_str(), base_frame_idx);

        cv::imwrite(pname, rgb);
        assert(rgb.data && "Can not load images!");
        assert(dept.data && "Can not load images!");
        sprintf(pname, "%s%04d.pgm", PATH.c_str(), base_frame_idx);
        // base_frame_idx++;
        Mat outsd;
        dept.convertTo(outsd, CV_16U, 1, 0);

        cv::imwrite(pname, outsd);
        Mat asd = imread(pname);
        cv::applyColorMap(asd, asd, cv::COLORMAP_JET);
        cv::imshow("asd", asd);
        cv::waitKey(0);

        cout << string(pname) << endl;
        base_frame_idx++;
    }
}