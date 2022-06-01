
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

string PATH ="./infinitam/";// "/home/lei/docker/res/InfiniTAM/Files/tum/Frames/";

map<string, string> rgbdepth;

int main()
{
    string folder_name = "/home/lei/app/dataset/rgbd_dataset_freiburg3_walking_xyz/"; //"freiburg3_office/";///root/app/lei/dataset/
    ifstream fin(folder_name + "associations.txt", std::ios::in);
    string ground_file = folder_name + "tmp.txt"; //"freiburg3_office/";///root/app/lei/dataset/
    ifstream gfin(ground_file, std::ios::in);

    cout << ground_file << endl;
    cout << folder_name + "associations.txt" << endl;

    assert(fin || gfin);
    char *pname = new char[128];

    string s0;
    // readsome( s0);
    getline(fin, s0);
    getline(fin, s0);
    getline(fin, s0);
    // getline(gfin, s0);
    // getline(gfin, s0);
    // getline(gfin, s0);
    // cout<<s0<<endl;
    std::string data_path = PATH;

    int base_frame_idx = 0;
    while (true)
    {
        double time[2];
        string rgb_path, depth_path;
        fin >> time[0] >> rgb_path >> time[1] >> depth_path;

        sprintf(pname, "/home/lei/docker/res/killingFusionCuda-master/data/hat/depth_%06d.png", base_frame_idx);
        Mat rgb = cv::imread(string(pname)); //;cv::IMREAD_GRAYSCALE
        sprintf(pname, "/home/lei/docker/res/killingFusionCuda-master/data/hat/color_%06d.png", base_frame_idx);

        Mat dept = cv::imread(string(pname), -1); //;cv::IMREAD_GRAYSCALE
        // Mat rgb = cv::imread(folder_name + rgb_path);        //;cv::IMREAD_GRAYSCALE
        // Mat dept = cv::imread(folder_name + depth_path, -1); //;cv::IMREAD_GRAYSCALE
        cv::imshow("rgb", rgb);
        cv::imshow("dept", dept);
        cv::waitKey(1);
        sprintf(pname, "%s%04d.ppm", PATH.c_str(), base_frame_idx);

        // cv::imwrite(pname, rgb);
        assert(rgb.data && "Can not load images!");
        assert(dept.data && "Can not load images!");
        sprintf(pname, "%s%04d.pgm", PATH.c_str(), base_frame_idx);
        // mat2ppm5save(pname, &dep);
        base_frame_idx++;
        Mat outsd;
        dept.convertTo(outsd, CV_16U, 0.1, 0);

        // cv::imwrite(pname, outsd);
        Mat asd = imread(pname);
        cv::applyColorMap(asd, asd, cv::COLORMAP_JET);
        cv::imshow("asd", asd);

        cout << string(pname) << endl;
        if (fin.eof())
            break;
    }
    fin.close();
}