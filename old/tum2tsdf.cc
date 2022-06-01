
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

string PATH = "/home/lei/docker/tsdf-fusion-master/data/rgbd-frames";

map<string, string> rgbdepth;


int main()
{
    string folder_name = "/home/lei/app/dataset/freiburg3_office/"; //"freiburg3_office/";///root/app/lei/dataset/
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
        rgbdepth[rgb_path] = depth_path;

        if (fin.eof())
            break;
    }
    fin.close();
    while (true) // for(int i=0;i<5;i++)
    {
        double time[2], x, y, z, q1, q2, q3, q4;
        string rgb_path, depth_path;
        gfin >> time[0] >> rgb_path >> time[1] >> x >> y >> z >> q1 >> q2 >> q3 >> q4;
        //    cout<< time[0] <<rgb_path << time[1]<< depth_path<<x<<y<<z<<q1<<q2<<q3<<q4;

        // cout<<rgb_path<<endl;

        string dep = rgbdepth[rgb_path];
        if (dep == "")
            continue;
        Mat rgb = cv::imread(folder_name + rgb_path); //;cv::IMREAD_GRAYSCALE
                                                      //         // sprintf(pname,"/home/lei/app/dataset/home/frame-%06d.depth.png",cnt);
                                                      //         // Mat dep = cv::imread(pname, -1); //;cv::IMREAD_GRAYSCALE
        std::ostringstream base_frame_prefix;
        base_frame_prefix << std::setw(6) << std::setfill('0') << base_frame_idx;
        std::string base2world_file = data_path + "/frame-" + base_frame_prefix.str() + ".color.png";
        cv::imwrite(base2world_file, rgb);

        base_frame_prefix.clear();
        //           cout<<"folder_name+dep:"<<folder_name+dep<<endl;
        Mat dept = cv::imread(folder_name + dep); //;cv::IMREAD_GRAYSCALE
        assert(dept.data && "Can not load images!");
        assert(rgb.data && "Can not load images!");
        base2world_file = data_path + "/frame-" + base_frame_prefix.str() + ".depth.png";
        // cout << base2world_file << endl;
        //   cv::imshow("asd",dept);
        cv::imwrite(base2world_file, dept);
        Eigen::Matrix<double, 3, 4> P;

        Eigen::Quaterniond quaternion(q1, q2, q3, q4);
        Eigen::Matrix3d rotation_matrix;
        rotation_matrix = quaternion.toRotationMatrix();
        Eigen::Matrix<double, 3, 1> t;

        t << x, y, z;
        P.block(0, 0, 3, 3) = rotation_matrix; //.transpose(); //转置
        P.col(3) = t;
        // cout << P.matrix() << endl;

        // base_frame_prefix.clear();
        base2world_file = data_path + "/frame-" + base_frame_prefix.str() + ".pose.txt";
        cout<<base2world_file<<endl;
        ofstream fout(base2world_file, std::ios::out);
        fout << P.matrix() << endl;
        fout << "0 0 0 1";
        fout.close();
        base_frame_idx++;
        if (gfin.eof())
            break;
    }
    gfin.close();
}