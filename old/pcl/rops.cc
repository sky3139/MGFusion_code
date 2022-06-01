#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/viz.hpp"
using namespace cv;
using namespace std;
#include <opencv2/viz/vizcore.hpp>
// #include "pcl_tsdf.hpp"
using namespace cv;
using namespace std;

cv::Mat ReadRGB(cv::Mat &depth)
{

    if (depth.empty())
    {
        assert(0);
    }
    cv::imshow("rgb", depth);
    cv::waitKey(1);
    Mat point_cloud; // = Mat::zeros(height, width, CV_32FC3);

    for (int row = 0; row < depth.rows; row++)
        for (int col = 0; col < depth.cols; col++)
        {

            uchar *temp_ptr = &((uchar *)(depth.data + depth.step * row))[col * 3];

            Vec3b vec;
            vec[0] = temp_ptr[0];
            vec[1] = temp_ptr[1];
            vec[2] = temp_ptr[2];
            point_cloud.push_back(vec);
        }
    return point_cloud;
}
void LoadMatrixFromFile(std::string filename, double *matrix, int M, int N)
{
    FILE *fp = fopen(filename.c_str(), "r");
    for (int i = 0; i < M * N; i++)
    {

        int iret = fscanf(fp, "%lf", &matrix[i]);
    }
    fclose(fp);
    return;
}

void depth2color(cv::Mat &color, const cv::Mat &depth, const double max, const double min)
{

    // double max=0, min=70000;
    // int imrow = depth.rows;
    // int imcol = depth.cols * depth.channels();
    // for (int i = 0; i < imrow; i++)
    // {
    //     for (int j = 0; j < imcol; j++)
    //     {
    //         ushort data = depth.at<ushort>(i, j);
    //         if (min >= data && data!=0)
    //         {
    //             min = data;
    //         }
    //         if (max <= data)
    //         {
    //             max = data;
    //         }

    //     }
    // }

    cv::Mat grayImage;
    double alpha = 255.0 / (max - min);
    depth.convertTo(grayImage, CV_8UC1, alpha, -alpha * min); // expand your range to 0..255. Similar to histEq();
    cv::applyColorMap(grayImage, color, cv::COLORMAP_JET);    // this is great. It converts your grayscale image into a tone-mapped one, much more pleasing for the eye function is found in contrib module, so include contrib.hpp  and link accordingly
}
#include <fstream>

bool Mat_save_by_binary(cv::Mat &image, string filename) //单个写入
{
    int channl = image.channels();
    int rows = image.rows;
    int cols = image.cols;
    short em_size = image.elemSize();
    short type = image.type();

    fstream file(filename, ios::out | ios::binary); // | ios::app
    file.write(reinterpret_cast<char *>(&channl), 1);
    file.write(reinterpret_cast<char *>(&type), 1);
    file.write(reinterpret_cast<char *>(&em_size), 2);
    file.write(reinterpret_cast<char *>(&cols), 4);
    file.write(reinterpret_cast<char *>(&rows), 4);
    printf("SAVE:cols=%d,type=%d,em_size=%d,rows=%d,channels=%d\n", cols, type, em_size, rows, channl);
    file.write(reinterpret_cast<char *>(image.data), em_size * cols * rows * channl);
    file.close();
    return true;
}
bool Mat_read_binary(cv::Mat &img_vec, string filename) //整体读出
{
    int channl(0);
    int rows(0);
    int cols(0);
    short type(0);
    short em_size(0);
    ifstream fin(filename, ios::binary);
    fin.read((char *)&channl, 1);
    fin.read((char *)&type, 1);
    fin.read((char *)&em_size, 2);
    fin.read((char *)&cols, 4);
    fin.read((char *)&rows, 4);
    printf("SAVE:cols=%d,type=%d,em_size=%d,rows=%d,channels=%d\n", cols, type, em_size, rows, channl);
    img_vec = cv::Mat(rows, cols, type);
    fin.read((char *)&img_vec.data[0], rows * cols * em_size);
    fin.close();
    return true;
}
int main()
{
    // vtkSmartPointer<vtkCamera> camera;
    //加载深度图（这里用的kinect2 的深度图）
    //初始化
    viz::Viz3d window("window");
    //显示坐标系
    window.showWidget("Coordinate", viz::WCoordinateSystem());

    //创建一个储存point cloud的图片
    char *name = new char[256];
    int i = 10;

    Mat point_cloud2, colora;
    // Mat_read_binary(point_cloud2, string("/home/lei/kin/kint/build/qp.bin"));
    // Mat_read_binary(colora, string("/home/lei/kin/kint/build/qc.bin"));
    Mat_read_binary(point_cloud2, string("/home/lei/kin/kint/build/points.bin"));
    Mat_read_binary(colora, string("/home/lei/kin/kint/build/cvcolr.bin"));
    // cv::viz::WCloud cloud(point_cloud1, cv::viz::Color::yellow());
    // window.showWidget("cloud", cloud);
    // sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/depth-%d.png", i++);
    // depth = cv::imread(name, 2);
    // Mat point_cloud2 = getCload(depth);

    // cv::viz::WCloud cloud2(point_cloud2, colora);
    // window.showWidget("cloud2", cloud2);

    double min;
    double max;
    int maxIDX;
    int minIDX;

    vector<Mat> channels;
    vector<Mat> channelsForMege;
    Mat imgBlueChannel;
    Mat imgGrayChannel;
    Mat imgRedChannel;

    //分离处颜色通道
    split(point_cloud2, channels);

    cv::minMaxIdx(channels[0], &min, &max, &minIDX, &maxIDX);

    std::cout << "min:" << min << " max:" << max << " minIDX:" << minIDX << " maxIDX:" << maxIDX << endl;
    Mat point3, color3;
    Mat point4, color4;
    for (size_t i = 0; i < 1110000; i++)
    {
        // if (point_cloud2.at<cv::Vec3f>(0, i)[1] < 0.35)
        {
            point3.push_back(point_cloud2.at<cv::Vec3f>(0, i));
            color3.push_back(colora.at<cv::Vec3b>(0, i));
        }
        // point_cloud2.at<cv::Vec3f>(0,i);
    }
    for (size_t i = 1110000; i < point_cloud2.rows; i++)
    {
        // if (point_cloud2.at<cv::Vec3f>(0, i)[1] < 0.35)
        {
            point4.push_back(point_cloud2.at<cv::Vec3f>(0, i));
            color4.push_back(colora.at<cv::Vec3b>(0, i));
        }
        // point_cloud2.at<cv::Vec3f>(0,i);
    }

    cv::Vec3d t(5, 5, 5);
    cv::viz::WCloud cloud3(point3, color3);
    cv::Affine3d aff(cv::Mat::eye(3, 3, CV_64FC1), t);
    window.showWidget("cloud3", cloud3, aff);


    cv::viz::WCloud cloud4(point4, color4);
    window.showWidget("cloud4", cloud4);


    while (!window.wasStopped())
    {
        // sprintf(name, "/home/lei/docker/res/killingFusionCuda-master/data/hat/depth_%06d.png", i++);
        // sprintf(name, "/home/lei/dataset/infiniTAM/Files/tum/Frames/%04d.pgm", i++);
        // sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/depth-%d.png", i++);

        // // std::string filename = "" + to_string(i++) + ".png";
        // // sprintf(name,, i);
        // cv::Mat depth = cv::imread("/home/lei/dataset/img/depth/0.png", 2);

        // // Mat pcasdco; // = ReadRGB(name);
        // Mat point_cloud1 = getCload(depth);
        // cv::imshow("depth", depth);
        // if (point_cloud1.rows == 0)
        //     continue;
        // Mat_save_by_binary(point_cloud1, "point_cloud1");

        window.spinOnce(1, false);
    }
    return 0;
}

int main2222222222()
{
    // vtkSmartPointer<vtkCamera> camera;
    //加载深度图（这里用的kinect2 的深度图）
    //初始化
    viz::Viz3d window("window");
    //显示坐标系
    window.showWidget("Coordinate", viz::WCoordinateSystem());

    //创建一个储存point cloud的图片
    char *name = new char[256];
    int i = 10;
    cv::Mat mat;
    // Mat_read_binary<float>(mat, std::string("/home/lei/kin/kint/build/points.bin"));
    while (!window.wasStopped())
    {
        sprintf(name, "/home/lei/docker/res/killingFusionCuda-master/data/hat/depth_%06d.png", i++);
        sprintf(name, "/home/lei/dataset/infiniTAM/Files/tum/Frames/%04d.pgm", i++);
        cv::viz::WCloudCollection cloud;

        sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/depth-%d.png", i);
        cv::Mat depth = cv::imread(name, 2);
        // sprintf(name, "./infinitam/%04d.pgm", i - 10);
        // cv::imwrite(name, depth);

        double cam2world[16];
        sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/%d.txt", i);
        std::string cam2world_file = string(name); //data_path + "/frame-" + curr_frame_prefix.str() + ".pose.txt";
        LoadMatrixFromFile(cam2world_file, cam2world, 4, 4);
        Affine3d pose(cam2world);
        cout << pose.matrix << " " << endl;
        sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/color-%d.png", i++);
        cv::Mat _reacolor = cv::imread(name);
        //**************************
        Mat point_cloud1 = getCload(depth);
        if (point_cloud1.rows == 0)
            continue;
        cloud.addCloud(point_cloud1, cv::viz::Color::white(), pose);
        //**********************
        sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/depth-%d.png", i);
        depth = cv::imread(name, 2);
        // sprintf(name, "./infinitam/%04d.pgm", i - 10);
        // cv::imwrite(name, depth);

        sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/%d.txt", i);
        LoadMatrixFromFile(string(name), cam2world, 4, 4);
        Affine3d pose2(cam2world);

        //**************************
        cout << pose2.matrix << " " << endl;

        cloud.addCloud(getCload(depth), cv::viz::Color::yellow(), pose2);
        cv::Mat wcolor;
        depth2color(wcolor, depth, 5000, 0);
        cv::imshow("wcolor", wcolor);
        sprintf(name, "./infinitam/%04d.ppm", i - 11);
        cv::imwrite(name, wcolor);
        Mat pcasdco = ReadRGB(wcolor);
        cout << mat << endl;
        // cv::viz::WCloud cloud(mat);
        cloud.addCloud(mat, cv::viz::Color::yellow());
        // window.showWidget("cloud", cloud);
        window.spinOnce(1, false);
    }

    return 0;
}