#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/viz.hpp"
using namespace cv;
using namespace std;
#include <opencv2/viz/vizcore.hpp>

void Save2SurfacePointCloud(const std::string &file_name, cv::Mat &points, cv::Mat &color)
{

    FILE *fp = fopen(file_name.c_str(), "w");
    fprintf(fp, "ply\n");
    fprintf(fp, "format binary_little_endian 1.0\n");
    fprintf(fp, "element vertex %d\n", points.rows);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "end_header\n");

    // Create point cloud content for ply file
    for (int i = 0; i < points.rows; i++)
    {

        // Convert voxel indices to float, and save coordinates to ply file
        float *pt_base_x = (float *)&points.at<Vec3f>(0, i)[0];

        // float *pt_base_x = (float *)&points.at<Vec3f>(0, i)[0];
        // float *pt_base_y = (float *)&points.at<Vec3f>(0, i)[1];
        // float *pt_base_z = (float *)&points.at<Vec3f>(0, i)[2];

        uint8_t *bgr = (uint8_t *)&color.at<Vec3b>(0, i)[0];
        fwrite(pt_base_x, sizeof(float), 3, fp);
        // fwrite(pt_base_y, sizeof(float), 1, fp);
        // fwrite(pt_base_z, sizeof(float), 1, fp);

        fwrite(bgr, sizeof(uint8_t), 3, fp);
    }
    fclose(fp);
}

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
cv::Mat getCload(cv::Mat &depth)
{
    Mat point_cloud; // = Mat::zeros(height, width, CV_32FC3);
    //point cloud 赋值，其中 fx,fy,cx,cy 为Kinect2 的内参
    double asd = 500;
    double fx = asd, fy = asd, cx = 320.0, cy = 240.0;

    for (int row = 0; row < depth.rows; row++)
        for (int col = 0; col < depth.cols; col++)
        {
            float dz = ((float)depth.at<unsigned short>(row, col)) / 1000.0;

            Vec3f vec;
            // dz ;
            vec[0] = dz * (col - cx) / fx;
            vec[1] = dz * (row - cy) / fy;
            vec[2] = dz;
            point_cloud.push_back(vec);
        }
    return point_cloud;
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
void addpylogo(viz::Viz3d &win, Mat &point2, string name, Mat &color)
{
    Mat polygon;
    polygon.push_back(point2.rows);

    for (int i = 1; i <= point2.rows; i++)
    {
        polygon.push_back(i);
    }
    viz::Mesh mesh;
    mesh.cloud = point2.t();
    mesh.polygons = polygon;
    mesh.colors = color.t();
    viz::WMesh wmesh(mesh);
    // win.showWidget("wmesh", wmesh);
}
int mainpylogo()
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

    // std::cout << "min:" << min << " max:" << max << " minIDX:" << minIDX << " maxIDX:" << maxIDX << endl;
    // Mat point3, color3;
    // Mat point4, color4;
    // std::cout << "min:" << point_cloud2.rows << endl;
    // for (size_t i = 0; i < 100000; i++)
    // {
    //     // if (point_cloud2.at<cv::Vec3f>(0, i)[1] < 0.35)
    //     {
    //         point3.push_back(point_cloud2.at<cv::Vec3f>(0, i));
    //         color3.push_back(colora.at<cv::Vec3b>(0, i));
    //     }
    //     // point_cloud2.at<cv::Vec3f>(0,i);
    // }

    // for (size_t i = 2420000; i < point_cloud2.rows; i++)
    // {
    //     // if (point_cloud2.at<cv::Vec3f>(0, i)[1] < 0.35)
    //     {
    //         point4.push_back(point_cloud2.at<cv::Vec3f>(0, i));
    //         color4.push_back(colora.at<cv::Vec3b>(0, i));
    //     }
    //     // point_cloud2.at<cv::Vec3f>(0,i);
    // }

    // cv::Vec3d t(5, 5, 5);
    // cv::viz::WCloud cloud3(point3, color3);
    // cv::Affine3d aff(cv::Mat::eye(3, 3, CV_64FC1), t);
    // window.showWidget("cloud3", cloud3, aff);

    // cv::viz::WCloud cloud4(point4, color4);
    // window.showWidget("cloud4", cloud4);
    // Save2SurfacePointCloud("p1.ply", point4, color4);
    // Save2SurfacePointCloud("p2.ply", point3, color3);

    ///显示wmesh
    // imshow("original image", img);
    viz::Viz3d win("My 3D Window");
    // my3DWin.setBackgroundColor(viz::Color::cyan());

    cv::Mat point;
    Mat color;
    point.push_back(cv::Vec3f(0, 0, 0));
    point.push_back(cv::Vec3f(0, 0, 1));
    point.push_back(cv::Vec3f(0, 1, 1));
    point.push_back(cv::Vec3f(0, 2, 1.5));
    point.push_back(cv::Vec3f(0, 1, 2));
    point.push_back(cv::Vec3f(0, 1.4, 2));

    color.push_back(cv::Vec3b(0, 0, 0));
    color.push_back(cv::Vec3b(0, 0, 1));
    color.push_back(cv::Vec3b(0, 1, 1));
    color.push_back(cv::Vec3b(0, 2, 3));
    color.push_back(cv::Vec3b(0, 1, 2));
    color.push_back(cv::Vec3b(0, 3, 2));

    addpylogo(win, point, "asdasd", color);

    while (!win.wasStopped())
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
        win.spinOnce(1, false);

        // window.spinOnce(1, false);
    }
    return 0;
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
    int i = 0;
    cv::Mat mat;
    // Mat_read_binary<float>(mat, std::string("/home/lei/kin/kint/build/points.bin"));
    while (!window.wasStopped())
    {
        // sprintf(name, "/home/u16/docker/img/rgb/%d.png", i++);
        sprintf(name, "/home/u16/dataset/home/frame-%06d.depth.png", i);
        cv::viz::WCloudCollection cloud;

        // // sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/depth-%d.png", i);
        cv::Mat depth = cv::imread(name, 2);
        // // // sprintf(name, "./infinitam/%04d.pgm", i - 10);
        // // // cv::imwrite(name, depth);

        //对Mat进行赋值和其他操作
        double max, min;
        cv::Point min_loc, max_loc;
        cv::minMaxLoc(depth, &min, &max, &min_loc, &max_loc);
        // std::cout << "max:" << max << std::endl;
        double cam2world[16];
        sprintf(name, "/home/u16/dataset/home/frame-%06d.pose.txt", i);
        std::string cam2world_file = string(name); //data_path + "/frame-" + curr_frame_prefix.str() + ".pose.txt";
        // std::cout<<name<<endl;
        LoadMatrixFromFile(cam2world_file, cam2world, 4, 4);
        Affine3d pose(cam2world);
        // cout << pose.matrix << " " << endl;
        sprintf(name, "/home/u16/dataset/home/frame-%06d.color.jpg", i);
        cv::Mat _reacolor = cv::imread(name);
        cv::Mat gray = cv::imread(name, 0);

        // // //**************************
        Mat point_cloud1 = getCload(depth);
        if (point_cloud1.rows == 0)
            continue;
        cloud.addCloud(point_cloud1, cv::viz::Color::white(), pose); //pose

        // Matx33f intrisicParams(K(0, 0), 0.0, K(0, 2), 0.0, K(1, 1), K(1, 2), 0.0, 0.0, 1.0);   // 内参矩阵
        cv::viz::Camera mainCamera = cv::viz::Camera::KinectCamera(Size(640, 480));           //new viz::Camera();         // 初始化相机类
        viz::WCameraPosition camParamsp(mainCamera.getFov(), gray, 1.0, viz::Color::white()); // 相机参数设置
        window.showWidget("Camera", camParamsp, pose);                                        // cv::Affine3f(pose)

        // //**********************
        // sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/depth-%d.png", i);
        // depth = cv::imread(name, 2);
        // // sprintf(name, "./infinitam/%04d.pgm", i - 10);
        // // cv::imwrite(name, depth);
        Mat test;
        int P = 800; //高度
        int H = 640 * 1.3;
        int W = 480 * 1.3;
        int pp = P / 1.3;
        for (int i = 0; i < P;) //
        {
            for (int j = i * 0.5; j < H - i * 0.5;)
            {
                for (int k = i * 0.5; k < (W - i * 0.5);) //(500-i)*4
                {
                    Vec3f vec;
                    vec[2] = pp * 0.006 - i * 0.006f;
                    vec[1] = j * 0.006f - H * 0.006 * 0.5;
                    vec[0] = k * 0.006f - W * 0.006 * 0.5;
                    test.push_back(Vec3f(vec));
                    k += 15;
                }

                j += 15;
            }
            i += 15;
        }
        Mat largetest;

        P = P * 1.15; //高度
        H *= 1.3;
        W *= 1.3;
        pp = P / 1.3;

        for (int i = 0; i < P;) //
        {
            for (int j = i * 0.5; j < H - i * 0.5;)
            {
                for (int k = i * 0.5; k < (W - i * 0.5);) //(500-i)*4
                {
                    Vec3f vec;
                    vec[2] = pp * 0.006 - i * 0.006f;
                    vec[1] = j * 0.006f - H * 0.006 * 0.5;
                    vec[0] = k * 0.006f - W * 0.006 * 0.5;
                    largetest.push_back(Vec3f(vec));
                    k += 15;
                }
                j += 15;
            }
            i += 15;
        }
        cout << "P:" << P << " W:" << W << " H:" << H << " pp: " << pp << " " << largetest.rows / 1024.0 / 1024.0*6*15*225<<"MB" << std::endl;
        cloud.addCloud(largetest, cv::viz::Color::blue(), pose); //,cv::Affine3f(pose)
        cloud.addCloud(test, cv::viz::Color::yellow(), pose); //,cv::Affine3f(pose)
        //         for (int i = z; i < 90;)
        //         {
        //             Vec3f vec;

        //             for (int j = z; j < 90;)
        //             {
        //             //    for (int k = 0; k < 100;) //(500-i)*4
        //                 {
        //                     Vec3f vec;
        //                     int k=10;
        //                     vec[2] = 1 + i * 0.006f;

        //                     vec[1] = 1 + j * 0.006f;

        //                     // dz ;
        //                     vec[0] = 1 + k * 0.006f;
        //                     test.push_back(Vec3f(vec));
        //                     k += 5;
        //                 }

        //                 j += 5;
        //             }
        // // z++;
        //             i += 5;
        //         }

        // for (int i = 5; i < 100;)
        // {
        //     for (int j = 5; j < 100;)
        //     {
        //         for (int k =  5; k < 10;) //(500-i)*4
        //         {

        //             Vec3f vec;
        //             // dz ;
        //             vec[0] = 1 + k * 0.006f;
        //             vec[1] = 1 + j * 0.006f;
        //             vec[2] = 1 + i * 0.006f;
        //             test.push_back(vec);
        //             k += 5;
        //         }
        //         j += 5;
        //     }
        //     i += 5;
        // }

        // window.showWidget("test", cloud);

        // sprintf(name, "/home/lei/docker/ros/src/gazebo_with_camera/mrobot_teleop/scripts/img/%d.txt", i);
        // LoadMatrixFromFile(string(name), cam2world, 4, 4);
        // Affine3d pose2(cam2world);

        // //**************************
        // cout << pose.matrix << " " << endl;
        i++;
        // cloud.addCloud(getCload(depth), cv::viz::Color::yellow(), pose2);
        // cv::Mat wcolor;
        // depth2color(wcolor, depth, 5000, 0);
        cv::imshow("wcolor", _reacolor);
        // sprintf(name, "./infinitam/%04d.ppm", i - 11);
        // cv::imwrite(name, wcolor);
        // Mat pcasdco = ReadRGB(wcolor);
        // cout << mat << endl;
        // // cv::viz::WCloud cloud(mat);
        // cloud.addCloud(mat, cv::viz::Color::yellow());
        window.showWidget("cloud", cloud);
        // cv::waitKey();
        // break;
        window.spinOnce(1, false);
        // break;
    }
    window.spin();

    return 0;
}