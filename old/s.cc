#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main2()
{
    /// Create a window
    viz::Viz3d myWindow("Coordinate Frame");
    /// Add coordinate axes
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
    /// Add line to represent (1,1,1) axis
    viz::WLine axis(Point3f(-1.0f, -1.0f, -1.0f), Point3f(1.0f, 1.0f, 1.0f));
    axis.setRenderingProperty(viz::LINE_WIDTH, 4.0);
    myWindow.showWidget("Line Widget", axis);
    /// Construct a cube widget
    viz::WCube cube_widget(Point3f(0.5, 0.5, 0.0), Point3f(0.0, 0.0, -0.5), true, viz::Color::blue());
    cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);
    /// Display widget (update if already displayed)
    myWindow.showWidget("Cube Widget", cube_widget);
    /// Rodrigues vector
    Mat rot_vec = Mat::zeros(1, 3, CV_32F);
    float translation_phase = 0.0, translation = 0.0;
    while (!myWindow.wasStopped())
    {
        //* Rotation using rodrigues
        /// Rotate around (1,1,1)
        rot_vec.at<float>(0, 0) += CV_PI * 0.01f;
        rot_vec.at<float>(0, 1) += CV_PI * 0.01f;
        rot_vec.at<float>(0, 2) += CV_PI * 0.01f;
        /// Shift on (1,1,1)
        translation_phase += CV_PI * 0.01f;
        translation = sin(translation_phase);
        Mat rot_mat;
        Rodrigues(rot_vec, rot_mat);
        /// Construct pose
        Affine3f pose(rot_mat, Vec3f(translation, translation, translation));
        myWindow.setWidgetPose("Cube Widget", pose);
        myWindow.spinOnce(1, true);
    }
    return 0;
}

#include "opencv2/opencv.hpp"
#include "opencv2/viz.hpp"
using namespace cv;
using namespace std;

int main23()
{
    //加载深度图（这里用的kinect2 的深度图）
    std::string filename = "/home/lei/dataset/freiburg3_office/depth/1341847980.723020.png";
    // std::string filename = "/media/lei/7cf850e3-374b-4022-bc4f-3a10d2fefca3/home/lei/桌面/SLAMBench_1_1/SLAMBench_1_1/synthetic/test_00003.png";
    cv::Mat depth = cv::imread(filename, CV_16UC1);
    //初始化
    viz::Viz3d window("window");
    //显示坐标系
    window.showWidget("Coordinate", viz::WCoordinateSystem());
    int height = depth.rows;
    int width = depth.cols;
    // double fx = 570.342, fy = 570.34, cx = 320, cy = 240;
    double fx = 1, fy = 1, cx = 0, cy = 0;
    char *pname = new char[256];
    int idx = 1;
    while (!window.wasStopped())
    {
        // sprintf(pname, "/home/lei/dataset/freiburg3_office/dd/%d.png", idx++);

        sprintf(pname, "/home/lei/文档/pic/%d.png", idx);
        cv::Mat depth = cv::imread(pname, -1); //
        assert(depth.data);
        std::cout << depth << endl;
        //创建一个储存point cloud的图片
        // Mat cloud(rows, cols, CV_32FC3);

        Mat point_cloud; //= Mat::zeros(height, width, CV_32FC3);
        //point cloud 赋值，其中 fx,fy,cx,cy 为Kinect2 的内参
        for (int row = 0; row < depth.rows; row++)
            for (int col = 0; col < depth.cols; col++)
            {
                float dz = depth.at<uint8_t>(row, col);
                std::cout << dz << " ";
                if (dz < 0.01 || dz > 250.0f) //
                {
                    continue;
                }
                dz /= 100;
                Vec3f ves;

                ves[0] = dz * (col - cx) / fx;
                ves[1] = dz * (row - cy) / fy;
                ves[2] = dz;
                point_cloud.push_back(ves);
            }
        std::cout << endl;
        ///创建polygons矩阵，并为其赋值，它的第0个元素是三维点的个数
        /// 剩下的元素值分别为1、2、3......、N。此次的N表示总的三维点数
        // Mat polygon(1, width * height + 1, CV_32SC1, Scalar::all(0));
        // polygon.ptr<int>(0)[0] = width * height;
        // for (int i = 1; i <= polygon.cols; i++)
        //     polygon.ptr<int>(0)[i] = i;
        ///下面是把二维的cloud转化为一维的cloud
        // Mat cloud1d = point_cloud.reshape(0, point_cloud.rows * point_cloud.cols);
        // cloud1d = cloud1d.t();
        ///创建并赋值Mesh对象
        // viz::Mesh mesh;
        // mesh.cloud = cloud1d;
        // mesh.polygons = polygon;
        ///创建颜色矩阵
        // Mat color;
        // color=img.reshape(0,rows*cols);
        // color=color.t();
        // mesh.colors=color;
        ///mesh本身无法在三维窗口中显示，需要经过WMesh处理，得到WMesh对象wmesh
        // viz::WMesh wmesh(mesh);

        ///显示wmesh
        // imshow("original image",img);

        cv::viz::WCloud cloud(point_cloud);
        window.showWidget("cloud", cloud);
        cv::imshow("depth", depth);
        // cv::waitKey(0);
        window.spinOnce(1, false);
    }
    return 0;
}

#include <opencv2/viz/vizcore.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
using namespace cv;
using namespace std;
//#include "myfunction.h"

int maina()
{
    Mat img = imread("/home/lei/app/u.png");
    ; //imread("/home/lei/dataset/freiburg3_office/depth/1341847980.723020.png",0);

    Mat gray = imread("/home/lei/app/u.png");
    img.convertTo(gray, CV_32FC1);

    int rows = gray.rows, cols = gray.cols;
    //为了赋值方便，这里我用了与img同行、同列的矩阵
    Mat cloud(rows, cols, CV_32FC3);
    int cx = cols / 2, cy = rows / 2;

    //为cloud矩阵赋值
    for (int i = 0; i < rows; i++)
    {
        float y = i - cy;
        for (int j = 0; j < cols; j++)
        {
            float x = j - cx;
            cloud.ptr<Vec3f>(i)[j][0] = x;
            cloud.ptr<Vec3f>(i)[j][1] = -y;
            cloud.ptr<Vec3f>(i)[j][2] = (img.ptr<Vec3b>(i)[j][0] + img.ptr<Vec3b>(i)[j][1] + img.ptr<Vec3b>(i)[j][2]) / 3;
        }
    }
    ///创建polygons矩阵，并为其赋值，它的第0个元素是三维点的个数
    /// 剩下的元素值分别为1、2、3......、N。此次的N表示总的三维点数
    Mat polygon(1, cols * rows + 1, CV_32SC1, Scalar::all(0));
    polygon.ptr<int>(0)[0] = rows * cols;
    for (int i = 1; i <= polygon.cols; i++)
        polygon.ptr<int>(0)[i] = i;
    ///下面是把二维的cloud转化为一维的cloud
    Mat cloud1d = cloud.reshape(0, cloud.rows * cloud.cols);
    cloud1d = cloud1d.t();
    ///创建并赋值Mesh对象
    viz::Mesh mesh;
    mesh.cloud = cloud1d;
    mesh.polygons = polygon;
    ///创建颜色矩阵
    Mat color;
    color = img.reshape(0, rows * cols);
    color = color.t();
    mesh.colors = color;
    ///mesh本身无法在三维窗口中显示，需要经过WMesh处理，得到WMesh对象wmesh
    viz::WMesh wmesh(mesh);

    ///显示wmesh
    imshow("original image", img);
    viz::Viz3d my3DWin("My 3D Window");
    my3DWin.setBackgroundColor(viz::Color::cyan());
    my3DWin.showWidget("wmesh", wmesh);

    my3DWin.spin();

    return 0;
}

#include "opencv2/opencv.hpp"
#include "opencv2/viz.hpp"
using namespace cv;
using namespace std;

// #include <vtkVersion.h>
// #include <vtkConfigure.h>
// #include <iostream>
// #include <vtkSmartPointer.h>
// #include <vtkSphereSource.h>
// #include <vtkActor.h>

// #include <vtkRenderWindow.h>
// #include <vtkPolyDataMapper.h>
// #include <vtkRenderWindowInteractor.h>

// #define vtkSPtr vtkSmartPointer
// #define vtkSPtrNew(Var, Type) vtkSPtr<Type> Var = vtkSPtr<Type>::New();

int main()
{
    // vtkSmartPointer<vtkCamera> camera;
    //加载深度图（这里用的kinect2 的深度图）

    //初始化
    viz::Viz3d window("window");

    // cout << depth << endl;
    //显示坐标系
    window.showWidget("Coordinate", viz::WCoordinateSystem());

    //创建一个储存point cloud的图片
    char *name = new char[256];
    int i = 10;
    while (!window.wasStopped())
    {
        // sprintf(name, "/home/lei/docker/res/killingFusionCuda-master/data/hat/depth_%06d.png", i++);
        sprintf(name, "/home/lei/dataset/infiniTAM/Files/tum/Frames/%04d.pgm", i++);

        // std::string filename = "" + to_string(i++) + ".png";
        cv::Mat depth = cv::imread(name, 2);
        cout << name << endl;
        Mat point_cloud; // = Mat::zeros(height, width, CV_32FC3);
        //point cloud 赋值，其中 fx,fy,cx,cy 为Kinect2 的内参
        double asd = 500;
        double fx = asd, fy = asd, cx = 320.0, cy = 240.0;

        for (int row = 0; row < depth.rows; row++)
            for (int col = 0; col < depth.cols; col++)
            {
                float dz = ((float)depth.at<unsigned short>(row, col)) / 1000.0;

                if (dz < 0.11)
                    continue;
                // if (dz > 0.85)
                //     continue;
                std::cout << dz << " ";
                Vec3f vec;
                // dz ;
                vec[0] = dz * (col - cx) / fx;
                vec[1] = dz * (row - cy) / fy;
                vec[2] = dz;
                point_cloud.push_back(vec);
            }
        if (point_cloud.rows == 0)
            continue;
        // if (i >= 100)
        // {
        //     i = 0;
        // }
        cv::viz::WCloud cloud(point_cloud);
        window.showWidget("cloud", cloud);

        window.spinOnce(1, false);
    }
    return 0;
}
