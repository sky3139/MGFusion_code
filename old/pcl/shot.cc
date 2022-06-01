#include <iostream> //输入输出流
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <ctime>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/fpfh.h>     //FPFH
#include <pcl/features/pfh.h>      //PFH
#include <pcl/features/shot_omp.h> //SHOT
#include <pcl/features/shot.h>     //SHOT

// #include <pcl/visualization/histogram_visualizer.h> //直方图的可视化
// #include <pcl/visualization/pcl_plotter.h>          // 直方图的可视化 方法2
// #include <pcl/registration/ia_ransac.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <boost/thread/thread.hpp>
// #include <pcl/features/fpfh_omp.h> //openmp加速计算
// #include <pcl/registration/correspondence_estimation.h>
// #include <pcl/registration/correspondence_rejection_features.h>
// #include <pcl/registration/correspondence_rejection_sample_consensus.h>

// // #include <iostream>
// // #include <fstream>
// // #include <string>
// // #include <iostream>
// // #include <cstdlib>
// // #include <pcl/io/io.h>
// // #include <pcl/io/pcd_io.h>
// // #include <opencv2/core/core.hpp>
// // #include <opencv2/highgui/highgui.hpp>
// // #include <opencv2/highgui/highgui_c.h>
// // #include <opencv2/imgcodecs/imgcodecs.hpp>
// // #include <opencv2/core/hal/interface.h>
// // #include <opencv2/imgproc/imgproc.hpp>
// // #include <pcl/features/normal_3d.h>
// // #include <pcl/features/principal_curvatures.h>
// // #include <pcl/gpu/features/features.hpp>
// // bool getModelCurvatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int k)//, vector<PCURVATURE> &tempCV
// // {

// //     if (cloud->size() == 0)
// //     {
// //         return false;
// //     }

// //     pcl::gpu::NormalEstimation::PointCloud gpuCloud;

// //     pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
// //     kdtree->setInputCloud(cloud);

// //     size_t cloud_size = cloud->points.size();

// //     std::vector<float> dists;
// //     std::vector<std::vector<int> > neighbors_all;
// //     std::vector<int> sizes;
// //     neighbors_all.resize(cloud_size);
// //     sizes.resize(cloud_size);
// // #pragma omp parallel for
// //     for (int64 i = 0; i < cloud_size; ++i)
// //     {
// //         kdtree->nearestKSearch(cloud->points[i], k, neighbors_all[i], dists);
// //         sizes[i] = (int)neighbors_all[i].size();
// //     }
// //     int max_nn_size= *max_element(sizes.begin(), sizes.end());
// //     vector<int> temp_neighbors_all(max_nn_size * cloud->size());
// //     pcl::gpu::PtrStep<int> ps(&temp_neighbors_all[0], max_nn_size * pcl::gpu::PtrStep<int>::elem_size);
// //     for (size_t i = 0; i < cloud->size(); ++i)
// //         std::copy(neighbors_all[i].begin(), neighbors_all[i].end(), ps.ptr(i));

// //     pcl::gpu::NeighborIndices indices;
// //     gpuCloud.upload(cloud->points);
// //     indices.upload(temp_neighbors_all, sizes, max_nn_size);

// //     pcl::gpu::NormalEstimation::Normals normals;
// //     pcl::gpu::NormalEstimation::computeNormals(gpuCloud, indices, normals);
// //     pcl::gpu::NormalEstimation::flipNormalTowardsViewpoint(gpuCloud, 0.f, 0.f, 0.f, normals);

// //     // vector<pcl::PointXYZ> downloaded;
// //     // normals.download(downloaded);
// //     // tempCV.resize(downloaded.size());
// //     // for (size_t i = 0; i < downloaded.size(); ++i)
// //     // {

// //     //     tempCV[i].index = i;
// //     //     tempCV[i].curvature = downloaded[i].data[3];
// //     // }
// //     return true;
// // }

// using namespace std;
// typedef pcl::PointCloud<pcl::PointXYZ> pointcloud;


// int main(int argc, char *argv[])
// {
//     clock_t start, end; //long
//     start = clock();    //开始时间
//     //======【1】 读取点云文件　填充点云对象======
//     pointcloud::Ptr source(new pointcloud);
//     pointcloud::Ptr target(new pointcloud);
//         cout << "source point size : " << source->size() << endl;
//     cout << "target point size : " << target->size() << endl;
//     getModelCurvatures(source,10);
// }
// int main2(int argc, char *argv[])
// {
//     clock_t start, end; //long
//     start = clock();    //开始时间
//     //======【1】 读取点云文件　填充点云对象======
//     pointcloud::Ptr source(new pointcloud);
//     pointcloud::Ptr target(new pointcloud);
//     // pcl::io::loadPCDFile("1.pcd", *source);
//     // pcl::io::loadPCDFile("2.pcd", *target);
//     pcl::io::loadPLYFile<pcl::PointXYZ>("./p1.ply", *source);
//     pcl::io::loadPLYFile<pcl::PointXYZ>("./p2.ply", *target);
//     cout << "source point size : " << source->size() << endl;
//     cout << "target point size : " << target->size() << endl;

//     // =====【2】计算法线========创建法线估计类================
//     // 源点云
//     pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//     ne.setInputCloud(source);
//     // 添加搜索算法 kdtree search
//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
//     ne.setSearchMethod(tree); //设置近邻搜索算法
//     // 输出点云 带有法线描述
//     pcl::PointCloud<pcl::Normal>::Ptr source_normals_ptr(new pcl::PointCloud<pcl::Normal>);
//     pcl::PointCloud<pcl::Normal> &source_normals = *source_normals_ptr;
//     ne.setKSearch(5); //搜索临近点10
//     // 计算表面法线特征
//     ne.compute(source_normals);
//     cout << "目标点云" << endl;
//     // 目标点云
//     ne.setInputCloud(target);
//     pcl::PointCloud<pcl::Normal>::Ptr target_normals_ptr(new pcl::PointCloud<pcl::Normal>);
//     pcl::PointCloud<pcl::Normal> &target_normals = *target_normals_ptr;
//     ne.setKSearch(10);
//     ne.compute(target_normals);

//     //=======【3】创建SHOT估计对象shot，并将输入点云数据集cloud和法线normals传递给它=================
//     pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot; // shot特征估计其器
//     shot.setInputCloud(source);
//     shot.setInputNormals(source_normals_ptr);
//     //pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>());
//     //shot.setSearchMethod(tree2);//设置近邻搜索算法
//     pcl::PointCloud<pcl::SHOT352>::Ptr source_shot(new pcl::PointCloud<pcl::SHOT352>());
//     //注意：此处使用的半径必须要大于估计表面法线时使用的半径!!!
//     // lucy 02/30/100 、05/70、1/50、 2/40
//     //s1 500
//     shot.setRadiusSearch(300);
//     shot.setNumberOfThreads(11);
//     //shot.setSearchSurface(source);
//     shot.compute(*source_shot); //计算shot特征值

//     shot.setInputCloud(target);
//     shot.setInputNormals(target_normals_ptr);
//     pcl::PointCloud<pcl::SHOT352>::Ptr target_shot(new pcl::PointCloud<pcl::SHOT352>());
//     shot.setRadiusSearch(300);
//     shot.setNumberOfThreads(11);
//     //shot.setSearchSurface(target);
//     shot.compute(*target_shot); //计算shot特征值
//     end = clock();
//     cout << "calculate time is " << float(end - start) / CLOCKS_PER_SEC << endl;

//     cout << "source SHOT feature size : " << source_shot->points.size() << endl;

//     ofstream write;
//     write.open("1.txt");
//     for (int i = 0; i < source_shot->size(); i++)
//     {
//         pcl::SHOT352 descriptor = source_shot->points[i];
//         write << descriptor;
//         write << "/n";
//     }
//     write.close();

//     ofstream write1;
//     write1.open("2.txt");
//     for (int i = 0; i < target_shot->size(); i++)
//     {
//         pcl::SHOT352 descriptor = target_shot->points[i];
//         write1 << descriptor;
//         write1 << "/n";
//     }
//     write1.close();
//     system("pause");
//     return 0;
// }
