#include <pcl/io/pcd_io.h>

#include <pcl/features/rops_estimation.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/cloud_viewer.h>

int main(int argc, char **argv)
{

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	if (pcl::io::loadPCDFile("../data/points.pcd", *cloud) == -1)
	{
		return -1;
	}

	//加载点云中需要计算ROPS特征的关键点标号indices，若对点云中每个点均需要计算ROPS则可以注释
	pcl::PointIndicesPtr indices(new pcl::PointIndices());
	std::ifstream indices_file;
	indices_file.open("../data/indices.txt", std::ifstream::in);

	for (std::string line; std::getline(indices_file, line);)
	{
		std::istringstream in(line);
		unsigned int index = 0;
		in >> index;
		indices->indices.push_back(index - 1);
	}
	indices_file.close();
	std::cout <<indices->indices.size()<<std::endl;

	//加载Mesh的面元信息triangles
	//若输入为纯点云，则需将下述代码替换为三角化程序
	std::vector<pcl::Vertices> triangles; //通过基本存储索引数组来描述多边形网格中的一组顶点
	std::ifstream triangles_file;
	triangles_file.open("../data/triangles.txt", std::ifstream::in);
	for (std::string line; std::getline(triangles_file, line);)
	{
		pcl::Vertices triangle;
		std::istringstream in(line);
		unsigned int vertex = 0;
		in >> vertex;
		triangle.vertices.push_back(vertex - 1);
		in >> vertex;
		triangle.vertices.push_back(vertex - 1);
		in >> vertex;
		triangle.vertices.push_back(vertex - 1);
		triangles.push_back(triangle);
	}

	float support_radius = 0.0285f; //设置局部邻域大小的支撑半径，该数值越大，包含表面信息越多，同时受遮挡和背景干扰的影响越大。
	unsigned int number_of_partition_bins = 5;
	unsigned int number_of_rorations = 3;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	//实例化
	pcl::ROPSEstimation<pcl::PointXYZ, pcl::Histogram<135> > feature_estimator;
	feature_estimator.setInputCloud(cloud);
	feature_estimator.setSearchMethod(tree);
	feature_estimator.setSearchSurface(cloud);
	feature_estimator.setIndices(indices);
	feature_estimator.setRadiusSearch(support_radius);
	feature_estimator.setTriangles(triangles);
	feature_estimator.setNumberOfRotations(number_of_rorations);
	feature_estimator.setNumberOfPartitionBins(number_of_partition_bins);
	feature_estimator.setSupportRadius(support_radius);
	pcl::PointCloud<pcl::Histogram<135> >::Ptr histograms(new pcl::PointCloud<pcl::Histogram<135> >);
	feature_estimator.compute(*histograms);

	//*******************可视化*******************
	pcl::visualization::PCLPlotter plotter;
	// pcl::visualization::CloudViewer view("Simple Cloud Viewer"); //直接创造一个显示窗口
	// view.showCloud(cloud); //再这个窗口显示点云
	std::cout << histograms->points.size()<<std::endl;
	std::string title = "PCL - ROPS特征描述子";
	plotter.setWindowName(title);
	plotter.setShowLegend(true);
	plotter.addFeatureHistogram(*histograms, 135, "ROPS"); //设置的很坐标长度，该值越大，则显示的越细致
	plotter.spin();

	/*pcl::visualization::PCLPlotter::addFeatureHistogram (
    const pcl::PointCloud<PointT> &cloud, 
    const std::string &field_name,
    const int index, 
    const std::string &id, int win_width, int win_height),但本例中的pcl::Histogram <135>是没用POINT_CLOUD_REGISTER_POINT_STRUCT注册过，即没有field_name，简单的显示方式就是把想显示
	的点对应的特征向量，作为单独一个新的点云来对待，就可以显示*/

	// while (1)
	// {
	// 	/* code */
	// }

	return 0;
}
