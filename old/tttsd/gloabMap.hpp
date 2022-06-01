#pragma once
#ifndef GLOABMAP_H__
#define GLOABMAP_H__
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <memory>
#include <map>
// #include <set>
#include <iostream>
#include "skiplist.h"
#define FILE_PATH "./store/dumpFile"
#include "MyTSDF.h"

using namespace cv;
using namespace std;
class MyTSDF;
class gloabMap
{
public:
  SkipList<uint64_t, struct box32 *> *pskipList;

  gloabMap();
  void addcube(std::shared_ptr<MyTSDF> &spTsdf);

  void getcube(std::shared_ptr<MyTSDF> &spTsdf);
  void Voxel2PointCloud(cv::Mat &color, cv::Mat &cloud);
};
#endif