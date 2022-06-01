#pragma once
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <memory>
#include "gloabMap.hpp"
#include <unordered_set>
class gloabMap;

using namespace cv;
using namespace std;

#define ADDRESS_NUMBER 0X1000000
#pragma pack(push, 1)
union Voxel
{
  uint8_t byte[8];
  struct
  {
    uint8_t weight;
    uint8_t r;
    uint8_t g;
    uint8_t b;
    float tsdfval = 1.0f;
  };
};

union u32_4byte
{
  uint32_t u32 = 0x00000000;
  int8_t byte4[4];
};
struct box32
{
  uint32_t index = 0xffffffff;
  // int16_t x;
  // int16_t y;
  // int16_t z;

  // box32(uint32_t &_index)
  // {
  //   index = _index;
  // }
  // void tobuff()
  // {
  //   // uint8_t *pbuf=new
  // }
  union Voxel pVoxel[32 * 32 * 32];

  bool save_binary(string filename) //单个写入
  {
    // int channl = image.channels();
    // // cout << image << endl;
    // int rows = image.rows;
    // int cols = image.cols;
    // uint8_t type = sizeof(float);
    fstream file(filename, ios::out | ios::binary | ios::app); // |
    // file.write(reinterpret_cast<char *>(&channl), 1);
    // file.write(reinterpret_cast<char *>(&type), 1);
    // file.write(reinterpret_cast<char *>(&cols), 4);
    // file.write(reinterpret_cast<char *>(&rows), 4);
    // printf("SAVE:cols=%d,type=%d,rows=%d,channels=%d\n", cols, type, rows, channl);
    file.write(reinterpret_cast<char *>(&index), sizeof(box32));
    file.close();
    return true;
  }
  int read(string savename, int _index)
  {
    fstream file(savename, ios::in | ios::binary);
    if (!file)
    {
      cout << "Error opening file." << savename << endl;
      return 0;
    }
    // file.read(reinterpret_cast<char *>(&data), sizeof(struct _data));
    file.seekg(sizeof(box32) * _index);
    // cout << "sizeof(box32)*index:" << sizeof(box32) * index << "  " << file.gcount() << endl;
    file.read(reinterpret_cast<char *>(&index), sizeof(box32));
    // file.read(reinterpret_cast<char *>(pVoxel->byte), sizeof(Voxel)*32*32*32);
    // std::cout << "[INFO] "<<_index<< std::endl;
    // for (int i = 0; i < data.pic_number; i++)
    // {
    //     cout << timestamps[i] << endl;
    // }
    // delete pdata;
    file.close();
    return 0;
  }

  void getcloud(Mat &color, Mat &cloud, cv::Vec3f &tsdfwordpose, cv::Vec4f m_centerf = cv::Vec4f(0, 0, 0, 0))
  {
    for (int8_t pt_grid_z = 0; pt_grid_z < 32; pt_grid_z++)
    {
      for (int8_t pt_grid_y = 0; pt_grid_y < 32; pt_grid_y++)
      {
        for (int8_t pt_grid_x = 0; pt_grid_x < 32; pt_grid_x++)
        {
          int volume_idx = pt_grid_z * 32 * 32 + pt_grid_y * 32 + pt_grid_x;
          union Voxel &voxel = pVoxel[volume_idx];

          if (std::abs(voxel.tsdfval) < 0.05f && voxel.weight > 20)
          // cout << tsdfwordpose << endl;
          // if (pt_grid_x % 5 == 0 && pt_grid_y % 5 == 0)
          {
            cv::Vec3f vec;
            vec[0] = tsdfwordpose[0] + pt_grid_x * 0.01f  + m_centerf[0];
            vec[1] = tsdfwordpose[1] + pt_grid_y * 0.01f  + m_centerf[1];
            vec[2] = tsdfwordpose[2] + pt_grid_z * 0.01f  + m_centerf[2];
            cloud.push_back(vec);

            cv::Vec<uint8_t, 3> vescolor;
            vescolor[0] = voxel.r;
            vescolor[1] = voxel.g;
            vescolor[2] = voxel.b;
            color.push_back(vescolor);
          }
        }
      }
    }
  }
};
union u64_4byte
{
  uint64_t u64 = 0x00;
  struct
  {
    int16_t x;
    int16_t y;
    int16_t z;
    int16_t _rev;
  };
  // struct
  // {
  //   int64_t x : 24;
  //   int64_t y : 24;
  //   int64_t z : 16;
  // };

  // int8_t byte4[4];
};

#pragma pack(pop)

class Warehouse
{
private:
public:
  vector<box32 *> box32s; //以已经分配了的空间,并且在使用
  Warehouse()
  {
    box32s.reserve(512);
  }
  void allocate(int number)
  {
    for (int i = 0; i < number; i++)
    {
      box32s.push_back(new struct box32);
    }
  }
};
class MyTSDF
{
private:
  /* data */
public:
  vector<box32 *> box32s; //以已经分配了的空间,并且在使用
  // std::vector<box32 *> pbox32s(4194304, {NULL});
  vector<box32 *> *pbox32s; //存放所有大体素的地址
  // box32 (*pbox32s);//[ADDRESS_NUMBER]; //存放所有大体素的地址

  unordered_set<uint32_t> pre_box;
  int alsize = 1024;
  // int nowsize = 0;
  int id = 0;
  float *_pval;
  float *_weight;
  uint8_t *_rgb;
  cv::Mat position;
  Vec4s m_center;
  Vec4f m_centerf{0.0, 0.0, 0.0, 1.0f};

  // db *pnewdb;
  MyTSDF(int a, Vec4s);
  void geicurrbox(int i);
  void addresstotsdfworld(union u32_4byte &u32_4, cv::Vec3f &tsdfwordpose);

  // void geicurrbox(int i)
  // {
  //   if ((*pbox32s)[i] == NULL)
  //   {
  //     if (nowsize >= box32s.size())
  //     {
  //       cout << __LINE__ << " " << nowsize << " " << box32s.size() << endl;
  //       assert(0);
  //     }
  //     if (nowsize + 1 >= box32s.size())
  //     {
  //       // cout << __LINE__ << "new memory nowsize:" << nowsize << " bsize " << box32s.size() << endl;
  //       // alsize = alsize * 2;
  //       box32s.reserve(alsize * 2);
  //       cout << "" << box32s.size() << endl;
  //       // box32s.resize(alsize); //resize()既修改capacity大小，也修改size大小。
  //       for (int i = 0; i < alsize; i++)
  //       {
  //         struct box32 *pbox32s_l = new struct box32();
  //         // pbox32s_l->weight = new float[32 * 32 * 32];
  //         assert(pbox32s_l->pVoxel);
  //         box32s.push_back(pbox32s_l);
  //       }
  //       alsize += alsize;
  //       cout << "" << box32s.size() << " " << nowsize << endl;
  //     }
  //     (*pbox32s)[i] = box32s[nowsize];
  //     (*pbox32s)[i]->index = i;
  //     nowsize++;
  //   }
  // }
  void integrate(float *cam_K, float *cam2base, uint8_t *rgb_im, float *depth_im,
                 int im_height, int im_width, float trunc_margin);
  void Voxel2PointCloud(cv::Mat &color, cv::Mat &p);
  bool word2tsdfax(Vec4f &wpoint, Vec4s &tsdfval, gloabMap &gm);
  bool word2tsdfax(Mat &dep, Vec<float, 16> &po, gloabMap &gm);
  void tsdf2wordax(Vec3b &wpoint);
  void print();
  void over(std::shared_ptr<MyTSDF> &mytsdf, cv::Mat &pos);
  static bool Mat_save_by_binary(cv::Mat &image, string filename);
  static bool Mat_read_binary(cv::Mat &img_vec, string filename); // void savefile()
  // {
  //   Mat_save_by_binary(position, "position.bin");
  //   // printf("%ld\n", position.rows);
  // }
};
class dataset
{
public:
  cv::Mat posss;
  dataset()
  {

    // Mat_read_binary(posss, "/home/u16/dataset/img/loop22pose.bin");
  }

  string getdataset(int frame_idx, cv::Mat &_depth, cv::Mat &_rgb_img, string &color_path)
  {
    string depth_path = cv::format("/home/u16/dataset/img/depth/%04d.png", frame_idx);
    _depth = cv::imread(depth_path, 2);
    if (_depth.empty())
    {
      string ss = "ls " + depth_path;
      system(ss.c_str());
      assert(0);
    }

    color_path = cv::format("/home/u16/dataset/img/rgb/%04d.png", frame_idx);

    _rgb_img = cv::imread(color_path);
    if (_rgb_img.empty())
    {
      string ss = "ls " + color_path;
      system(ss.c_str());
      assert(0);
    }
    cv::imshow("_depth", _depth);     // depth_im_file = data_path + "/frame-" + .str() + ".color.jpg";
    cv::imshow("_rgb_img", _rgb_img); // depth_im_file = data_path + "/frame-" + .str() + ".color.jpg";
    cv::waitKey(1);
    return depth_path;
  }
};