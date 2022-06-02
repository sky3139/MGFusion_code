#include <iomanip>
#include <opencv2/opencv.hpp>
#include <memory>
#include "gloabMap.hpp"
#include "MyTSDF.h"

using namespace cv;
using namespace std;

#define ADDRESS_NUMBER 0X1000000

// db *pnewdb;
MyTSDF::MyTSDF(int a, Vec4s _center = Vec4s(0, 0, 0, 1))
{
  alsize = 0;
  id = a;
  pbox32s = new vector<Voxel32 *>(ADDRESS_NUMBER, NULL);
  m_center = _center;
  m_centerf = _center * 0.32f;
  //***********************db
  // db_init(pnewdb, "dataset.bin");
  // unsigned char *buf = new unsigned char[1024];
  // unsigned char *key = new unsigned char[4];
  // for (size_t i = 0; i < 1024; i++)
  // {
  //   key = random_str();
  //   putkeyvalue(pnewdb, key, key);
  // }
  // char *value = getvalue(pnewdb, key);
  // puts((char *)key);
  // puts(value);
  //***********************db
}

void MyTSDF::geicurrbox(int i)
{
  if ((*pbox32s)[i] == NULL)
  {
    struct Voxel32 *pbox32s_l = new struct Voxel32();
    pbox32s_l->index = i;
    box32s.push_back(pbox32s_l);
    (*pbox32s)[i] = pbox32s_l;
  }
}

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
//         struct Voxel32 *pbox32s_l = new struct Voxel32();
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

void MyTSDF::addresstotsdfworld(union u32_4byte &u32_4, cv::Vec3f &tsdfwordpose)
{
  static float tsdf_base_xyz[3] = {0, 0, 0};
  tsdfwordpose[0] = u32_4.byte4[0] * 0.32f - tsdf_base_xyz[0];
  tsdfwordpose[1] = u32_4.byte4[1] * 0.32f - tsdf_base_xyz[1];
  tsdfwordpose[2] = u32_4.byte4[2] * 0.32f - tsdf_base_xyz[2];
}
void MyTSDF::integrate(float *cam_K, float *cam2base, uint8_t *rgb_im, float *depth_im,
                       int im_height, int im_width, float trunc_margin)
{
  for (auto it = pre_box.begin(); it != pre_box.end(); ++it)
  {
    uint32_t i = *it;
    u32_4byte u32_4;
    u32_4.u32 = i;
    Vec3f tsdfwordpose;

    addresstotsdfworld(u32_4, tsdfwordpose);

    if (u32_4.u32 >= ADDRESS_NUMBER) //22 bit
    {
      printf("%x,%d,%d,%d\n", u32_4.u32, u32_4.byte4[2], u32_4.u32, 256 * 256 * 64);
      assert(0);
    }
    geicurrbox(i);

    // if ((*pbox32s)[i] == NULL)
    // {

    //   if (nowsize >= box32s.size())
    //   {
    //     cout << __LINE__ << " " << nowsize << " " << box32s.size() << endl;
    //     assert(0);
    //   }

    //   if (nowsize + 1 >= box32s.size())
    //   {
    //     cout << __LINE__ << "new memory nowsize:" << nowsize << " bsize " << box32s.size() << endl;

    //     // alsize = alsize * 2;
    //     box32s.reserve(alsize * 2);
    //     cout << "" << box32s.size() << endl;
    //     // box32s.resize(alsize); //resize()既修改capacity大小，也修改size大小。
    //     for (int i = 0; i < alsize; i++)
    //     {
    //       struct Voxel32 *pbox32s_l = new struct Voxel32();
    //       // pbox32s_l->weight = new float[32 * 32 * 32];
    //       assert(pbox32s_l->pVoxel);
    //       box32s.push_back(pbox32s_l);
    //     }
    //     alsize += alsize;
    //     cout << "" << box32s.size() << " " << nowsize << endl;
    //   }
    //   (*pbox32s)[i] = box32s[nowsize];
    //   (*pbox32s)[i]->index = i;
    //   nowsize++;
    // }
    // #pragma omp parallel
    // #pragma omp for
    for (int8_t pt_grid_z = 0; pt_grid_z < 32; pt_grid_z++)
    {
      for (int8_t pt_grid_y = 0; pt_grid_y < 32; pt_grid_y++)
      {
        for (int8_t pt_grid_x = 0; pt_grid_x < 32; pt_grid_x++)
        {

          float pt_base_x = tsdfwordpose[0] + pt_grid_x * 0.01f + m_centerf[0];
          float pt_base_y = tsdfwordpose[1] + pt_grid_y * 0.01f + m_centerf[1];
          float pt_base_z = tsdfwordpose[2] + pt_grid_z * 0.01f + m_centerf[2];

          // Convert from base frame camera coordinates to current frame camera coordinates
          float tmp_pt[3] = {0};
          tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
          tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
          tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
          float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
          float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
          float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

          if (pt_cam_z <= 0)
            continue;

          int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
          int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
          if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
            continue;

          float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];

          if (depth_val <= 0 || depth_val > 6)
            continue;

          float diff = depth_val - pt_cam_z;
          // std::cout << depth_val<<" "<<pt_cam_z<<" \n";// << " ";

          if (diff <= -trunc_margin)
            continue;

          // Integrate
          int volume_idx = pt_grid_z * 32 * 32 + pt_grid_y * 32 + pt_grid_x;
          union Voxel &voxel = (*pbox32s)[i]->pVoxel[volume_idx];

          int dist = fmin(1.0f, diff / trunc_margin);
          auto weight_old = voxel.weight;
          auto weight_new = std::min(weight_old + 1, 255);
          {
            voxel.weight = weight_new;
            voxel.tsdfval = (voxel.tsdfval * (float)weight_old + dist) / (float)weight_new;
          }

          // Integrate color

          uint8_t rgb_val[3];
          rgb_val[0] = rgb_im[pt_pix_y * im_width + pt_pix_x];
          rgb_val[1] = rgb_im[pt_pix_y * im_width + pt_pix_x + 640 * 480];
          rgb_val[2] = rgb_im[pt_pix_y * im_width + pt_pix_x + 640 * 480 * 2];

          int mval = (voxel.r * weight_old + rgb_val[0]) / weight_new;
          voxel.r = mval > 255 ? 255 : mval;
          mval = (voxel.g * weight_old + rgb_val[1]) / weight_new;
          voxel.g = mval > 255 ? 255 : mval;
          mval = (voxel.b * weight_old + rgb_val[2]) / weight_new;
          voxel.b = mval > 255 ? 255 : mval;
        }
      }
    }
    // printf("%d,%d,%d,%x\n", x, y, z, i);
    // cout << "tsdfwordpose" << tsdfwordpose << endl;
  }
  pre_box.clear();
  // print();
}
void MyTSDF::Voxel2PointCloud(cv::Mat &color, cv::Mat &cloud)
{
  color = Mat();
  cloud = Mat();
  assert(box32s.size());
  for (auto &it : box32s)
  {
    if (it->index != 0xffffffff)
    {
      // cout << it->index << " ";
      uint32_t i = it->index;
      Vec3f tsdfwordpose;

      u32_4byte u32_4;
      u32_4.u32 = i;
      addresstotsdfworld(u32_4, tsdfwordpose);

      ((*pbox32s)[i])->getcloud(color, cloud, tsdfwordpose, m_centerf);

      // printf("%d,%d\.n", u32_4.u32,i );
      // for (int8_t pt_grid_z = 0; pt_grid_z < 32; pt_grid_z++)
      // {
      //   for (int8_t pt_grid_y = 0; pt_grid_y < 32; pt_grid_y++)
      //   {
      //     for (int8_t pt_grid_x = 0; pt_grid_x < 32; pt_grid_x++)
      //     {
      //       int volume_idx = pt_grid_z * 32 * 32 + pt_grid_y * 32 + pt_grid_x;
      //       union Voxel &voxel = (*pbox32s)[i]->pVoxel[volume_idx];

      //       if (std::abs(voxel.tsdfval) < 0.2f && voxel.weight > 0)
      //       // cout << tsdfwordpose << endl;
      //       // if (pt_grid_x % 5 == 0 && pt_grid_y % 5 == 0)
      //       {
      //         cv::Vec3f vec;
      //         vec[0] = tsdfwordpose[0] + pt_grid_x * 0.01f - 0.16 + m_centerf[0];
      //         vec[1] = tsdfwordpose[1] + pt_grid_y * 0.01f - 0.16 + m_centerf[1];
      //         vec[2] = tsdfwordpose[2] + pt_grid_z * 0.01f - 0.16 + m_centerf[2];
      //         cloud.push_back(vec);

      //         cv::Vec<uint8_t, 3> vescolor;
      //         vescolor[0] = voxel.r;
      //         vescolor[1] = voxel.g;
      //         vescolor[2] = voxel.b;
      //         color.push_back(vescolor);
      //       }
      //     }
      //   }
      // }
    }
  }
}

bool MyTSDF::word2tsdfax(Vec4f &wpoint, Vec4s &tsdfval, gloabMap &gm)
{
  // if (wpoint[0] - m_center[0] * 0.32 > 10.24)
  // {
  //   tsdfvalbak = wpoint * 3.125f; // / 0.32;
  //   cout << tsdfvalbak << endl;
  //   // assert(wpoint[0] < 10.24); //10.24=2<<6 /2
  //   return false;
  // }
  // assert(wpoint[1] < 10.24); //10.24=2<<6 /2
  // assert(wpoint[2] > 0);     //10.24=2<<6 /2
  // wpoint[0] = wpoint[0];
  // wpoint[1] = wpoint[1];
  // wpoint[2] = wpoint[2] + 10.24; //保证Z为正
  // wpoint
  tsdfval = wpoint * 3.125f; // / 0.32;
  // printf("tsdfval%x,%x,%x\n", tsdfval[0], tsdfval[1], tsdfval[2]);

  u32_4byte u32_4;
  u32_4.byte4[0] = (tsdfval[0]);
  u32_4.byte4[1] = (tsdfval[1]);
  u32_4.byte4[2] = (tsdfval[2]);
  u32_4.byte4[3] = 0;

  Vec4s wp;                                                       //原cube的世界坐标
  wp[0] = ((int16_t)u32_4.byte4[0] + (int16_t)this->m_center[0]); // * 0.32f; // + spTsdf->m_centerf[0];
  wp[1] = ((int16_t)u32_4.byte4[1] + (int16_t)this->m_center[1]); // * 0.32f; // + spTsdf->m_centerf[1];
  wp[2] = ((int16_t)u32_4.byte4[2] + (int16_t)this->m_center[2]); // * 0.32f; // + spTsdf->m_centerf[2];
  //   Vec4f wps = wp - mytsdf->m_centerf;
  // Vec4s d4s = wp * 3.125f; //相对体素坐标
  u64B4 u64;
  u64.x = wp[0];
  u64.y = wp[1];
  u64.z = wp[2];

  Voxel32 *pb;
  bool ret = gm.pskipList->search_element(u64.u64, pb);
  if (ret == true)
  {
    (*pbox32s)[u32_4.u32] = pb;
  }
  // printf("%x,%x,%x,%x\n", tsdfval[0], tsdfval[1], tsdfval[2], u32_4.u32);

  // printf("%x,%x,%x,%x\n", ca, tsdfval[1], tsdfval[2], u32_4.u32);
  pre_box.insert(u32_4.u32);
  return true;
  // std::cout << "tsdfval:" << tsdfval << endl;
}
bool MyTSDF::word2tsdfax(Mat &dep, Vec<float, 16> &po, gloabMap &gm)
{
  Mat_<float> cam_pose(4, 4, &po.val[0]);
  // cout << "dep" << dep.t() << endl;

  Mat dep1dm = dep.reshape(1);
  Mat wdpoint = dep1dm * cam_pose.t();

  Mat cen = Mat_<Vec4f>(wdpoint.rows, 1, m_centerf);
  cen = cen.reshape(1);
  Mat dst = wdpoint - cen;
  Mat o3434 = dst * 3.125f;
  Mat asd;
  // o3434.convertTo(asd, CV_16S);
  for (int i = 0; i < o3434.rows; i++)
  {
    Vec4f src_ct = o3434.at<Vec4f>(i, 0);
    Vec4s src_ct4s; // = o3434.at<Vec4s>(i, 0);
    src_ct4s[0] = floor(src_ct[0]);
    src_ct4s[1] = floor(src_ct[1]);
    src_ct4s[2] = floor(src_ct[2]);
    // cout <<src_ct4s << endl;

    // Vec4s src_ct = o3434.at<Vec4s>(i, 0);
    u32_4byte u32_4;
    u32_4.byte4[0] = (src_ct4s[0]);
    u32_4.byte4[1] = (src_ct4s[1]);
    u32_4.byte4[2] = (src_ct4s[2]);
    // u32_4.byte4[3] = 0;
    // // printf("%x,%x,%x,%x\n", tsdfval[0], tsdfval[1], tsdfval[2], u32_4.u32);
    // // printf("%x,%x,%x,%x\n", ca, tsdfval[1], tsdfval[2], u32_4.u32);
    pre_box.insert(u32_4.u32);
  }

  return true;
}
void MyTSDF::tsdf2wordax(Vec3b &wpoint)
{
  Vec3f tsdfval = wpoint * 0.32; // * 0.32;

  // uint32_t i0 = ((0xff & tsdfval[2]) << 16 | (0xff & tsdfval[1]) << 8 | (0xff & tsdfval[0]));
  // printf("%x,%x,%x,%x\n", tsdfval[0], tsdfval[1], tsdfval[2], i0);
  // pre_box.insert(i0);
  // std::cout << "tsdfval:" << tsdfval << endl;
}
void MyTSDF::print()
{
  // std::cout << "nowsize:" << nowsize << " box32s.size():" << box32s.size() << endl;
}

void MyTSDF::over(std::shared_ptr<MyTSDF> &mytsdf, cv::Mat &pos)
{

  // for (int i = 0;  box32s.size()>0; i++) // for (auto &it : box32s)
  // {
  // bo  box32s.pop_back();
  //   u32_4byte u324;
  //   u324.u32 = box32s[i]->index; //原cube 序号
  //   Vec4f wp;                    //原cube的世界坐标
  //   wp[0] = (u324.byte4[0] + 1) * 0.32 + m_centerf[0];
  //   wp[1] = (u324.byte4[1] + 1) * 0.32 + m_centerf[1];
  //   wp[2] = (u324.byte4[2] + 1) * 0.32 + m_centerf[2];
  //   Vec4f wps = wp - mytsdf->m_centerf;
  //   Vec4s d4s = wps * 3.125f; //相对体素坐标

  //   if (std::fabs(wps[0]) > 4.0 || std::fabs(wps[1]) > 4.0 || std::fabs(wps[2]) > 4.0)
  //   {
  //     pos.push_back(d4s);
  //     box32s[i]->save_binary("box.bin");
  //     delete box32s[i];
  //     box32s[i] = nullptr;
  //     // printf("%ld\n", position.rows);
  //     continue;
  //   }

  //   // putkeyvalue(pnewdb, (unsigned char *)&u324.byte4[0], "key");
  //   // printf("");
  //   // Vec4f ct2;

  //   // ct2[0] = src_ct[0] - mytsdf->m_centerf[0];
  //   // ct2[1] = src_ct[1] - mytsdf->m_centerf[1];
  //   // ct2[2] = src_ct[2] - mytsdf->m_centerf[2];
  //   //   Vec4s d4s = ct2 * 3.125;
  //   //   // bool ret = currtsdf->word2tsdfax(ct, val);

  //   u32_4byte uew324;
  //   uew324.byte4[0] = (int8_t)d4s[0];
  //   uew324.byte4[1] = (int8_t)d4s[1];
  //   uew324.byte4[2] = (int8_t)d4s[2];
  //   box32s[i]->index = uew324.u32;

  //   // cout << "val:" << mytsdf->box32s.size() << " " << box32s[i]->index << endl;
  //   // printf("%x,%x,%x,%x\n", 0, uew324.byte4[0], uew324.byte4[1], uew324.byte4[2]);
  //   // cout << uew324.u32 << endl;
  //   // printf("%x,%ld,%d,%x\n", 0, mytsdf->pbox32s->size(), box32s[i]->index, uew324.u32);
  //   //   // cout << d4s << "  1 " << newtsdf->pbox32s->size() << " " << uew324.u32 << endl;
  //   //   it->index = uew324.u32;
  //   mytsdf->box32s.push_back(box32s[i]);
  //   // mytsdf->nowsize++;
  //   (*(mytsdf->pbox32s))[uew324.u32] = box32s[i];
  //   //   // cout << d4s << " 2 " << newtsdf->pbox32s->size() << " " << uew324.u32 << endl;
  //   // break;
  //   //   //
  // }

  // // cout << "over    " << nowsize << endl;
  // vector<Voxel32 *> p;
  // pbox32s->swap(p);
  // cout << "ok" << endl;
  // pbox32s->clear(); //清空元素，但不回收空间
}
bool MyTSDF::Mat_save_by_binary(cv::Mat &image, string filename) //单个写入
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
  file.write(reinterpret_cast<char *>(image.data), em_size * cols * rows);
  file.close();
  return true;
}

bool MyTSDF::Mat_read_binary(cv::Mat &img_vec, string filename) //整体读出
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
  printf("READ:cols=%d,type=%d,em_size=%d,rows=%d,channels=%d\n", cols, type, em_size, rows, channl);
  img_vec = cv::Mat(rows, cols, type);
  fin.read((char *)&img_vec.data[0], rows * cols * em_size);
  fin.close();
  return true;
}
// void savefile()
// {
//   Mat_save_by_binary(position, "position.bin");
//   // printf("%ld\n", position.rows);
// }
