#include <iomanip>
#include <opencv2/opencv.hpp>
#include <memory>
#include <map>
// #include <set>
#include <iostream>
#include "skiplist.h"
#define FILE_PATH "./store/dumpFile"
#include "MyTSDF.h"
#include "gloabMap.hpp"

class MyTSDF;

using namespace cv;
using namespace std;


gloabMap::gloabMap()
{
  pskipList = new SkipList<uint64_t, struct Voxel32 *>(6);
};
void gloabMap::addcube(std::shared_ptr<MyTSDF> &spTsdf)
{
  // cout << "mapSet" << mapSet.size() << endl;
  for (int i = 0; i < spTsdf->box32s.size(); i++) // for (auto &it : box32s)
  {
    u32_4byte u324;
    // Voxel32 *p=(spTsdf->box32s)[i];
    u324.u32 = spTsdf->box32s[i]->index; //原cube 序号

    // printf("%x,%x,%x,%x\n", u324.u32, u324.byte4[0], u324.byte4[1], u324.byte4[2]);
    Vec4s wp;                                                        //原cube的世界坐标
    wp[0] = ((int16_t)u324.byte4[0] + (int16_t)spTsdf->m_center[0]); // * 0.32f; // + spTsdf->m_centerf[0];
    wp[1] = ((int16_t)u324.byte4[1] + (int16_t)spTsdf->m_center[1]); // * 0.32f; // + spTsdf->m_centerf[1];
    wp[2] = ((int16_t)u324.byte4[2] + (int16_t)spTsdf->m_center[2]); // * 0.32f; // + spTsdf->m_centerf[2];
    //   Vec4f wps = wp - mytsdf->m_centerf;
    // Vec4s d4s = wp * 3.125f; //相对体素坐标
    u64B4 u64;
    u64.x = wp[0];
    u64.y = wp[1];
    u64.z = wp[2];
    // while (1)
    // {
      int8_t ret = pskipList->insert_element(u64.u64, spTsdf->box32s[i]);
    //   if (ret == 1)
    //   {
    //     struct Voxel32 *pbox;
    //     ret = pskipList->search_element(u64.u64, pbox);
    //     if (ret == true)
    //     {
    //       if (pbox == spTsdf->box32s[i])
    //       {
    //         break;
    //       }
    //       else
    //       {
    //         delete pbox;
    //         pskipList->delete_element(u64.u64);
    //       }
    //     }
    //   }
    //   else
    //   {
    //     break;
    //   }
    // }
  }
}
// void addresstotsdfworld(union u64B4 &u32_4, cv::Vec3f &tsdfwordpose)
// {
//   // static float tsdf_base_xyz[3] = {0, 0, 0};
//   tsdfwordpose[0] = u32_4.x * 0.32f;
//   tsdfwordpose[1] = u32_4.y * 0.32f;
//   tsdfwordpose[2] = u32_4.z * 0.32f; //0//-10.24; //- tsdf_base_xyz[2];
// }
void gloabMap::getcube(std::shared_ptr<MyTSDF> &spTsdf)
{
}
void gloabMap::Voxel2PointCloud(cv::Mat &color, cv::Mat &cloud)
{
  color = Mat();
  cloud = Mat();
  int nodesize = 0;
  for (int i = 0; i <= pskipList->_skip_list_level; i++)
  {
    Node<uint64_t, struct Voxel32 *> *node = pskipList->_header->forward[i];
    // std::cout << "Level " << i << ": ";
    while (node != NULL)
    {
      u64B4 u64;
      u64.u64 = node->get_key();
      Vec3f vec4;
      vec4[0] = u64.x * 0.32;
      vec4[1] = u64.y * 0.32;
      vec4[2] = u64.z * 0.32;

      node->get_value()->getcloud(color, cloud, vec4);
      // std::cout < < < < ":" << << ";";
      node = node->forward[i];
      nodesize++;
    }
  }
  std::cout << "nodesize:" << nodesize << std::endl;

  // assert(mapSet.size());
  // for (auto &it : box32s)
  // for (auto it = mapSet.begin(); it != mapSet.end(); ++it)
  // {
  //   // it->second->index = 0;
  //   if (it->second->index == 0xffffffff)
  //     continue;
  //   assert(it->second->index == it->first);
  //   // {
  //   // cout << it->index << " ";
  //   uint32_t i = it->second->index;
  //   Vec3f tsdfwordpose;

  //   u32_4byte u64;
  //   u64.u32 = i;
  //   addresstotsdfworld(u64, tsdfwordpose);
  //   // printf("0x%016lx,%d,%d,%d,%d\n", u64.u64, u64.x, u64.y, u64.z,sizeof( u64.u64));

  //   //   // printf("%d,%d\.n", u32_4.u32,i );
  //   for (int8_t pt_grid_z = 0; pt_grid_z < 32; pt_grid_z++)
  //   {
  //     for (int8_t pt_grid_y = 0; pt_grid_y < 32; pt_grid_y++)
  //     {
  //       for (int8_t pt_grid_x = 0; pt_grid_x < 32; pt_grid_x++)
  //       {
  //         int volume_idx = pt_grid_z * 32 * 32 + pt_grid_y * 32 + pt_grid_x;
  //         union Voxel &voxel = it->second->pVoxel[volume_idx];

  //         if (std::abs(voxel.tsdfval) < 0.2f && voxel.weight > 0)
  //         // cout << tsdfwordpose << endl;
  //         // if (pt_grid_x % 5 == 0 && pt_grid_y % 5 == 0)
  //         {
  //           cv::Vec3f vec;
  //           vec[0] = tsdfwordpose[0] + pt_grid_x * 0.01f;
  //           vec[1] = tsdfwordpose[1] + pt_grid_y * 0.01f;
  //           vec[2] = tsdfwordpose[2] + pt_grid_z * 0.01f;
  //           cloud.push_back(vec);

  //           cv::Vec<uint8_t, 3> vescolor;
  //           vescolor[0] = voxel.r;
  //           vescolor[1] = voxel.g;
  //           vescolor[2] = voxel.b;
  //           color.push_back(vescolor);
  //         }
  //       }
  //     }
  //   }
  // }
  // }
  // cout << cloud << endl;
  // return cloud;
}
