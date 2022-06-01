#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "dviz.hpp"
#include <iterator>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <unordered_set>
#include <iostream>
#include "gloabMap.hpp"

#include "MyTSDF.h"
using namespace cv;
using namespace std;
static float tsdf_base_xyz[3] = {0, 0, 0};
void KeybdCallback(const viz::KeyboardEvent &keyEvent, void *val)
{
  int *pKey = ((int *)val);

  if (keyEvent.action == viz::KeyboardEvent::KEY_DOWN)
  {
    if (keyEvent.code == 'a')
    {
      *pKey = 1;
      // MyTSDF::Mat_save_by_binary(pKey->position, "position.bin");

      // pKey->savefile();
      // printf("position 1\n");
    }
    if (keyEvent.code == 's')
    {
      *pKey = 2;
    }
    if (keyEvent.code == 'd')
    {
      *pKey = 3;
      printf("Key 3\n");
    }
  }
}
int main(int argc, char *argv[])
{
  // system("rm box.bin posindex.bin");
  std::string cam_K_file = "../data/camera-intrinsics.txt";
  int base_frame_idx = 1 - 1;
  int first_frame_idx = 1500;
  int num_frames = 4000;
  float cam_K[3 * 3];
  float base2world[4 * 4];
  float cam2base[4 * 4];
  float cam2world[4 * 4];
  int im_width = 640;
  int im_height = 480;
  float depth_im[im_height * im_width];
  uint8_t RGB_im[im_height * im_width * 3];
  float voxel_size = 0.01f;
  float trunc_margin = voxel_size * 5;
  //读取相机内参
  std::vector<float> cam_K_vec = LoadMatrixFromFile(cam_K_file, 3, 3);
  std::copy(cam_K_vec.begin(), cam_K_vec.end(), cam_K);
  // 读取第一帧的位姿
  cv::Mat posss;
  Mat_read_binary(posss, "/home/u16/dataset/img/loop22pose.bin");
  // cout << posss << endl;
  Vec<float, 16> po = posss.at<Vec<float, 16>>(first_frame_idx, 0);
  for (int i = 0; i < 16; i++)
  {
    base2world[i] = po.val[i];
  }
  Affine3f base(po.val);
  // 位姿逆矩阵
  float base2world_inv[16] = {0};
  invert_matrix(base2world, base2world_inv);
  // 初始化格子
  Affine3f a3f(cv::Mat_<float>::eye(Size(3, 3)), cv::Vec3f(0.0, 0.0, 0.5));
  char *pname = new char[256];
  int idx = 1;
  int m_keyv = 0;
  cv::viz::Viz3d window("window");
  // cout << depth << endl;
  //显示坐标系
  window.showWidget("Coordinate", viz::WCoordinateSystem());
  //创建一个储存point cloud的图片
  char *name = new char[256];
  int i = 11;

  // viz::WGrid wg1(Vec3d(0, 0, 1), Vec3d(1, 0, 0), Vec3d(0, 0, 1), Vec2i::all(17), Vec2d::all(voxel_size * 32));
  // window.showWidget("asd1", wg1);
  // cv::viz::WCube cube(Vec3d(-10.24, -10.24, -10.24), Vec3d(10.24, 10.24, 10.24));
  // window.showWidget("tsdfmax", cube);
  // window.showWidget("mpc", wpl, a3f); //, Affine3f(base2world)

  std::shared_ptr<MyTSDF> currtsdf(new MyTSDF(0, Vec4s(0, 0, 0, 0)));

  gloabMap gm;
  cv::Affine3f cam2wd = cv::Affine3f::Identity();
  // std::shared_ptr<MyTSDF> currtsdf = f1;
  vector<std::shared_ptr<MyTSDF>> old_tsdfs;
  int keyval = 0;
  const cv::viz::Color colors[] = {cv::viz::Color::white(), cv::viz::Color::yellow(), cv::viz::Color::blue()};
  int color_id = 0;
  // std::shared_ptr<MyTSDF> newtsdf = std::make_shared<MyTSDF>(Vec4s(32,3,48,0));
  window.registerKeyboardCallback(KeybdCallback, &keyval);
  Mat pos;
  // 重建 TSDF voxel grid
  int now_id = 0;
  dataset dt;
  Mat allclone[2];
  for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; frame_idx += 1, now_id++)
  {
    cv::viz::WCloudCollection cloud;
    Vec<float, 16> src = posss.at<Vec<float, 16>>(frame_idx, 0);
    // cout << src << endl;
    src.val[3] -= base2world[3];
    src.val[7] -= base2world[7];
    src.val[11] -= base2world[11];
    po = src;
    // src.val[3] -= currtsdf->m_centerf[0];
    // src.val[7] -= currtsdf->m_centerf[1];
    // src.val[11] -= currtsdf->m_centerf[2];
    cv::Affine3f pose(&po.val[0]);
    cam2wd = pose; //cam2wd.concatenate(pose);
    string color_path;
    Mat _depth, _rgb_img;
    string depth_path = dt.getdataset(frame_idx, _depth, _rgb_img, color_path);
    ReadDepth(depth_path, im_height, im_width, depth_im);
    Mat dep = getCload(_depth);
    ReadRGB(color_path, im_height, im_width, RGB_im);

    currtsdf->word2tsdfax(dep, po, gm);
    // cout << asd << endl;
    // // printMAT(outtt);
    // int i = 0;
    Vec4s val;
    // cout << __LINE__ << " inte:" << 0 << endl;
    currtsdf->integrate(cam_K, &po.val[0], RGB_im, depth_im, im_height, im_width, trunc_margin);
    // cout << __LINE__ << " inte:" << 1 << endl;

    cv::Mat color;
    Mat points;

    // for (int i = 0; i < old_tsdfs.size(); i++)
    // {
    //   old_tsdfs[i]->Voxel2PointCloud(color, points);
    //   assert(points.rows);
    //   assert(color.rows);
    //   cloud.addCloud(points, color); //,, pose,, Affine3f(cam2base),cv::viz::Color::yellow()
    // }

    // for (int i = 0; i < gm.mapSet.size(); i++)
    // {
    //   points = gm.Voxel2PointCloud(color);
    //   assert(points.rows);
    //   assert(color.rows);
    //   // cv::viz::WCloud wc(points, color);
    //   cloud.addCloud(points, color); //,, pose,, Affine3f(cam2base), colors[color_id%3]
    //   color_id++;
    //   color = Mat();
    //   break;
    // }

    currtsdf->Voxel2PointCloud(color, points);
    if (points.rows > 0)
    {
      cloud.addCloud(points, color); //,, pose,, Affine3f(cam2base),cv::viz::Color::yellow()
    }
    cloud.addCloud(dep, cv::viz::Color::blue(), cam2wd); //,, pose,, Affine3f(cam2base)cam2w_aff
    // assert(points.rows);
    // assert(color.rows);
    // window.spin();
    //     // cloud.addCloud(jix, cv::viz::Color::red(), pose);          //,cv::Affine3f(pose) , pose
    //     cloud.addCloud(jix[0], jixrgb[0]); //,cv::Affine3f(pose) , pose, cv::viz::Color::blue()
    //     cloud.addCloud(jix[1], jixrgb[1]); //,cv::Affine3f(pose) , pose, cv::viz::Color::blue()
    // cout << sizeof(box32) << endl;
    // if (currtsdf->box32s.size() > 400)
    if (now_id % 3 == 5)
    {
      // gm.addcube(currtsdf);

      currtsdf->Voxel2PointCloud(color, points);
      allclone[0].push_back(points);
      allclone[1].push_back(color);

      // old_tsdfs.push_back(currtsdf);
      Vec4s ct(0, 0, 0, 0);
      ct[0] = po.val[3] * 3.125f;  //(0, 0, 0, 0)
      ct[1] = po.val[7] * 3.125f;  //(0, 0, 0, 0)
      ct[2] = po.val[11] * 3.125f; //(0, 0, 0, 0)

      currtsdf = std::make_shared<MyTSDF>(1, ct); //Vec4s(32,3,48,0));
                                                  // gm.getcube(currtsdf);
                                                  // gm.Voxel2PointCloud(color, points);
                                                  // //   // points = old_tsdfs[i]->Voxel2PointCloud(color);
                                                  // assert(points.rows);
                                                  // assert(color.rows);
                                                  // window.spin();
                                                  //   //
                                                  //   //   currtsdf->m_center = currtsdf->m_centerf * 3.125f;
                                                  //   //   // old_tsdfs.back()->over(currtsdf, pos);
                                                  //   //   cout << "val" << currtsdf->m_center << endl;
                                                  //   //   cv::viz::WCube cube(Vec3d(-3.24 + currtsdf->m_centerf[0], -3.24 + currtsdf->m_centerf[1], -3.24 + currtsdf->m_centerf[2]),
                                                  //   //                       Vec3d(3.24 + currtsdf->m_centerf[0], 3.24 + currtsdf->m_centerf[1], 3.24 + currtsdf->m_centerf[2]));
                                                  //   //   window.showWidget("tsdfmax", cube);
                                                  //   cout << "asaa" << endl;
    }
    if (allclone[0].rows > 0)
      cloud.addCloud(allclone[0], allclone[1]); //,, pose,, Affine3f(cam2base),cv::viz::Color::yellow()
    if (keyval == 1)
    {
      keyval = 0;
      MyTSDF::Mat_save_by_binary(pos, "posindex.bin");
      break;
    }
    window.showWidget("cloud", cloud); //,Affine3f(cv::Mat_<float>::eye(Size(3, 3)), cv::Vec3f(voxel_grid_origin_x, 0.0, 0))

    window.spinOnce(1, false);
  }
  // db_close(&newdb);
  return 0;
}
