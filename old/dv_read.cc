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
  std::string cam_K_file = "../data/camera-intrinsics.txt";

  int base_frame_idx = 460 - 1;
  int first_frame_idx = 460;
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
  MyTSDF::Mat_read_binary(posss, "/home/u16/dataset/img/loop22pose.bin");
  // cout << posss << endl;
  Vec<float, 16> po = posss.at<Vec<float, 16>>(base_frame_idx, 0);
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
  // cv::viz::WCube cube(Vec3d(-81.92,-81.92, -10.24), Vec3d(81.92, 81.92, 10.24));

  // int indexxx[] = {3, 5, 7, 9, 11, 13, 15, 17, 17, 17, 17, 17, 17, 17, 17};
  // for (int i = 0; i < 15; i++)
  // {
  //   viz::WGrid *wg = new viz::WGrid(Vec3d(0, 0, (i + 1) * 32 * voxel_size), Vec3d(0, 0, 1), Vec3d(0, 1, 0), Vec2i::all(indexxx[i]), Vec2d::all(voxel_size * 32));
  //   window.showWidget("asd" + to_string(i), *wg, a3f);
  // }

  // viz::WGrid wg
  // viz::WGrid wg1(Vec3d(0, 0, 1), Vec3d(1, 0, 0), Vec3d(0, 0, 1), Vec2i::all(17), Vec2d::all(voxel_size * 32));

  // window.showWidget("asd1", wg1);
  // cv::viz::WCube cube(Vec3d(-10.24, -10.24, -10.24), Vec3d(10.24, 10.24, 10.24));
  // window.showWidget("tsdfmax", cube);
  // cv::Mat mpc;
  float aplen = 512 * voxel_size;

  // window.showWidget("mpc", wpl, a3f); //, Affine3f(base2world)

  std::shared_ptr<MyTSDF> currtsdf(new MyTSDF(0, Vec4s(0, 0, 0, 0)));

  cv::Affine3f cam2wd = cv::Affine3f::Identity();
  // std::shared_ptr<MyTSDF> currtsdf = f1;
  vector<std::shared_ptr<MyTSDF>> old_tsdfs;
  int keyval = 0;
  // std::shared_ptr<MyTSDF> newtsdf = std::make_shared<MyTSDF>(Vec4s(32,3,48,0));
  window.registerKeyboardCallback(KeybdCallback, &keyval);
  Mat pos;
  MyTSDF::Mat_read_binary(pos, "posindex.bin");

  for (size_t i = 0; i < pos.rows; i++)
  {
    Vec4s tsdfwordpose = pos.at<Vec4s>(i, 0);
    box32 *pbox = new struct box32;
    pbox->read("box.bin", i);
    u32_4byte u32;
    u32.byte4[0] = tsdfwordpose[0];
    u32.byte4[1] = tsdfwordpose[1];
    u32.byte4[2] = tsdfwordpose[2];

    pbox->index = u32.u32;
    (*(currtsdf->pbox32s))[u32.u32]=pbox;
    currtsdf->box32s.push_back(pbox);
    // currtsdf->nowsize++;
  }
  cv::viz::WCloudCollection cloud;
  Mat points, color;
  cout << pos.rows << endl;
  points = currtsdf->Voxel2PointCloud(color);
  // cloud.addCloud(dep, cv::viz::Color::blue(), cam2wd); //,, pose,, Affine3f(cam2base)cam2w_aff
  assert(points.rows);
  // assert(color.rows);
  // cv::viz::WCloud wc(points, color);
  cloud.addCloud(points, color); //,, pose,, Affine3f(cam2base), 
  window.showWidget("cloud", cloud);                //,Affine3f(cv::Mat_<float>::eye(Size(3, 3)), cv::Vec3f(voxel_grid_origin_x, 0.0, 0))
  // window.spin();
  //     // cloud.addCloud(jix, cv::viz::Color::red(), pose);          //,cv::Affine3f(pose) , pose
  //     cloud.addCloud(jix[0], jixrgb[0]); //,cv::Affine3f(pose) , pose, cv::viz::Color::blue()
  //     cloud.addCloud(jix[1], jixrgb[1]); //,cv::Affine3f(pose) , pose, cv::viz::Color::blue()
  window.spin();

  assert(0);
  // // 重建 TSDF voxel grid
  // for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; frame_idx += 22)
  // {
  //   cv::viz::WCloudCollection cloud;
  //   Vec<float, 16> src = posss.at<Vec<float, 16>>(frame_idx, 0);
  //   po = src;
  //   src.val[3] -= currtsdf->m_centerf[0];
  //   src.val[7] -= currtsdf->m_centerf[1];
  //   src.val[11] -= currtsdf->m_centerf[2];
  //   cv::Affine3f pose(&po.val[0]);
  //   cam2wd = pose; //cam2wd.concatenate(pose);
  //   std::ostringstream curr_frame_prefix;
  //   curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;
  //   // sprintf(name, "/home/lei/dataset/infiniTAM/Files/tum/Frames/%04d.pgm", i++);
  //   // sprintf(name, "/home/lei/dataset/surfulwrap/ubnm/frame-%06d.depth.png", i++);
  //   // std::string depth_im_file = "/home/lei/图片/img/" + std::to_string(i) + ".png";
  //   // i += 2;
  //   // 读取深度图
  //   // sprintf(name, "/home/lei/docker/res/killingFusionCuda-master/data/hat/depth_%06d.png", frame_idx);
  //   // sprintf(name, "/home/lei/dataset/infiniTAM/Files/tum/Frames/%04d.pgm", i++);

  //   string depth_path = cv::format("/home/u16/dataset/img/depth/%04d.png", frame_idx);
  //   cv::Mat _depth = cv::imread(depth_path, 2);
  //   if (_depth.empty())
  //   {
  //     string ss = "ls " + depth_path;
  //     system(ss.c_str());
  //     assert(0);
  //   }

  //   // std::string depth_im_file = data_path + "/frame-" + curr_frame_prefix.str() + ".depth.png";
  //   ReadDepth(depth_path, im_height, im_width, depth_im);
  //   // Mat dp = imread(depth_im_file, 2);
  //   Mat dep = getCload(_depth);
  //   // cv::Mat dp = cv::imread("/home/u16/dataset/img/depth/1600.png", 2);
  //   string color_path = cv::format("/home/u16/dataset/img/rgb/%04d.png", frame_idx);

  //   cv::Mat _reacolor = cv::imread(color_path);
  //   if (_reacolor.empty())
  //   {
  //     string ss = "ls " + color_path;
  //     system(ss.c_str());
  //     assert(0);
  //   }
  //   cv::imshow("_reacolor", _reacolor); // depth_im_file = data_path + "/frame-" + curr_frame_prefix.str() + ".color.jpg";
  //   cv::waitKey(1);
  //   ReadRGB(color_path, im_height, im_width, RGB_im);
  //   // // 读取位姿
  //   float pppp[16];
  //   for (int i = 0; i < 16; i++)
  //   {
  //     cam2world[i] = po.val[i];
  //     // cout << po.val[i] << " " << cam2world[i] << "\n";
  //   }
  //   // float sadasd[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  //   // // std::vector<float> cam2world_vec(sadasd);// = LoadMatrixFromFile(cam2world_file, 4, 4);
  //   // //  cam2world_vec.push_back(1);//
  //   // // std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);

  //   // // 计算相对于第一帧的位姿 (camera-to-base frame)
  //   // std::copy(cam2world_vec.begin(), cam2world_vec.end(), std::ostream_iterator<float>(std::cout, " "));
  //   multiply_matrix(base2world_inv, cam2world, cam2base); //

  //   // std::cout << "Fusing: " << depth_im_file << std::endl;
  //   Affine3f cam2w_aff(cam2world);
  //   // cv::viz::Camera mainCamera = cv::viz::Camera::KinectCamera(Size(640, 480));     //new viz::Camera();         // 初始化相机类
  //   // viz::WCameraPosition camParamsp(mainCamera.getFov(), 4.0, viz::Color::white()); // 相机参数设置
  //   // window.showWidget("Camera", camParamsp, cam2w_aff);
  //   // Vec3f ct = cam2w_aff.translation();

  //   // vconcat（B, C，A）; // 等同于A=[B ;C]
  //   Mat_<float> mp(4, 4, &po.val[0]);
  //   // cout << "dep" << dep.t() << endl;

  //   Mat adep41 = dep.reshape(1);
  //   Mat outtt = adep41 * mp.t();
  //   // printMAT(adep41);
  //   // printMAT(mp);
  //   // printMAT(outtt);
  //   int i = 0;
  //   Vec4s val;
  //   for (; i < outtt.rows; i++)
  //   {
  //     Vec4f src_ct = outtt.at<Vec4f>(i, 0);
  //     Vec4f ct;
  //     ct[0] = src_ct[0] - currtsdf->m_centerf[0];
  //     ct[1] = src_ct[1] - currtsdf->m_centerf[1];
  //     ct[2] = src_ct[2] - currtsdf->m_centerf[2]; //保证Z为正

  //     // - * 0.32f;

  //     bool ret = currtsdf->word2tsdfax(ct, val);
  //     // cout << "tf over2:" << ct << " " << val << endl;

  //     // if (!ret)
  //     // {
  //     // cout << "a1" << endl;

  //     // old_tsdfs.push_back(currtsdf);

  //     //   currtsdf = std::make_shared<MyTSDF>(val);//Vec4s(32,3,48,0));
  //     //   i = 0;
  //     //   cout << "tf over1:" << ct << " " << val << endl;

  //     //   cout << " WCube1:" << 0 << endl;

  //     //   continue;
  //     //   // cout << po << endl;
  //     //   // assert(0);
  //     // }
  //   }
  //   // if (i != outtt.rows)
  //   // {
  //   //   continue;
  //   // }
  //   // cout << __LINE__ << " inte:" << 0 << endl;
  //   currtsdf->integrate(cam_K, &po.val[0], RGB_im, depth_im, im_height, im_width, trunc_margin);
  //   // cout << __LINE__ << " inte:" << 1 << endl;

  //   cv::Mat color;
  //   Mat points;
  //   // for (int i = 0; i < old_tsdfs.size(); i++)
  //   // {
  //   //   points = old_tsdfs[i]->Voxel2PointCloud(color);
  //   //   assert(points.rows);
  //   //   assert(color.rows);
  //   //   // cv::viz::WCloud wc(points, color);
  //   //   cloud.addCloud(points, color); //,, pose,, Affine3f(cam2base),cv::viz::Color::yellow()
  //   //   color = Mat();
  //   // }

  //   points = currtsdf->Voxel2PointCloud(color);
  //   cloud.addCloud(dep, cv::viz::Color::blue(), cam2wd); //,, pose,, Affine3f(cam2base)cam2w_aff
  //   assert(points.rows);
  //   assert(color.rows);
  //   // cv::viz::WCloud wc(points, color);
  //   cloud.addCloud(points, color);     //,, pose,, Affine3f(cam2base),cv::viz::Color::yellow()
  //   window.showWidget("cloud", cloud); //,Affine3f(cv::Mat_<float>::eye(Size(3, 3)), cv::Vec3f(voxel_grid_origin_x, 0.0, 0))
  //   // window.spin();
  //   //     // cloud.addCloud(jix, cv::viz::Color::red(), pose);          //,cv::Affine3f(pose) , pose
  //   //     cloud.addCloud(jix[0], jixrgb[0]); //,cv::Affine3f(pose) , pose, cv::viz::Color::blue()
  //   //     cloud.addCloud(jix[1], jixrgb[1]); //,cv::Affine3f(pose) , pose, cv::viz::Color::blue()
  //   window.spinOnce(1, false);
  //   // cout << sizeof(box32) << endl;
  //   if (currtsdf->nowsize > 400)
  //   {
  //     old_tsdfs.push_back(currtsdf);
  //     currtsdf = std::make_shared<MyTSDF>(1, val); //Vec4s(32,3,48,0));
  //     currtsdf->m_centerf[0] = po.val[3];
  //     currtsdf->m_centerf[1] = po.val[7];
  //     currtsdf->m_centerf[2] = po.val[11];
  //     old_tsdfs.back()->over(currtsdf, pos);
  //     // cout << "val" << currtsdf->position.rows << endl;
  //     cv::viz::WCube cube(Vec3d(-3.24 + currtsdf->m_centerf[0], -3.24 + currtsdf->m_centerf[1], -3.24 + currtsdf->m_centerf[2]),
  //                         Vec3d(3.24 + currtsdf->m_centerf[0], 3.24 + currtsdf->m_centerf[1], 3.24 + currtsdf->m_centerf[2]));
  //     window.showWidget("tsdfmax", cube);
  //   }
  //   if (keyval == 1)
  //   {
  //     keyval = 0;
  //     MyTSDF::Mat_save_by_binary(pos, "posindex.bin");

  //     break;
  //   }
  // }
  // db_close(&newdb);
  return 0;
}
