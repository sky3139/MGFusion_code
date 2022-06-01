// #include <iostream>
// #include <fstream>
// #include <iomanip>
// #include <sstream>
// #include <string>
// #include "dviz.hpp"
// #include <iterator>
// #include <opencv2/viz.hpp>
// #include <opencv2/calib3d.hpp>
// #include <unordered_set>
// #include <iostream>

// #include "MyTSDF.h"
// using namespace cv;
// using namespace std;
// static float tsdf_base_xyz[3] = {0, 0, 0};

// class MyTSDF
// {
// private:
//   /* data */
// public:
//   vector<box32 *> box32s;
//   // std::vector<box32 *> pbox32s(4194304, {NULL});
//   vector<box32 *> *pbox32s;
//   unordered_set<uint32_t> pre_box;
//   int alsize = 1024;
//   int nowsize = 0;
//   float *_pval;
//   float *_weight;
//   uint8_t *_rgb;

//   MyTSDF()
//   {
//     box32s.resize(alsize);
//     pbox32s = new vector<box32 *>(0xffff * 64, NULL);
//     _weight = new float[alsize * 32 * 32 * 32];
//     _rgb = new uint8_t[alsize * 32 * 32 * 32 * 3];
//     for (int i = 0; i < alsize; i++)
//     {
//       box32s[i] = new struct box32();
//     }
//   }
//   void integrate(float *cam_K, float *cam2base, uint8_t *rgb_im, float *depth_im,
//                  int im_height, int im_width, float trunc_margin)
//   {
//     for (auto it = pre_box.begin(); it != pre_box.end(); ++it)
//     {
//       uint32_t i = *it;
//       u32_4byte u32_4;
//       u32_4.u32 = i;

 
//       Vec3f tsdfwordpose;
//       addresstotsdfworld(u32_4, tsdfwordpose);
//       // u32_4.byte4[0] = 0xff & (tsdfval[0] - 10);
//       // u32_4.byte4[1] = 0xff & (tsdfval[1] - 10);
//       // u32_4.byte4[2] = 0xff & (tsdfval[2] - 10);

//       // tsdfwordpose[0] = x * 0.32;
//       // tsdfwordpose[1] = y * 0.32;
//       // tsdfwordpose[2] = z * 0.32;

//       if (u32_4.u32 >= 256 * 256 * 64u) //22 bit
//       {
//         printf("%x,%d,%d,%d\n", u32_4.u32, u32_4.byte4[2], u32_4.u32, 256 * 256 * 64);
//         assert(0);
//       }
//       if ((*pbox32s)[i] == NULL)
//       {
//         (*pbox32s)[i] = box32s[nowsize];
//         (*pbox32s)[i]->index = i;
//         nowsize++;
//         if (nowsize + 5 >= alsize)
//         {
//           // alsize = alsize * 2;
//           box32s.reserve(alsize * 2);
//           cout << "" << box32s.size() << endl;
//           // box32s.resize(alsize); //resize()既修改capacity大小，也修改size大小。
//           for (int i = 0; i < alsize; i++)
//           {
//             struct box32 *pbox32s_l = new struct box32();
//             // pbox32s_l->weight = new float[32 * 32 * 32];
//             assert(pbox32s_l->pVoxel);
//             box32s.push_back(pbox32s_l);
//           }
//           alsize += alsize;
//           cout << "" << box32s.size() << " " << nowsize << endl;
//         }
//       }

//       for (int8_t pt_grid_z = 0; pt_grid_z < 32; pt_grid_z++)
//       {
//         for (int8_t pt_grid_y = 0; pt_grid_y < 32; pt_grid_y++)
//         {
//           for (int8_t pt_grid_x = 0; pt_grid_x < 32; pt_grid_x++)
//           {

//             float pt_base_x = tsdfwordpose[0] + pt_grid_x * 0.01f - 0.16;
//             float pt_base_y = tsdfwordpose[1] + pt_grid_y * 0.01f - 0.16;
//             float pt_base_z = tsdfwordpose[2] + pt_grid_z * 0.01f - 0.16;

//             // Convert from base frame camera coordinates to current frame camera coordinates
//             float tmp_pt[3] = {0};
//             tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
//             tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
//             tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
//             float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
//             float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
//             float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

//             if (pt_cam_z <= 0)
//               continue;

//             int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
//             int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
//             if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
//               continue;

//             float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];

//             if (depth_val <= 0 || depth_val > 6)
//               continue;

//             float diff = depth_val - pt_cam_z;
//             // std::cout << depth_val<<" "<<pt_cam_z<<" \n";// << " ";

//             if (diff <= -trunc_margin)
//               continue;

//             // Integrate
//             int volume_idx = pt_grid_z * 32 * 32 + pt_grid_y * 32 + pt_grid_x;
//             union Voxel &voxel = (*pbox32s)[i]->pVoxel[volume_idx];

//             float dist = fmin(1.0f, diff / trunc_margin);
//             auto weight_old = voxel.weight;
//             auto weight_new = std::min(weight_old + 1, 255);
//             {
//               voxel.weight = weight_new;
//               voxel.tsdfval = (voxel.tsdfval * (float)weight_old + dist) / (float)weight_new;
//             }

//             // Integrate color

//             uint8_t rgb_val[3];
//             rgb_val[0] = rgb_im[pt_pix_y * im_width + pt_pix_x];
//             rgb_val[1] = rgb_im[pt_pix_y * im_width + pt_pix_x + 640 * 480];
//             rgb_val[2] = rgb_im[pt_pix_y * im_width + pt_pix_x + 640 * 480 * 2];

//             int mval = (voxel.r * weight_old + rgb_val[0]) / weight_new;
//             voxel.r = mval > 255 ? 255 : mval;
//             mval = (voxel.g * weight_old + rgb_val[1]) / weight_new;
//             voxel.g = mval > 255 ? 255 : mval;
//             mval = (voxel.b * weight_old + rgb_val[2]) / weight_new;
//             voxel.b = mval > 255 ? 255 : mval;
//           }
//         }
//       }
//       // printf("%d,%d,%d,%x\n", x, y, z, i);
//       // cout << "tsdfwordpose" << tsdfwordpose << endl;
//     }
//     pre_box.clear();
//     print();
//   }
//   cv::Mat Voxel2PointCloud(cv::Mat &color)
//   {
//     cv::Mat cloud;
//     assert(box32s.size());
//     for (auto &it : box32s)
//     {
//       if (it->index != 0xffffffff)
//       {
//         // cout << it->index <</ " ";
//         int i = it->index;

//         u32_4byte u32_4;
//         u32_4.u32 = i;
//         cv::Vec3f tsdfwordpose;
//         addresstotsdfworld(u32_4, tsdfwordpose);
//         for (int8_t pt_grid_z = 0; pt_grid_z < 32; pt_grid_z++)
//         {
//           for (int8_t pt_grid_y = 0; pt_grid_y < 32; pt_grid_y++)
//           {
//             for (int8_t pt_grid_x = 0; pt_grid_x < 32; pt_grid_x++)
//             {
//               int volume_idx = pt_grid_z * 32 * 32 + pt_grid_y * 32 + pt_grid_x;
//               union Voxel &voxel = (*pbox32s)[i]->pVoxel[volume_idx];

//               // if (std::abs(voxel.tsdfval) < 0.2 && voxel.tsdfval > 0)
//               if (pt_grid_x % 5 == 0)
//               {
//                 cv::Vec3f vec;
//                 vec[0] = tsdfwordpose[0] + pt_grid_x * 0.01f - 0.16;
//                 vec[1] = tsdfwordpose[1] + pt_grid_y * 0.01f - 0.16;
//                 vec[2] = tsdfwordpose[2] + pt_grid_z * 0.01f - 0.16;
//                 cloud.push_back(vec);

//                 cv::Vec<uint8_t, 3> vescolor;
//                 vescolor[0] = voxel.r;
//                 vescolor[1] = voxel.g;
//                 vescolor[2] = voxel.b;
//                 color.push_back(vescolor);
//               }
//             }
//           }
//         }
//       }
//     }
//     return cloud;
//   }

//   // void word2tsdfax(Vec4f &wpoint)
//   // {

//   //   // wpoint[0] = wpoint[0] + tsdf_base_xyz[0];
//   //   // wpoint[1] = wpoint[1] + tsdf_base_xyz[1];
//   //   // assert(wpoint[2] < 10.24);     //10.24=2<<6 /2
//   //   // wpoint[2] = wpoint[2] + 5.24; //保证Z为正
//   //   // wpoint
//   //   Vec4s tsdfval = wpoint * 3.125f; // / 0.32;

//   //   u32_4byte u32_4;
//   //   // u32_4.byte4[0] = 0xff & (tsdfval[0]);
//   //   // u32_4.byte4[1] = 0xff & (tsdfval[1]);
//   //   // u32_4.byte4[2] = 0xff & (tsdfval[2]);
//   //   // u32_4.byte4[3] = 0;

//   //   u32_4.u32 = ((0xff & tsdfval[2]) << 16 | (0xff & tsdfval[1]) << 8 | (0xff & tsdfval[0]));
//   //   // if (u32_4.byte4[2] > 64)
//   //   // printf("%x,%x,%x,%x\n", tsdfval[0], tsdfval[1], tsdfval[2], u32_4.u32);
//   //   pre_box.insert(u32_4.u32);
//   //   // std::cout << "tsdfval:" << tsdfval << endl;
//   // }
//   void tsdf2wordax(Vec3b &wpoint)
//   {
//     Vec3f tsdfval = wpoint * 0.32; // * 0.32;

//     // uint32_t i0 = ((0xff & tsdfval[2]) << 16 | (0xff & tsdfval[1]) << 8 | (0xff & tsdfval[0]));
//     // printf("%x,%x,%x,%x\n", tsdfval[0], tsdfval[1], tsdfval[2], i0);
//     // pre_box.insert(i0);
//     // std::cout << "tsdfval:" << tsdfval << endl;
//   }
//   void print()
//   {
//     std::cout << "nowsize:" << nowsize << endl;
//   }
// };

// int main(int argc, char *argv[])
// {
//   std::string cam_K_file = "../data/camera-intrinsics.txt";

//   std::string data_path = "../data/data";
//   int base_frame_idx = 1 - 1;
//   int first_frame_idx = 1;
//   int num_frames = 4000;

//   float cam_K[3 * 3];
//   float base2world[4 * 4];
//   float cam2base[4 * 4];
//   float cam2world[4 * 4];
//   int im_width = 640;
//   int im_height = 480;
//   float depth_im[im_height * im_width];
//   uint8_t RGB_im[im_height * im_width * 3];

//   float voxel_size = 0.01f;
//   float trunc_margin = voxel_size * 5;

//   //读取相机内参
//   std::vector<float> cam_K_vec = LoadMatrixFromFile(cam_K_file, 3, 3);
//   std::copy(cam_K_vec.begin(), cam_K_vec.end(), cam_K);

//   // 读取第一帧的位姿
//   cv::Mat posss;
//   Mat_read_binary(posss, "/home/u16/dataset/img/loop22pose.bin");
//   // cout << posss << endl;
//   Vec<float, 16> po = posss.at<Vec<float, 16>>(base_frame_idx, 0);
//   for (int i = 0; i < 16; i++)
//   {
//     base2world[i] = po.val[i];
//     // cout << po.val[i] << " " << cam2world[i] << "\n";
//   }
//   Affine3f base(po.val);
//   // 位姿逆矩阵
//   float base2world_inv[16] = {0};
//   invert_matrix(base2world, base2world_inv);

//   // 初始化格子
//   Affine3f a3f(cv::Mat_<float>::eye(Size(3, 3)), cv::Vec3f(0.0, 0.0, 0.5));

//   char *pname = new char[256];
//   int idx = 1;

//   cv::viz::Viz3d window("window");
//   // cout << depth << endl;
//   //显示坐标系
//   window.showWidget("Coordinate", viz::WCoordinateSystem());

//   //创建一个储存point cloud的图片
//   char *name = new char[256];
//   int i = 11;
//   // cv::viz::WCube cube(Vec3d(-2.5, -2.5, 0.5), Vec3d(2.5, 2.5, 5.5));
//   // window.showWidget("tsdf", cube);
//   // int indexxx[] = {3, 5, 7, 9, 11, 13, 15, 17, 17, 17, 17, 17, 17, 17, 17};
//   // for (int i = 0; i < 15; i++)
//   // {
//   //   viz::WGrid *wg = new viz::WGrid(Vec3d(0, 0, (i + 1) * 32 * voxel_size), Vec3d(0, 0, 1), Vec3d(0, 1, 0), Vec2i::all(indexxx[i]), Vec2d::all(voxel_size * 32));
//   //   window.showWidget("asd" + to_string(i), *wg, a3f);
//   // }

//   // viz::WGrid wg
//   // viz::WGrid wg1(Vec3d(0, 0, 1), Vec3d(1, 0, 0), Vec3d(0, 0, 1), Vec2i::all(17), Vec2d::all(voxel_size * 32));

//   // window.showWidget("asd1", wg1);

//   // cv::Mat mpc;
//   float aplen = 512 * voxel_size;

//   // window.showWidget("mpc", wpl, a3f); //, Affine3f(base2world)

//   MyTSDF mytsdf;
//   cv::Affine3f cam2wd = cv::Affine3f::Identity();
//   // 重建 TSDF voxel grid
//   for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; frame_idx += 5)
//   {
//     cv::viz::WCloudCollection cloud;
//     // po.val[3] /= 1000;
//     // po.val[7] /= 1000;
//     // po.val[11] /= 1000;
//     po = posss.at<Vec<float, 16>>(frame_idx, 0);
//     // po.val[3] -= 10;
//     // po.val[7] -= 10;
//     // po.val[11] = 0;

//     cv::Affine3f pose(&po.val[0]);
//     // pose.translate
//     cam2wd = pose; //cam2wd.concatenate(pose);
//     cout << cam2wd.matrix << endl;
//     cout << po << endl;

//     std::ostringstream curr_frame_prefix;
//     curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;
//     // sprintf(name, "/home/lei/dataset/infiniTAM/Files/tum/Frames/%04d.pgm", i++);
//     // sprintf(name, "/home/lei/dataset/surfulwrap/ubnm/frame-%06d.depth.png", i++);
//     // std::string depth_im_file = "/home/lei/图片/img/" + std::to_string(i) + ".png";
//     // i += 2;
//     // 读取深度图
//     // sprintf(name, "/home/lei/docker/res/killingFusionCuda-master/data/hat/depth_%06d.png", frame_idx);
//     // sprintf(name, "/home/lei/dataset/infiniTAM/Files/tum/Frames/%04d.pgm", i++);

//     string depth_path = cv::format("/home/u16/dataset/img/depth/%04d.png", frame_idx);
//     cv::Mat _depth = cv::imread(depth_path, 2);
//     if (_depth.empty())
//     {
//       string ss = "ls " + depth_path;
//       system(ss.c_str());
//       assert(0);
//     }

//     // std::string depth_im_file = data_path + "/frame-" + curr_frame_prefix.str() + ".depth.png";
//     ReadDepth(depth_path, im_height, im_width, depth_im);
//     // Mat dp = imread(depth_im_file, 2);
//     Mat dep = getCload(_depth);
//     // cv::Mat dp = cv::imread("/home/u16/dataset/img/depth/1600.png", 2);
//     string color_path = cv::format("/home/u16/dataset/img/rgb/%04d.png", frame_idx);

//     cv::Mat _reacolor = cv::imread(color_path, 0);
//     if (_reacolor.empty())
//     {
//       string ss = "ls " + color_path;
//       system(ss.c_str());
//       assert(0);
//     }
//     // // ReadDepth(string(name), im_height, im_width, depth_im);
//     // depth_im_file = data_path + "/frame-" + curr_frame_prefix.str() + ".color.jpg";
//     ReadRGB(color_path, im_height, im_width, RGB_im);
//     // // 读取位姿
//     float pppp[16];
//     for (int i = 0; i < 16; i++)
//     {
//       cam2world[i] = po.val[i];
//       // cout << po.val[i] << " " << cam2world[i] << "\n";
//     }
//     // float sadasd[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
//     // // std::vector<float> cam2world_vec(sadasd);// = LoadMatrixFromFile(cam2world_file, 4, 4);
//     // //  cam2world_vec.push_back(1);//
//     // // std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);

//     // // 计算相对于第一帧的位姿 (camera-to-base frame)
//     // std::copy(cam2world_vec.begin(), cam2world_vec.end(), std::ostream_iterator<float>(std::cout, " "));
//     multiply_matrix(base2world_inv, cam2world, cam2base); //

//     // std::cout << "Fusing: " << depth_im_file << std::endl;
//     Affine3f cam2w_aff(cam2world);
//     // cv::viz::Camera mainCamera = cv::viz::Camera::KinectCamera(Size(640, 480));     //new viz::Camera();         // 初始化相机类
//     // viz::WCameraPosition camParamsp(mainCamera.getFov(), 4.0, viz::Color::white()); // 相机参数设置
//     // window.showWidget("Camera", camParamsp, cam2w_aff);
//     // Vec3f ct = cam2w_aff.translation();

//     // vconcat（B, C，A）; // 等同于A=[B ;C]
//     Mat_<float> mp(4, 4, &po.val[0]);
//     // cout << "dep" << dep.t() << endl;

//     Mat adep41 = dep.reshape(1);
//     Mat outtt = adep41 * mp.t();
//     // printMAT(adep41);
//     // printMAT(mp);
//     // printMAT(outtt);
//     for (int i = 0; i < outtt.rows; i++)
//     {
//       Vec4f ct = outtt.at<Vec4f>(i, 0);
//       mytsdf.word2tsdfax(ct);
//     }
//     mytsdf.integrate(cam_K, &po.val[0], RGB_im, depth_im, im_height, im_width, trunc_margin);
//     cv::Mat color;
//     Mat points = mytsdf.Voxel2PointCloud(color);
//     cloud.addCloud(dep, cv::viz::Color::blue(), cam2wd); //,, pose,, Affine3f(cam2base)cam2w_aff
//     assert(points.rows);
//     assert(color.rows);
//     cloud.addCloud(points, color);     //,, pose,, Affine3f(cam2base),cv::viz::Color::yellow()
//     window.showWidget("cloud", cloud); //,Affine3f(cv::Mat_<float>::eye(Size(3, 3)), cv::Vec3f(voxel_grid_origin_x, 0.0, 0))
//     // window.spin();
//     //     // cloud.addCloud(jix, cv::viz::Color::red(), pose);          //,cv::Affine3f(pose) , pose
//     //     cloud.addCloud(jix[0], jixrgb[0]); //,cv::Affine3f(pose) , pose, cv::viz::Color::blue()
//     //     cloud.addCloud(jix[1], jixrgb[1]); //,cv::Affine3f(pose) , pose, cv::viz::Color::blue()
//     window.spinOnce(1, false);
//   }
//   return 0;
// }
