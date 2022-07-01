#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <condition_variable> //头文件
#include <opencv2/viz.hpp>
using namespace std;
using namespace cv;

class viewer
{
private:
    /* data */
public:
    std::thread pthd;

    cv::viz::Viz3d *window; //("window");
    cv::Mat poses;
    std::mutex m_mutex;
    std::condition_variable m_cond;
    std::atomic_bool is_exit{true};
    viewer(int argc, char **argv)
    {
        window = new cv::viz::Viz3d("map");
        window->showWidget("Coordinate", cv::viz::WCoordinateSystem());
        pthd = std::thread(&viewer::loop, this);
    }
    int i = 0;
    void loop(void)
    {
        bool ret;
        do
        {
            ret = is_exit.load();
            vrand();
            // std::cout<<ret<<std::endl;

        } while (ret);
    }
    void vrand()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        {
            std::unique_lock<std::mutex> _Lock(m_mutex);
            // m_cond.wait(_Lock);
            window->spinOnce(1, true);
        }
        // m_cond.notify_all();
    }
    void inset_cloud(string name, cv::viz::WCloud wc) // const cv::Mat &_points, const cv::Mat &color)
    {
        // cv::viz::WCloud wc(_points, color);
        std::unique_lock<std::mutex> _Lock(m_mutex);
        window->showWidget(name, wc);
        // m_cond.notify_all();
    }
    void inset_traj(const cv::Vec<float, 16> &pose)
    {
        poses.push_back(pose);
        // cv::viz::WTrajectory wtraj(poses,cv::viz::WTrajectory::FRAMES, 1.0); //FRAMES
        cv::viz::WTrajectory wtraj(poses, cv::viz::WTrajectory::PATH, 1.0);

        std::unique_lock<std::mutex> _Lock(m_mutex);
        cv::Mat myMat_ = (cv::Mat_<double>(3, 3) << 480.0, 0.0, 320.0,
                          0.0, 480.0, 267.0,
                          0.0, 0.0, 1.0);
        const Matx33d K(myMat_);
        cv::viz::WCameraPosition wp(K, 3.0, cv::viz::Color::white());
        window->showWidget("wp", wp, cv::Affine3f(pose.val));

        window->showWidget("wtraj", wtraj);
        // window->setViewerPose(cv::Affine3f(pose.val));
        // m_cond.notify_all();
    }

    void inset_depth(const cv::Mat &depth, const cv::Affine3f &cp)
    {
        cv::viz::WCloud wc(depth, cv::viz::Color::blue());

        std::unique_lock<std::mutex> _Lock(m_mutex);
        window->showWidget("depth", wc, cp);
        // m_cond.notify_all();
    }
    void inset_depth2(const cv::Mat &depth, const cv::Affine3f &cp)
    {
        cv::viz::WCloud wc(depth, cv::viz::Color::yellow());

        std::unique_lock<std::mutex> _Lock(m_mutex);
        window->showWidget("depth2", wc, cp);
        // m_cond.notify_all();
    }

    void setstring(string str)
    {
        cv::viz::WText wt(str, cv::Point(0, 0), 20);
        std::unique_lock<std::mutex> _Lock(m_mutex);
        window->showWidget("txet", wt);
        // m_cond.notify_all();
    }

    cv::Mat getScreenshot()
    {
        std::unique_lock<std::mutex> _Lock(m_mutex);
        return window->getScreenshot();
    }

    cv::Affine3f getpose()
    {
        std::unique_lock<std::mutex> _Lock(m_mutex);
        return window->getViewerPose();
    }
};
