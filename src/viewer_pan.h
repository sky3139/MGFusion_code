#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <condition_variable> //头文件
#include <vector>
#include "cuda/datatype.cuh"
std::mutex m_mutex;
//     // std::vector<UPoints> vbp;
//     std::vector<std::shared_ptr<std::vector<UPoints>>> vbp;

//     void RenderScene(void)
//     {

//         {
//             std::unique_lock<std::mutex> _Lock(m_mutex);

//             // if (clouds.size() == 0 && vbp.size() != 0)
//             //     GL::clouds.push_back(new PangoCloud(vbp));
//             // for (size_t i = 0; i < clouds.size(); i++)
//             // {
//             //     std::cout << i << std::endl;
//             //     // clouds[i]->drawPoints();
//             //     // for (auto &it : clouds[i])
//             //     // {
//             //     //     it.print();
//             //     // }
//             // }
//             glBegin(GL_POINTS);
//             // GLuint vbo = 0;
//             // glGenBuffers(1, &vbo);
//             // glBindBuffer(GL_ARRAY_BUFFER, vbo);

//             // for (auto &pvter : vbp)
//             {

//                 // clouds[0]->drawPoints();
//                 // for (auto &it : *pvter)
//                 // {
//                 //     glColor3ubv(&it.color.x);
//                 //     glVertex3fv(&it.pos.x);
//                 //     // it.print();
//                 // }
//                 // break;
//             }

//             glEnd();
//         }

//         // printf("a\n");
//     }

//     //     void loop(void)
//     //     {

//     //         glutInit(&argc, argv);
//     //         glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
//     //         glutInitWindowSize(1080, 720);
//     //         glutCreateWindow("Ambient Light Demo");
//     //         glEnable(GL_BLEND);
//     //         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//     //         glEnable(GL_DEPTH_TEST);
//     //         glEnable(GL_PROGRAM_POINT_SIZE);
//     //         glewExperimental = GL_TRUE;
//     //         glewInit();
//     //         // glutReshapeFunc(GL::ChangeSize);
//     //         // // glutSpecialFunc(GL::SpecialKeys);
//     //         // glutKeyboardFunc(GL::Keyboard);
//     //         // glutMotionFunc(GL::Motion);
//     //         // glutMouseFunc(GL::Mouse);
//     //         glutDisplayFunc(GL::RenderScene);
//     //         // SetupRC();
//     //         GL::clouds.push_back(new PangoCloud(vbp));
//     //         glutMainLoop();
//     //     }
//     //     void vrand()
//     //     {
//     //         // std::this_thread::sleep_for(std::chrono::milliseconds(10));
//     //         // {
//     //         //     std::unique_lock<std::mutex> _Lock(m_mutex);
//     //         //     // m_cond.wait(_Lock);
//     //         //     window->spinOnce(1, true);
//     //         // }
//     //         // m_cond.notify_all();
//     //     }

//     //     // void inset_depth(const cv::Mat &depth, const cv::Affine3f &cp)
//     //     // {
//     //     //     cv::viz::WCloud wc(depth, cv::viz::Color::blue());

//     //     //     std::unique_lock<std::mutex> _Lock(m_mutex);
//     //     //     window->showWidget("depth", wc, cp);
//     //     //     // m_cond.notify_all();
//     //     // }
//     //     // void inset_depth2(const cv::Mat &depth, const cv::Affine3f &cp)
//     //     // {
//     //     //     cv::viz::WCloud wc(depth, cv::viz::Color::yellow());

//     //     //     std::unique_lock<std::mutex> _Lock(m_mutex);
//     //     //     window->showWidget("depth2", wc, cp);
//     //     //     // m_cond.notify_all();
//     //     // }

//     //     // void setstring(string str)
//     //     // {
//     //     //     cv::viz::WText wt(str, cv::Point(0, 0), 20);
//     //     //     std::unique_lock<std::mutex> _Lock(m_mutex);
//     //     //     window->showWidget("txet", wt);
//     //     //     // m_cond.notify_all();
//     //     // }

//     //     // cv::Mat getScreenshot()
//     //     // {
//     //     //     std::unique_lock<std::mutex> _Lock(m_mutex);
//     //     //     return window->getScreenshot();
//     //     // }

//     //     // cv::Affine3f getpose()
//     //     // {
//     //     //     std::unique_lock<std::mutex> _Lock(m_mutex);
//     //     //     return window->getViewerPose();
//     //     // }
//     //     void join()
//     //     {
//     //         pthd.join();
//     //     }
//     // };

#include <GL/glew.h>
#include <GL/glut.h>
#include <cstdio>
#include <iostream>

#define CHECK_GL_ERROR() checkGLError(__FILE__, __LINE__)

#pragma pack(push, 1)

#pragma pack(pop)

#include <vector>

struct PangoCloud
{
public:
    GLuint vbo;
    PangoCloud(std::vector<UPoints> &cloud)
        : numPoints(cloud.size()),
          offset(3),
          stride(sizeof(UPoints))
    {
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, cloud.size() * stride, cloud.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    virtual ~PangoCloud()
    {
        glDeleteBuffers(1, &vbo);
    }

    void drawPoints()
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexPointer(3, GL_FLOAT, stride, 0);
        glColorPointer(3, GL_UNSIGNED_BYTE, stride, (void *)(sizeof(float) * offset));

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        glDrawArrays(GL_POINTS, 0, numPoints);

        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    const int numPoints;

private:
    const int offset;
    const int stride;
};

using namespace std;

GLuint vao;
int width = 1080;
int height = 720;
void checkGLError(const char *file, int line)
{
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR)
    {
        std::cout << file << ":" << line << "   Error: " << gluErrorString(err) << std::endl;
    }
}

void printStatus(const char *step, GLuint context, GLuint status)
{
    GLint result = GL_FALSE;
    CHECK_GL_ERROR();
    glGetShaderiv(context, status, &result);
    CHECK_GL_ERROR();
    if (result == GL_FALSE)
    {
        char buffer[1024];
        if (status == GL_COMPILE_STATUS)
            glGetShaderInfoLog(context, 1024, NULL, buffer);
        else
            glGetProgramInfoLog(context, 1024, NULL, buffer);
        if (buffer[0])
            fprintf(stderr, "%s: %s\n", step, buffer);
    }
}
vector<UPoints> points(10000);
namespace GL
{
    float zoom = 10.f;
    float rotx = 20;
    float roty = 0;
    float tx = 0;
    float ty = 0;
    int lastx = 0;
    int lasty = 0;
    unsigned char Buttons[3] = {0};
    std::vector<PangoCloud *> clouds;
    std::queue<std::vector<UPoints> *> vclouds;
    std::vector<float3> poses;
    void Keyboard(unsigned char key, int x, int y)
    {
        switch (key)
        {
        case 'q':
        case 'Q':
        case 27: // ESC key
            // delete bf1, bf2;
            exit(0);
            break;
        }
        // 刷新窗口
        glutPostRedisplay();
    }
    void Motion(int x, int y)
    {
        int diffx = x - lastx;
        int diffy = y - lasty;
        lastx = x;
        lasty = y;

        if (Buttons[2])
        {
            tx += (float)0.1 * diffx;
            ty -= (float)0.1 * diffy;
        }
        else if (Buttons[0])
        {
            rotx += (float)0.51 * diffy;
            roty += (float)0.51 * diffx;
        }
        else if (Buttons[1]) //中
        {

            //
        }
        // printf("%d,a %f,diffy%d\n", Buttons[1], tx, diffy);
        glutPostRedisplay();
    }
    void Mouse(int b, int s, int x, int y)
    {
        lastx = x;
        lasty = y;
        switch (b)
        {
        case GLUT_LEFT_BUTTON:
            Buttons[0] = ((GLUT_DOWN == s) ? 1 : 0);
            break;
        case GLUT_MIDDLE_BUTTON:
            Buttons[1] = ((GLUT_DOWN == s) ? 1 : 0);
            break;
        case GLUT_RIGHT_BUTTON:
            Buttons[2] = ((GLUT_DOWN == s) ? 1 : 0);
            break;
        case 3:
            zoom -= (float)1;
            break;
        case 4:
            zoom += (float)1;
            break;
        default:
            // printf("s=%d,%x,%x\n", b, s, x);
            break;
        }
        glutPostRedisplay();
    }
    void DrawCoordinate(float _flengthX, float _flengthY, float _flengthZ);

    void onDisplay(void)
    {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 保存矩阵状态
        glPushMatrix();
        // glTranslatef(0, 0, );
        glTranslatef(GL::tx, GL::ty, -GL::zoom);
        glRotatef(GL::rotx, 1.0f, 0.0f, 0.0f);
        glRotatef(GL::roty, 0.0f, 1.0f, 0.0f);
        DrawCoordinate(1.0, 1.0, 1.0);
        {
            std::unique_lock<std::mutex> _Lock(m_mutex);
            // delete clouds[0];
            // clouds[0] = new PangoCloud(*vclouds.back());
            for (auto &pvter : clouds)
            {
                pvter->drawPoints();
            }

            if (vclouds.size())
            {
                auto p = vclouds.front();
                if (p != 0)
                {
                    delete clouds[0];
                    clouds[0] = (new PangoCloud(*p));
                    // PangoCloud pd(*p);
                    // pd.drawPoints();
                }

                delete p;

                vclouds.pop();
            }
            glColor3f(1, 0, 0);
            glEnableClientState(GL_VERTEX_ARRAY);

            glVertexPointer(3, GL_FLOAT, 0, poses.data());
            glDrawArrays(GL_LINE_STRIP, 0, poses.size());
            glDisableClientState(GL_VERTEX_ARRAY);

            // glBegin(GL_LINES);
            //
            // for (auto &it : poses)
            // {
            //     glVertex3fv(&it.x);
            //     // for (auto &pvter : *p)
            //     // {
            //     //
            //     // }
            // }
            // glEnd();
            //
            //
            // glVertex3f(_flengthX, 0, 0);
        }

        glPopMatrix();
        // points.resize(0);
        // for (auto it : Shape::allShapes) it->render();
        glutSwapBuffers();
        glutPostRedisplay();
    }
    void DrawCoordinate(float _flengthX, float _flengthY, float _flengthZ)
    {
        glLineWidth(5);
        glBegin(GL_LINES);
        glColor3f(1, 0, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(_flengthX, 0, 0);
        glEnd();

        glBegin(GL_LINES);
        glColor3f(0, 1, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(0, _flengthY, 0);
        glEnd();

        glBegin(GL_LINES);
        glColor3f(0, 0, 1);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, _flengthZ);
        glEnd();
    }
    // 窗口大小改变时的处理
    void ChangeSize(int w, int h)
    {
        GLfloat nRange = 80.0f;
        // 避免除0
        if (h == 0)
            h = 1;
        // 设置视口大小
        glViewport(0, 0, w, h);
        // 重置投影矩阵
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45, (float)w / h, 0.1, 1000);
        // 重置模型视图矩阵
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }
}

class viewer
{
public:
    int argc;
    char **argv;
    viewer(int argc, char **argv) : argc(argc), argv(argv)
    {

        for (int i = 0; i < 10000; ++i)
        {
            UPoints p;
            p.xyz[0] = float(rand()) / RAND_MAX;
            p.xyz[1] = float(rand()) / RAND_MAX;
            p.xyz[2] = float(rand()) / RAND_MAX;

            p.rgb[0] = 255;
            p.rgb[1] = 100;
            p.rgb[2] = 0;
            points[i] = p;
        }
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowSize(width, height);
        glutCreateWindow("mini");

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_PROGRAM_POINT_SIZE);

        glewExperimental = GL_TRUE;
        glewInit();
        GL::clouds.push_back(new PangoCloud(points));
        glutDisplayFunc(GL::onDisplay);
        glutReshapeFunc(GL::ChangeSize);

        glutKeyboardFunc(GL::Keyboard);
        glutMotionFunc(GL::Motion);
        glutMouseFunc(GL::Mouse);
    }
    void inset_cloud(string name, std::vector<UPoints> *wc) // const cv::Mat &_points, const cv::Mat &color)
    {
        std::unique_lock<std::mutex> _Lock(m_mutex);
        // delete GL::clouds[0];
        // wc->swap();
        // points.insert(wc->begin(), wc->end(), points.end());
        // // GL::clouds[0]=(new PangoCloud(*wc));
        GL::vclouds.push(wc);
        // if (wc->size())
        //     GL::clouds.push_back(new PangoCloud(*wc));
        // printf("%ld %ld\n", wc.size(), GL::clouds.size());

        // cv::viz::WCloud wc(_points, color);
        // m_cond.notify_all();
    }
    void inset_traj(cv::Vec<float, 16> &pose)
    {
        std::unique_lock<std::mutex> _Lock(m_mutex);
        GL::poses.push_back(make_float3(pose.val[3], pose.val[7], pose.val[11]));
    }
    void loop()
    {
        glutMainLoop();
    }
};
