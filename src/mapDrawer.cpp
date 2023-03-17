#include <mapDrawer.h>

namespace superVO
{
    mapDrawer::mapDrawer()
    {
        mKeyFrameSize = 0.05;
        mKeyFrameLineWidth = 1;
        mGraphLineWidth = 0.9;
        mPointSize = 2;
        mCameraSize = 0.08;
        mCameraLineWidth = 3;

        mViewpointX = 0;
        mViewpointY = -0.7;
        mViewpointZ = -1.8;
        mViewpointF = 500;
    }


    void mapDrawer::run()
    {
        pangolin::CreateWindowAndBind("Superpoint Map Viewer", 1024, 768);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
                );
        pangolin::View& d_cam = pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
                .SetHandler(new pangolin::Handler3D(s_cam));
        Twc.SetIdentity();
        while(pangolin::ShouldQuit() == false)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            // s_cam.Follow(Twc);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            d_cam.Activate(s_cam);

            Twc = getOpenGLCameraMatrix(mCameraPose);
            drawFrame(Twc);
            pangolin::FinishFrame();
        }

    }

    pangolin::OpenGlMatrix mapDrawer::getOpenGLCameraMatrix(cv::Mat m)
    {
        pangolin::OpenGlMatrix T;
        cv::Mat pose(3, 4, CV_64F);
        pose = m;

        T.m[0] = pose.at<float>(0, 0);
        T.m[1] = pose.at<float>(0, 1);
        T.m[2] = pose.at<float>(0, 2);
        T.m[3] = 0.0;

        T.m[4] = pose.at<float>(1, 0);
        T.m[5] = pose.at<float>(1, 1);
        T.m[6] = pose.at<float>(1, 2);
        T.m[7] = 0.0;

        T.m[8] = pose.at<float>(2, 0);
        T.m[9] = pose.at<float>(2, 1);
        T.m[10] = pose.at<float>(2, 2);
        T.m[11] = 0.0;

        T.m[12] = pose.at<float>(0, 3);
        T.m[13] = pose.at<float>(1, 3);
        T.m[14] = pose.at<float>(2, 3);
        T.m[15] = 1.0;
        return T;
    }


    void mapDrawer::drawFrame(pangolin::OpenGlMatrix T)
    {
        const float w = mCameraSize;
        const float h = w * 0.75;
        const float z = w * 0.6;

        glPushMatrix();
        glMultMatrixd(Twc.m);

        glLineWidth(mCameraLineWidth);
        glColor3f(0.0f,1.0f,0.0f);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();

        pangolin::FinishFrame();
    }

    void mapDrawer::drawKeyFrame(cv::Mat pose)
    {

    }

    mapDrawer::~mapDrawer()
    {

    }
}
