#include <mapPoint.h>

namespace superVO
{
    mapPoint::mapPoint(cv::Point3f &mp)
    {
        static unsigned long id = 0;
        mid = id++;
        mpose = Eigen::Vector3d(mp.x, mp.y, mp.z);
    }

    void mapPoint::addObservation(superPoint_* p)
    {
        mobservations_.emplace_back(p);
        mobservationTimes++;
    }

    std::vector<superPoint_*> mapPoint::getObs()
    {
        return mobservations_;
    }

    unsigned long mapPoint::getID()
    {
        return mid;
    }

    Eigen::Vector3d mapPoint::getPose()
    {
        return mpose;
    }

    void mapPoint::setPose(Eigen::Vector3d p)
    {
        mpose = p;
    }

    mapPoint::~mapPoint()
    {
        
    }
}