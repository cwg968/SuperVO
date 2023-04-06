#include <tracker.h>

#define UNDISTORT 0

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./superVO -s setting/xxx.yaml // serialize model to plan file" << std::endl;
        std::cerr << "./superVO -d setting/xxx.yaml PATH_TO_SEQUENCE_FOLDER // deserialize plan file and run inference" << std::endl;
        return -1;
    }
    if(argv[1] == "-s" && argc != 3)
    {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./superVO -s setting/xxx.yaml // serialize model to plan file" << std::endl;
        return -1;
    }
    if(argv[1] == "-d" && argc != 4)
    {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./superVO -d setting/xxx.yaml PATH_TO_SEQUENCE_FOLDER // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    if(std::string(argv[1]) == "-s")
    {
        // load camera setting
        std::shared_ptr<superVO::dataLoader> spDataLoader(new superVO::dataLoader(argv[2]));
        // load tensorRT engine
        std::shared_ptr<superVO::trtEngine> spTrtEngine(new superVO::trtEngine(spDataLoader->INPUT_H(), spDataLoader->INPUT_W(), 1, "./build/superpoint_kitti.engine"));
        
        spTrtEngine->buildTRTEngine("./weights/superpoint_v1.wts");
    }
    else if(std::string(argv[1]) == "-d")
    {
        // load camera setting
        std::shared_ptr<superVO::dataLoader> spDataLoader(new superVO::dataLoader(argv[2], argv[3]));
        // load tensorRT engine
        std::shared_ptr<superVO::trtEngine> spTrtEngine(new superVO::trtEngine(spDataLoader->INPUT_H(), spDataLoader->INPUT_W(), 1, "./build/superpoint_kitti.engine"));

        std::shared_ptr<superVO::map> spMap(new superVO::map(spDataLoader->K()));
        std::shared_ptr<superVO::keyFrame> spKeyFrame(new superVO::keyFrame(spTrtEngine, spMap, spDataLoader->K(), spDataLoader->D(), spDataLoader->extrisic(), UNDISTORT));
        std::shared_ptr<superVO::frame> spFrame(new superVO::frame(spKeyFrame, spDataLoader->K(), spDataLoader->D(), UNDISTORT));
        std::shared_ptr<superVO::optimizer> spOptimizer(new superVO::optimizer(spMap));
        std::shared_ptr<superVO::mapDrawer> spDrawer(new superVO::mapDrawer()); 
        superVO::tracker spTracker(spTrtEngine, spDataLoader, spFrame, spKeyFrame, spMap, spOptimizer, spDrawer);
    }
    return 0;
}