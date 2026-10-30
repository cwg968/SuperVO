#include <trtEngine.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>
int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr << "Usage: supervo_build_engine WEIGHTS ENGINE WIDTH HEIGHT\n";
            return 1;
        }
        size_t used = 0;
        int width = std::stoi(argv[3], &used);
        if (used != std::string(argv[3]).size())
            throw std::runtime_error("Invalid width");
        int height = std::stoi(argv[4], &used);
        if (used != std::string(argv[4]).size() || width < 8 || height < 8 || width % 8 ||
            height % 8)
            throw std::runtime_error("Engine dimensions must be positive multiples of eight");
        if (!std::filesystem::is_regular_file(argv[1]))
            throw std::runtime_error("Weights do not exist");
        std::filesystem::path output(argv[2]);
        if (std::filesystem::exists(output))
            throw std::runtime_error("Engine output already exists");
        if (!output.parent_path().empty())
            std::filesystem::create_directories(output.parent_path());
        superVO::trtEngine engine(height, width, 1, output.string());
        engine.buildTRTEngine(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
