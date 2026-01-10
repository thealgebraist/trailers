// Simple ALF assets renderer (C++23)
// - Copies images from assets_soviet_alf/images to rendered frames
// - Copies voice/music files to rendered/voice and rendered/music
// - Calls ffmpeg to assemble a slideshow video if ffmpeg is available

#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <random>
#include <fstream>
#include <cmath>

namespace fs = std::filesystem;

int main() {
    const fs::path assets = "assets_soviet_alf";
    const fs::path images_dir = assets / "images";
    const fs::path voice_dir = assets / "voice";
    const fs::path music_dir = assets / "music";
    const fs::path out_dir = "rendered";
    const fs::path out_frames = out_dir / "frames";
    const fs::path out_voice = out_dir / "voice";
    const fs::path out_music = out_dir / "music";

    if (!fs::exists(assets)) {
        std::cerr << "Error: assets folder not found: " << assets << "\n";
        return 1;
    }

    fs::create_directories(out_frames);
    fs::create_directories(out_voice);
    fs::create_directories(out_music);

    // Collect image files
    std::vector<fs::path> images;
    if (fs::exists(images_dir)) {
        for (auto &p: fs::directory_iterator(images_dir)) {
            if (!p.is_regular_file()) continue;
            auto ext = p.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                images.push_back(p.path());
            }
        }
    }

    if (images.empty()) {
        std::cerr << "No images found in " << images_dir << "\n";
    } else {
        std::sort(images.begin(), images.end());
        // copy to frames as frame000.png ...
        int idx = 0;
        for (auto &p: images) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "frame%03d.png", idx);
            fs::path dest = out_frames / buf;
            try {
                fs::copy_file(p, dest, fs::copy_options::overwrite_existing);
                std::cout << "Copied: " << p << " -> " << dest << "\n";
            } catch (const std::exception &e) {
                std::cerr << "Copy failed: " << e.what() << "\n";
            }
            idx++;
        }
    }

    // Copy voice files
    if (fs::exists(voice_dir)) {
        for (auto &p: fs::directory_iterator(voice_dir)) {
            if (!p.is_regular_file()) continue;
            auto ext = p.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".wav" || ext == ".mp3") {
                fs::path dest = out_voice / p.path().filename();
                try { fs::copy_file(p, dest, fs::copy_options::overwrite_existing); }
                catch(...){}
                std::cout << "Copied voice: " << dest << "\n";
            }
        }
    }

    // Copy music files
    fs::path music_file;
    if (fs::exists(music_dir)) {
        for (auto &p: fs::directory_iterator(music_dir)) {
            if (!p.is_regular_file()) continue;
            auto ext = p.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".wav" || ext == ".mp3") {
                fs::path dest = out_music / p.path().filename();
                try { fs::copy_file(p, dest, fs::copy_options::overwrite_existing); }
                catch(...){}
                std::cout << "Copied music: " << dest << "\n";
                music_file = dest;
            }
        }
    }

    // If no images found, synthesize placeholder PPM frames
    std::string frame_ext = ".png";
    if (images.empty()) {
        frame_ext = ".ppm";
        std::cout << "Generating placeholder frames...\n";
        const int WIDTH = 512, HEIGHT = 512;
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> noise(-30,30);

        for (int i = 0; i < 6; ++i) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "frame%03d.ppm", i);
            fs::path dest = out_frames / buf;
            std::ofstream f(dest, std::ios::binary);
            if (!f) continue;
            f << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";

            for (int y = 0; y < HEIGHT; ++y) {
                for (int x = 0; x < WIDTH; ++x) {
                    // base gray with vignette
                    float nx = (x - WIDTH/2.0f) / (WIDTH/2.0f);
                    float ny = (y - HEIGHT/2.0f) / (HEIGHT/2.0f);
                    float r = std::sqrt(nx*nx + ny*ny);
                    int base = static_cast<int>(200 - 120 * r);
                    base += noise(rng);
                    if (base < 0) base = 0; if (base > 255) base = 255;

                    // simple bright spot moving across frames
                    float cx = WIDTH * (0.2f + 0.12f * i);
                    float cy = HEIGHT * 0.45f;
                    float dist = std::hypot(x - cx, y - cy);
                    if (dist < 60) base = std::min(255, base + static_cast<int>(80 * (1 - dist/60)));

                    unsigned char pix = static_cast<unsigned char>(base);
                    f.put(pix); f.put(pix); f.put(pix);
                }
            }
            f.close();
            std::cout << "Created placeholder: " << dest << "\n";
        }
    }

    // Check for ffmpeg
    if (std::system("which ffmpeg > /dev/null 2>&1") != 0) {
        std::cout << "ffmpeg not found on PATH — frames copied to 'rendered/frames'.\n";
        return 0;
    }

    // Build ffmpeg command
    // framerate: 1 fps per image (can be adjusted)
    std::string ff_cmd;
    if (!fs::exists(out_frames) || fs::is_empty(out_frames)) {
        std::cerr << "No frames to assemble.\n";
        return 0;
    }

    std::string pattern = "frame%03d" + frame_ext;
    if (!music_file.empty()) {
        ff_cmd = std::string("ffmpeg -y -framerate 1 -i ") + out_frames.string() + "/" + pattern + " -i " + music_file.string() + " -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest rendered/alf_video.mp4";
    } else {
        ff_cmd = std::string("ffmpeg -y -framerate 1 -i ") + out_frames.string() + "/" + pattern + " -c:v libx264 -pix_fmt yuv420p rendered/alf_video.mp4";
    }

    std::cout << "Running: " << ff_cmd << "\n";
    int rc = std::system(ff_cmd.c_str());
    if (rc != 0) {
        std::cerr << "ffmpeg failed with code " << rc << "\n";
        return 1;
    }

    std::cout << "Video rendered: rendered/alf_video.mp4\n";
    return 0;
}
