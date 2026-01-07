#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>

namespace fs = std::filesystem;

// Simple WAV writer (16-bit PCM)
bool save_wav16(const std::string &path, const std::vector<float> &samples, int sample_rate = 22050) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t data_bytes = int32_t(samples.size() * 2);
    int32_t chunk_size = 36 + data_bytes;
    auto write_le = [&](auto v, size_t bytes) {
        for (size_t i = 0; i < bytes; ++i) f.put(char((v >> (8 * i)) & 0xFF));
    };
    f.write("RIFF",4);
    write_le(chunk_size,4);
    f.write("WAVE",4);
    f.write("fmt ",4);
    write_le(16,4);            // subchunk1 size
    write_le(1,2);             // PCM
    write_le(1,2);             // channels
    write_le(sample_rate,4);
    write_le(sample_rate*2,4); // byte rate
    write_le(2,2);             // block align
    write_le(16,2);            // bits per sample
    f.write("data",4);
    write_le(data_bytes,4);
    for (float s : samples) {
        float clamped = std::max(-1.0f, std::min(1.0f, s));
        int16_t v = int16_t(std::lround(clamped * 32767.0f));
        write_le(uint16_t(v),2);
    }
    return true;
}

// Simple PPM writer (RGB uint8)
bool save_ppm(const std::string &path, int w, int h, const std::vector<float> &rgb) {
    if ((int)rgb.size() < w*h*3) return false;
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int i=0;i<w*h*3;++i) {
        float c = rgb[i];
        int v = int(std::round(std::max(0.0f, std::min(1.0f, c)) * 255.0f));
        f.put(char(v));
    }
    return true;
}

// Read token ids from a whitespace-separated text file (fallback to precomputed token dumps)
std::vector<int64_t> read_tokens_txt(const std::string &path) {
    std::vector<int64_t> tokens;
    std::ifstream f(path);
    if (!f) return tokens;
    int64_t v;
    while (f >> v) tokens.push_back(v);
    return tokens;
}

// Run ONNX model with int64 input (token ids) and return float output vector
std::vector<float> run_onnx_int64(const std::string &model_path, const std::vector<int64_t> &input_ids, const std::string &input_name, const std::string &output_name) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "gen");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    Ort::Session session(env, model_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<int64_t> input_shape = {1, (int64_t)input_ids.size()};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(mem_info, const_cast<int64_t*>(input_ids.data()), input_ids.size(), input_shape.data(), input_shape.size());

    const char* in_names[] = { input_name.c_str() };
    const char* out_names[] = { output_name.c_str() };
    auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);
    Ort::Value &out0 = outputs.front();
    size_t nelem = out0.GetTensorTypeAndShapeInfo().GetElementCount();
    float* pdata = out0.GetTensorMutableData<float>();
    return std::vector<float>(pdata, pdata + nelem);
}

int main(int argc, char** argv) {
    std::string flux_onnx = argc>1?argv[1]:"flux1_schnell.onnx";
    std::string ldm2_onnx = argc>2?argv[2]:"ldm2.onnx";
    std::string bark_onnx = argc>3?argv[3]:"bark.onnx";

    // Prefer precomputed token id text files (produced by export scripts)
    std::string bark_ids_txt = "bark_token_ids.txt";
    std::string flux_ids_txt = "flux_token_ids.txt";
    std::string ldm2_ids_txt = "ldm2_token_ids.txt";

    auto bark_tokens = read_tokens_txt(bark_ids_txt);
    auto flux_tokens = read_tokens_txt(flux_ids_txt);
    auto ldm2_tokens = read_tokens_txt(ldm2_ids_txt);

    if (bark_tokens.empty() && flux_tokens.empty() && ldm2_tokens.empty()) {
        std::cerr << "No token id files found (bark, flux, ldm2); aborting." << std::endl;
        return 1;
    }

    // Run Bark ONNX (expecting input name "input_ids" and output "audio" - adjust as needed)
    if (!bark_tokens.empty()) {
        std::cout << "Running Bark ONNX..." << std::endl;
        try {
            auto bark_out = run_onnx_int64(bark_onnx, bark_tokens, "input_ids", "audio");
            if (!bark_out.empty()) {
                save_wav16("bark_voice.wav", bark_out, 22050);
                std::cout << "Saved bark_voice.wav (" << bark_out.size() << " samples)" << std::endl;
            } else {
                std::cerr << "Bark ONNX returned empty output." << std::endl;
            }
        } catch (const std::exception &e) {
            std::cerr << "Bark ONNX failed: " << e.what() << std::endl;
        }
    }

    // Run LDM2 ONNX (music)
    if (!ldm2_tokens.empty()) {
        std::cout << "Running LDM2 ONNX..." << std::endl;
        try {
            auto audio_out = run_onnx_int64(ldm2_onnx, ldm2_tokens, "input_ids", "audio");
            if (!audio_out.empty()) {
                save_wav16("ldm2_music.wav", audio_out, 44100);
                std::cout << "Saved ldm2_music.wav (" << audio_out.size() << " samples)" << std::endl;
            }
        } catch (const std::exception &e) {
            std::cerr << "LDM2 ONNX failed: " << e.what() << std::endl;
        }
    }

    // Run FLUX ONNX (image)
    if (!flux_tokens.empty()) {
        std::cout << "Running FLUX ONNX..." << std::endl;
        try {
            auto img_out = run_onnx_int64(flux_onnx, flux_tokens, "input_ids", "output");
            if (!img_out.empty()) {
                size_t ne = img_out.size();
                int c = 3;
                int pixels = int(ne / c);
                int side = int(std::round(std::sqrt((double)pixels)));
                if (side*side*c == (int)ne) {
                    save_ppm("flux_image.ppm", side, side, img_out);
                    std::cout << "Saved flux_image.ppm (" << side << "x" << side << ")\n";
                } else {
                    std::ofstream f("flux_image.raw", std::ios::binary);
                    f.write(reinterpret_cast<const char*>(img_out.data()), img_out.size()*sizeof(float));
                    std::cout << "Saved flux_image.raw (raw floats, shape unknown)\n";
                }
            }
        } catch (const std::exception &e) {
            std::cerr << "FLUX ONNX failed: " << e.what() << std::endl;
        }
    }

    std::cout << "Done." << std::endl;
    return 0;
}
