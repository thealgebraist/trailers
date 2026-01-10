#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

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
bool save_ppm(const std::string &path, int w, int h, const std::vector<uint8_t> &rgb) {
    if ((int)rgb.size() < w*h*3) return false;
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P6\n" << w << " " << h << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()), w*h*3);
    return true;
}

int main() {
    std::cout << "=== C++ ONNX Runtime Test ===" << std::endl;
    
    // Test 1: ONNX Runtime initialization
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        std::cout << "✓ ONNX Runtime initialized successfully" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "✗ ONNX Runtime init failed: " << e.what() << std::endl;
        return 1;
    }
    
    // Test 2: Generate synthetic audio (440Hz sine wave)
    const int sample_rate = 22050;
    const float duration = 1.0f;
    const int num_samples = int(sample_rate * duration);
    std::vector<float> audio_samples;
    audio_samples.reserve(num_samples);
    
    for (int i = 0; i < num_samples; ++i) {
        float t = float(i) / float(sample_rate);
        float sample = std::sin(2.0f * 3.14159265f * 440.0f * t) * 0.5f;
        audio_samples.push_back(sample);
    }
    
    if (save_wav16("test_output_440hz.wav", audio_samples, sample_rate)) {
        std::cout << "✓ Generated test_output_440hz.wav (" << audio_samples.size() 
                  << " samples, " << duration << "s)" << std::endl;
    } else {
        std::cerr << "✗ Failed to write WAV file" << std::endl;
    }
    
    // Test 3: Generate synthetic image (RGB gradient)
    const int img_width = 256;
    const int img_height = 256;
    std::vector<uint8_t> img_data;
    img_data.reserve(img_width * img_height * 3);
    
    for (int y = 0; y < img_height; ++y) {
        for (int x = 0; x < img_width; ++x) {
            img_data.push_back(uint8_t(x));        // R
            img_data.push_back(uint8_t(y));        // G
            img_data.push_back(uint8_t((x+y)/2));  // B
        }
    }
    
    if (save_ppm("test_output_gradient.ppm", img_width, img_height, img_data)) {
        std::cout << "✓ Generated test_output_gradient.ppm (" << img_width << "x" 
                  << img_height << " RGB)" << std::endl;
    } else {
        std::cerr << "✗ Failed to write PPM file" << std::endl;
    }
    
    // Test 4: Try loading an ONNX model if available
    std::cout << "\n--- Testing ONNX Model Loading ---" << std::endl;
    const char* test_models[] = {"flux_tokenizer.onnx", "ldm2_tokenizer.onnx"};
    
    for (const char* model_path : test_models) {
        try {
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "model_test");
            Ort::SessionOptions opts;
            Ort::Session session(env, model_path, opts);
            
            size_t num_inputs = session.GetInputCount();
            size_t num_outputs = session.GetOutputCount();
            
            std::cout << "✓ Loaded " << model_path << std::endl;
            std::cout << "  Inputs: " << num_inputs << ", Outputs: " << num_outputs << std::endl;
            
            // Get input/output names
            Ort::AllocatorWithDefaultOptions allocator;
            for (size_t i = 0; i < num_inputs; ++i) {
                auto name = session.GetInputNameAllocated(i, allocator);
                auto type_info = session.GetInputTypeInfo(i);
                std::cout << "  Input[" << i << "]: " << name.get() << std::endl;
            }
            for (size_t i = 0; i < num_outputs; ++i) {
                auto name = session.GetOutputNameAllocated(i, allocator);
                std::cout << "  Output[" << i << "]: " << name.get() << std::endl;
            }
            
        } catch (const std::exception &e) {
            std::cout << "✗ Failed to load " << model_path << ": " << e.what() << std::endl;
        }
    }
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}
