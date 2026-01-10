#include <onnxruntime_cxx_api.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

// WAV writer (16-bit PCM)
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
    write_le(16,4);
    write_le(1,2);
    write_le(1,2);
    write_le(sample_rate,4);
    write_le(sample_rate*2,4);
    write_le(2,2);
    write_le(16,2);
    f.write("data",4);
    write_le(data_bytes,4);
    for (float s : samples) {
        float clamped = std::max(-1.0f, std::min(1.0f, s));
        int16_t v = int16_t(std::lround(clamped * 32767.0f));
        write_le(uint16_t(v),2);
    }
    return true;
}

// Text to character IDs (simple mapping)
std::vector<int64_t> text_to_char_ids(const std::string &text) {
    std::string chars = " abcdefghijklmnopqrstuvwxyz";
    std::vector<int64_t> ids;
    for (char c : text) {
        char lower = std::tolower(c);
        auto pos = chars.find(lower);
        ids.push_back(pos != std::string::npos ? int64_t(pos) : 0);
    }
    return ids;
}

// Generate TTS audio (C++ implementation matching Python)
std::vector<float> generate_tts_cpp(const std::string &text, int sample_rate = 22050) {
    auto char_ids = text_to_char_ids(text);
    
    float char_duration = 0.15f;
    int samples_per_char = int(sample_rate * char_duration);
    std::vector<float> audio;
    audio.reserve(char_ids.size() * samples_per_char);
    
    float phase = 0.0f;
    for (int64_t char_id : char_ids) {
        float freq = (char_id == 0) ? 0.0f : (200.0f + char_id * 30.0f);
        
        for (int i = 0; i < samples_per_char; ++i) {
            float sample = (freq > 0) ? std::sin(phase) * 0.3f : 0.0f;
            audio.push_back(sample);
            
            phase += 2.0f * 3.14159265f * freq / sample_rate;
            if (phase > 2.0f * 3.14159265f) {
                phase -= 2.0f * 3.14159265f;
            }
        }
    }
    
    return audio;
}

// Run TTS using ONNX Runtime
std::vector<float> run_tts_onnx(const std::string &model_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TTS");
    Ort::SessionOptions opts;
    Ort::Session session(env, model_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions allocator;
    
    auto out_name = session.GetOutputNameAllocated(0, allocator);
    const char *out_names[] = {out_name.get()};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, nullptr, nullptr, 0, out_names, 1);
    
    auto &out = outputs.front();
    size_t n = out.GetTensorTypeAndShapeInfo().GetElementCount();
    float *data = out.GetTensorMutableData<float>();
    
    return std::vector<float>(data, data + n);
}

// Run TTS using LibTorch
std::vector<float> run_tts_libtorch(const std::string &model_path, const std::string &text) {
    torch::jit::script::Module model = torch::jit::load(model_path);
    model.eval();
    
    auto char_ids = text_to_char_ids(text);
    torch::Tensor input = torch::tensor(char_ids, torch::kLong);
    
    torch::Tensor output = model.forward({input}).toTensor();
    output = output.to(torch::kCPU).contiguous();
    
    std::vector<float> audio(output.data_ptr<float>(), 
                             output.data_ptr<float>() + output.numel());
    return audio;
}

int main(int argc, char** argv) {
    std::string text = (argc > 1) ? argv[1] : "hello world";
    std::cout << "=== C++ Minimal TTS Test ===" << std::endl;
    std::cout << "Input text: '" << text << "'\n" << std::endl;
    
    const int sample_rate = 22050;
    
    // Method 1: Pure C++ implementation
    std::cout << "--- Method 1: Pure C++ implementation ---" << std::endl;
    try {
        auto audio_cpp = generate_tts_cpp(text, sample_rate);
        if (save_wav16("cpp_tts_native.wav", audio_cpp, sample_rate)) {
            std::cout << "✓ Saved cpp_tts_native.wav (" << audio_cpp.size() 
                      << " samples, " << float(audio_cpp.size())/sample_rate << "s)" << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "✗ C++ TTS failed: " << e.what() << std::endl;
    }
    
    // Method 2: ONNX Runtime
    std::cout << "\n--- Method 2: ONNX Runtime ---" << std::endl;
    try {
        auto audio_onnx = run_tts_onnx("tts_model.onnx");
        if (save_wav16("cpp_tts_onnx.wav", audio_onnx, sample_rate)) {
            std::cout << "✓ Saved cpp_tts_onnx.wav (" << audio_onnx.size() 
                      << " samples, " << float(audio_onnx.size())/sample_rate << "s)" << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "✗ ONNX TTS failed: " << e.what() << std::endl;
    }
    
    // Method 3: LibTorch
    std::cout << "\n--- Method 3: LibTorch (TorchScript) ---" << std::endl;
    try {
        auto audio_torch = run_tts_libtorch("tts_model.pt", text);
        if (save_wav16("cpp_tts_libtorch.wav", audio_torch, sample_rate)) {
            std::cout << "✓ Saved cpp_tts_libtorch.wav (" << audio_torch.size() 
                      << " samples, " << float(audio_torch.size())/sample_rate << "s)" << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "✗ LibTorch TTS failed: " << e.what() << std::endl;
    }
    
    std::cout << "\n=== C++ TTS Complete ===" << std::endl;
    return 0;
}
