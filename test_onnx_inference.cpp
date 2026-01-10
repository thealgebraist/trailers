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

// Read token ids from text file
std::vector<int64_t> read_tokens_txt(const std::string &path) {
    std::vector<int64_t> tokens;
    std::ifstream f(path);
    if (!f) return tokens;
    int64_t v;
    while (f >> v) tokens.push_back(v);
    return tokens;
}

// Run ONNX tokenizer model (no inputs, outputs constant tokens)
std::vector<int64_t> run_onnx_tokenizer(const std::string &model_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "tokenizer");
    Ort::SessionOptions opts;
    Ort::Session session(env, model_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions allocator;
    
    auto out_name = session.GetOutputNameAllocated(0, allocator);
    const char *out_names[] = {out_name.get()};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, nullptr, nullptr, 0, out_names, 1);
    
    auto &out = outputs.front();
    size_t n = out.GetTensorTypeAndShapeInfo().GetElementCount();
    auto *data = out.GetTensorMutableData<int64_t>();
    
    return std::vector<int64_t>(data, data + n);
}

int main() {
    std::cout << "=== ONNX Inference Test ===" << std::endl;
    
    // Test 1: Load and run tokenizers
    std::cout << "\n--- Running Tokenizers ---" << std::endl;
    
    try {
        auto flux_tokens = run_onnx_tokenizer("flux_tokenizer.onnx");
        std::cout << "✓ FLUX tokenizer: " << flux_tokens.size() << " tokens" << std::endl;
        std::cout << "  First 10 tokens: ";
        for (size_t i = 0; i < std::min(size_t(10), flux_tokens.size()); ++i) {
            std::cout << flux_tokens[i] << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "✗ FLUX tokenizer failed: " << e.what() << std::endl;
    }
    
    try {
        auto ldm2_tokens = run_onnx_tokenizer("ldm2_tokenizer.onnx");
        std::cout << "✓ LDM2 tokenizer: " << ldm2_tokens.size() << " tokens" << std::endl;
        std::cout << "  First 10 tokens: ";
        for (size_t i = 0; i < std::min(size_t(10), ldm2_tokens.size()); ++i) {
            std::cout << ldm2_tokens[i] << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "✗ LDM2 tokenizer failed: " << e.what() << std::endl;
    }
    
    try {
        auto bark_tokens = run_onnx_tokenizer("bark_tokenizer.onnx");
        std::cout << "✓ Bark tokenizer: " << bark_tokens.size() << " tokens" << std::endl;
        std::cout << "  First 10 tokens: ";
        for (size_t i = 0; i < std::min(size_t(10), bark_tokens.size()); ++i) {
            std::cout << bark_tokens[i] << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "✗ Bark tokenizer failed: " << e.what() << std::endl;
    }
    
    // Test 2: Generate synthetic audio from tokens (simulate inference)
    std::cout << "\n--- Simulating Audio Generation ---" << std::endl;
    
    auto tokens = read_tokens_txt("flux_token_ids.txt");
    if (!tokens.empty()) {
        // Generate synthetic audio based on token pattern with proper phase continuity
        const int sample_rate = 22050;
        const float duration = 2.0f;
        const float note_duration = 0.2f; // Each note lasts 200ms
        const int samples_per_note = int(sample_rate * note_duration);
        std::vector<float> audio;
        
        float phase = 0.0f;
        for (int i = 0; i < int(sample_rate * duration); ++i) {
            // Change frequency every note_duration seconds based on tokens
            int note_index = i / samples_per_note;
            float freq = 220.0f + (tokens[note_index % tokens.size()] % 12) * 20.0f;
            
            // Generate sample with continuous phase
            float sample = std::sin(phase) * 0.5f;
            audio.push_back(sample);
            
            // Advance phase
            phase += 2.0f * 3.14159265f * freq / sample_rate;
            if (phase > 2.0f * 3.14159265f) {
                phase -= 2.0f * 3.14159265f;
            }
        }
        
        if (save_wav16("inference_audio.wav", audio, sample_rate)) {
            std::cout << "✓ Generated inference_audio.wav (" << audio.size() << " samples)" << std::endl;
        }
    }
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    std::cout << "\nNote: This demonstrates the ONNX tokenizer loading." << std::endl;
    std::cout << "For full inference, you would need complete exported models (TTS, image gen, etc.)" << std::endl;
    
    return 0;
}
