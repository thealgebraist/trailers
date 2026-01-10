#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#if __has_include(<torch/script.h>)
#  define HAVE_TORCH 1
#  include <torch/script.h>
#else
#  define HAVE_TORCH 0
#endif

#if __has_include(<onnxruntime_cxx_api.h>)
#  define HAVE_ORT 1
#  include <onnxruntime_cxx_api.h>
#else
#  define HAVE_ORT 0
#endif

#if HAVE_TORCH
bool save_wav16(const std::string &path, const std::vector<float> &samples, int sample_rate = 24000) {
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
#endif

int run_cmd(const std::string &cmd) {
    std::cout << "> " << cmd << std::endl;
    int rc = std::system(cmd.c_str());
    if (rc != 0) std::cerr << "Command failed with code: " << rc << std::endl;
    return rc;
}

#if HAVE_TORCH
bool run_libtorch_tokenizer(const std::string &path) {
    try {
        torch::jit::script::Module module = torch::jit::load(path);
        auto out = module.forward({torch::zeros({1})}).toTensor().to(torch::kCPU);
        auto flat = out.flatten();
        std::vector<int64_t> tokens(flat.data_ptr<int64_t>(), flat.data_ptr<int64_t>() + flat.numel());
        std::cout << "Loaded TorchScript tokenizer (" << tokens.size() << " tokens) using libtorch." << std::endl;
        return true;
    } catch (const std::exception &e) {
        std::cerr << "LibTorch run failed: " << e.what() << std::endl;
        return false;
    }
}

bool generate_wav_libtorch(const std::string &out_path) {
    try {
        const int sample_rate = 24000;
        const float freq = 440.0f;
        const float two_pi = 6.28318530717958647692f;
        auto t = torch::linspace(0.0f, 1.0f, sample_rate);
        auto wave = torch::sin(two_pi * freq * t);
        std::vector<float> samples(wave.data_ptr<float>(), wave.data_ptr<float>() + wave.numel());
        if (!save_wav16(out_path, samples, sample_rate)) {
            std::cerr << "Failed to write wav file." << std::endl;
            return false;
        }
        std::cout << "Generated " << out_path << " using libtorch sine model (" << samples.size() << " samples)." << std::endl;
        return true;
    } catch (const std::exception &e) {
        std::cerr << "LibTorch audio generation failed: " << e.what() << std::endl;
        return false;
    }
}
#else
bool run_libtorch_tokenizer(const std::string &) {
    std::cerr << "LibTorch headers not available; rebuild with libtorch to use this backend." << std::endl;
    return false;
}

bool generate_wav_libtorch(const std::string &) {
    std::cerr << "LibTorch headers not available; rebuild with libtorch to use this backend." << std::endl;
    return false;
}
#endif

#if HAVE_ORT
bool run_onnx_tokenizer(const std::string &path) {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "tokenizer");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        Ort::Session session(env, path.c_str(), opts);
        Ort::AllocatorWithDefaultOptions allocator;
        auto out_name = session.GetOutputNameAllocated(0, allocator);
        const char *out_names[] = {out_name.get()};
        auto outputs = session.Run(Ort::RunOptions{nullptr}, nullptr, nullptr, 0, out_names, 1);
        auto &out = outputs.front();
        auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
        size_t n = out.GetTensorTypeAndShapeInfo().GetElementCount();
        auto *data = out.GetTensorMutableData<int64_t>();
        std::cout << "Loaded ONNX tokenizer (" << n << " tokens";
        if (shape.size() == 2) std::cout << ", shape=[" << shape[0] << "," << shape[1] << "]";
        std::cout << ")." << std::endl;
        return n > 0;
    } catch (const std::exception &e) {
        std::cerr << "ONNX Runtime run failed: " << e.what() << std::endl;
        return false;
    }
}
#else
bool run_onnx_tokenizer(const std::string &) {
    std::cerr << "ONNX Runtime headers not available; rebuild with onnxruntime to use this backend." << std::endl;
    return false;
}
#endif

int main(int argc, char **argv) {
    // Make sure we run from the executable's directory so relative scripts/data resolve.
    std::error_code ec;
    const auto exe_dir = std::filesystem::absolute(argc > 0 ? argv[0] : "").parent_path();
    std::filesystem::current_path(exe_dir, ec);
    if (ec) {
        std::cerr << "Failed to change working directory to executable location: " << ec.message() << std::endl;
        return 1;
    }

    const std::string backend = (argc > 1) ? argv[1] : "python";

    if (backend == "python") {
        if (run_cmd("python3 export_tokenizer_torchscript.py") != 0) return 1;
        if (run_cmd("python3 export_tokenizer_onnx.py") != 0) {
            std::cerr << "ONNX tokenizer export failed or onnx not installed; continuing with TorchScript tokenizers only." << std::endl;
        }
        int rc = run_cmd("python3 generate_absurd_trailers.py");
        if (rc != 0) {
            std::cerr << "Asset generation script failed." << std::endl;
            return rc;
        }
        std::cout << "Asset generation completed (launched from C++ orchestrator)." << std::endl;
        return 0;
    } else if (backend == "torch" || backend == "libtorch") {
        if (run_cmd("python3 export_tokenizer_torchscript.py") != 0) return 1;
        bool tok_ok = run_libtorch_tokenizer("bark_tokenizer.pt");
        bool wav_ok = generate_wav_libtorch("libtorch_sine.wav");
        return (tok_ok && wav_ok) ? 0 : 1;
    } else if (backend == "onnx") {
        if (run_cmd("python3 export_tokenizer_onnx.py") != 0) return 1;
        bool ok = run_onnx_tokenizer("bark_tokenizer.onnx");
        return ok ? 0 : 1;
    } else {
        std::cerr << "Unknown backend '" << backend << "'. Use one of: python (default), torch, onnx." << std::endl;
        return 1;
    }
}
