#include <cstdlib>
#include <iostream>
#include <string>

int run_cmd(const std::string &cmd) {
    std::cout << "> " << cmd << std::endl;
    int rc = std::system(cmd.c_str());
    if (rc != 0) std::cerr << "Command failed with code: " << rc << std::endl;
    return rc;
}

int main() {
    // Ensure tokenizers exist
    if (run_cmd("python3 export_tokenizer_torchscript.py") != 0) return 1;
    if (run_cmd("python3 export_tokenizer_onnx.py") != 0) {
        std::cerr << "ONNX tokenizer export failed or onnx not installed; continuing with TorchScript tokenizers only." << std::endl;
    }

    // Run the existing Python asset generator script
    // This will generate voice, images, and music according to generate_absurd_trailers.py
    int rc = run_cmd("python3 generate_absurd_trailers.py");
    if (rc != 0) {
        std::cerr << "Asset generation script failed." << std::endl;
        return rc;
    }

    std::cout << "Asset generation completed (launched from C++ orchestrator)." << std::endl;
    return 0;
}
