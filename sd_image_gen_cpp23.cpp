// Real Stable Diffusion Image Generation from C++23
// Uses stable-diffusion.cpp (like llama.cpp for images)
// NO DUMMY OUTPUT - Real neural image generation!

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <print>
#include <string>
#include <vector>
#include <expected>
#include <format>
#include <chrono>

namespace fs = std::filesystem;

struct SDConfig {
    std::string model_path;
    std::string output_path;
    std::string prompt;
    int width{512};
    int height{512};
    int steps{20};
    float cfg_scale{7.0f};
    int seed{42};
    int threads{-1};  // -1 = use all CPU cores (Metal will use GPU automatically)
    bool vae_tiling{true};  // Enable VAE tiling for lower memory
};

class StableDiffusionEngine {
public:
    static auto generate_image(const SDConfig& config) -> std::expected<std::string, std::string> {
        // Build command for sd-cli
        std::string cmd = "/tmp/stable-diffusion.cpp/build/bin/sd-cli";
        
        // Check if sd-cli exists
        if (!fs::exists(cmd)) {
            return std::unexpected("sd-cli not found. Please build stable-diffusion.cpp first.");
        }
        
        // Check if model exists
        if (!fs::exists(config.model_path)) {
            return std::unexpected(std::format("Model not found: {}", config.model_path));
        }
        
        // Build full command with arguments  
        std::string full_cmd = std::format(
            "{} -m '{}' -p '{}' -o '{}' --width {} --height {} --steps {} --cfg-scale {} --seed {} -t {}{} --clip-on-cpu",
            cmd,
            config.model_path,
            config.prompt,
            config.output_path,
            config.width,
            config.height,
            config.steps,
            config.cfg_scale,
            config.seed,
            config.threads,
            config.vae_tiling ? " --vae-tiling" : ""
        );
        
        std::println("Running: {}", full_cmd);
        std::println("\n🎨 Generating image with Stable Diffusion...");
        std::println("📝 Prompt: '{}'", config.prompt);
        std::println("📐 Resolution: {}x{}", config.width, config.height);
        std::println("🔧 Steps: {}, CFG Scale: {}", config.steps, config.cfg_scale);
        std::println("⚡ Acceleration: Metal GPU for UNet/VAE + CPU for CLIP (text encoding)");
        std::println("💾 VAE Tiling: {} (lower memory usage)", config.vae_tiling ? "enabled" : "disabled");
        
        auto start = std::chrono::steady_clock::now();
        
        // Execute the command
        int result = std::system(full_cmd.c_str());
        
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        
        if (result != 0) {
            return std::unexpected(std::format("sd-cli failed with code: {}", result));
        }
        
        // Check if output was created
        if (!fs::exists(config.output_path)) {
            return std::unexpected("Image file was not created");
        }
        
        auto size = fs::file_size(config.output_path);
        std::println("\n✅ Image generated successfully!");
        std::println("💾 Saved to: {}", config.output_path);
        std::println("📏 Size: {} KB", size / 1024);
        std::println("⏱️  Time: {} seconds", duration.count());
        
        return config.output_path;
    }
};

int main() {
    std::println("=== Stable Diffusion C++23 Image Generator ===\n");
    
    // Define test prompts with various scenes
    std::vector<SDConfig> configs = {
        {
            .model_path = "/Users/anders/models/sd/stable-diffusion-v1-5-Q8_0.gguf",
            .output_path = "cpp23_sd_cat.png",
            .prompt = "a fluffy orange cat sitting on a wooden desk, photorealistic, high quality",
            .width = 512,
            .height = 512,
            .steps = 20,
            .cfg_scale = 7.0f,
            .seed = 42
        },
        {
            .model_path = "/Users/anders/models/sd/stable-diffusion-v1-5-Q8_0.gguf",
            .output_path = "cpp23_sd_astronaut.png",
            .prompt = "an astronaut riding a horse on mars, digital art, trending on artstation",
            .width = 512,
            .height = 512,
            .steps = 20,
            .cfg_scale = 7.5f,
            .seed = 123
        },
        {
            .model_path = "/Users/anders/models/sd/stable-diffusion-v1-5-Q8_0.gguf",
            .output_path = "cpp23_sd_dalek.png",
            .prompt = "a chrome dalek robot in a retro 1960s living room, vintage photograph",
            .width = 512,
            .height = 512,
            .steps = 25,
            .cfg_scale = 8.0f,
            .seed = 999
        }
    };
    
    int success_count = 0;
    int total = configs.size();
    
    for (const auto& config : configs) {
        std::println("\n{}", std::string(60, '='));
        
        auto result = StableDiffusionEngine::generate_image(config);
        
        if (result) {
            success_count++;
            std::println("✓ Generated: {}", *result);
        } else {
            std::println("✗ Error: {}", result.error());
        }
    }
    
    std::println("\n{}", std::string(60, '='));
    std::println("\n📊 Summary: {} of {} images generated successfully", success_count, total);
    
    if (success_count == total) {
        std::println("\n🎉 All images generated using REAL Stable Diffusion neural network!");
        std::println("💡 NO DUMMY OUTPUT - These are actual SD 1.5 generations!");
        return 0;
    } else {
        std::println("\n⚠️  Some images failed to generate");
        return 1;
    }
}
