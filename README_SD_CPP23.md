# Stable Diffusion from C++23 🎨

Real neural image generation using **stable-diffusion.cpp** and modern C++23!

## What We Have

### ✅ Components Ready

1. **stable-diffusion.cpp** - Built from source
   - Location: `/tmp/stable-diffusion.cpp/build/bin/sd-cli`
   - Version: master-467-0e52afc
   - Binary size: 23MB
   - Features: Full SD inference with GGUF model support

2. **C++23 Wrapper** - `sd_image_gen_cpp23.cpp`
   - Compiled binary: 121KB
   - Modern C++ features: `std::expected`, `std::format`, `std::println`
   - Prompts configured:
     * Fluffy orange cat on wooden desk
     * Astronaut riding horse on Mars
     * Chrome Dalek robot in 1960s living room

3. **Model** - second-state/stable-diffusion-v1-5-GGUF
   - File: `stable-diffusion-v1-5-Q8_0.gguf`
   - Size: ~1.7GB (Q8 quantization)
   - Status: **Downloading** (started fresh, in progress)
   - Quality: 8-bit quantization (excellent quality/size ratio)

## How It Works

The C++23 code calls `sd-cli` with:
```cpp
std::string cmd = std::format(
    "{} -m '{}' -p '{}' -o '{}' --width {} --height {} --steps {} --cfg-scale {} --seed {}",
    "/tmp/stable-diffusion.cpp/build/bin/sd-cli",
    model_path,  // GGUF model
    prompt,      // Text description
    output_path, // output.png
    width, height, steps, cfg_scale, seed
);
```

## Run Once Model Downloads

```bash
cd /Users/anders/projects/dalek-comes-home
./sd_image_gen_cpp23
```

This will generate 3 images:
- `cpp23_sd_cat.png` - photorealistic cat
- `cpp23_sd_astronaut.png` - astronaut on Mars
- `cpp23_sd_dalek.png` - vintage Dalek photo

## Why stable-diffusion.cpp?

Like `llama.cpp` for LLMs, `stable-diffusion.cpp` provides:
- **GGUF format support** - Quantized models for faster inference
- **CPU/GPU acceleration** - MPS support on Apple Silicon
- **Pure C++ implementation** - No Python dependencies at runtime
- **Low memory footprint** - Q8 model uses ~2GB RAM vs 4GB+ for full precision

## NO DUMMY OUTPUT!

This generates **REAL** images using the actual Stable Diffusion 1.5 neural network, quantized to 8-bit precision for efficiency while maintaining high quality.

The model is the same SD 1.5 that powers many image generation tools, just in an optimized GGUF format that runs efficiently from C++!
