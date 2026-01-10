#!/usr/bin/env python3
"""
Export segmind/tiny-sd to ONNX for C++ inference
This is a tiny Stable Diffusion model (~600MB) perfect for testing
"""
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download
import os

print("=== Exporting segmind/tiny-sd for C++ ===\n")

# Download and load the model
print("Loading segmind/tiny-sd (tiny Stable Diffusion model)...")
model_id = "segmind/tiny-sd"

# Load pipeline on CPU first
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    safety_checker=None
)
pipe = pipe.to("mps")  # Use Metal on macOS

print("✓ Model loaded on MPS device")

# Generate a test image to verify it works
print("\n--- Generating test image ---")
prompt = "a photo of an astronaut riding a horse on mars"
print(f"Prompt: '{prompt}'")

with torch.no_grad():
    image = pipe(
        prompt,
        num_inference_steps=20,  # Fast generation
        guidance_scale=7.5,
        height=512,
        width=512
    ).images[0]

image.save("tinysd_python_reference.png")
print(f"✓ Saved: tinysd_python_reference.png")

# Try to export UNET to ONNX
print("\n--- Exporting UNET to ONNX ---")
try:
    unet = pipe.unet
    unet.eval()
    unet = unet.to("cpu")
    
    # Dummy inputs for tracing
    latent_model_input = torch.randn(2, 4, 64, 64)
    t = torch.tensor([999])
    encoder_hidden_states = torch.randn(2, 77, 512)
    
    print("  Tracing UNET...")
    torch.onnx.export(
        unet,
        (latent_model_input, t, encoder_hidden_states),
        "tinysd_unet.onnx",
        input_names=["latent", "timestep", "encoder_hidden_states"],
        output_names=["noise_pred"],
        dynamic_axes={
            "latent": {0: "batch"},
            "encoder_hidden_states": {0: "batch"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    print("✓ Exported UNET to: tinysd_unet.onnx")
except Exception as e:
    print(f"✗ UNET export failed: {e}")

# Export text encoder to ONNX
print("\n--- Exporting Text Encoder to ONNX ---")
try:
    text_encoder = pipe.text_encoder
    text_encoder.eval()
    text_encoder = text_encoder.to("cpu")
    
    # Dummy input
    input_ids = torch.randint(0, 1000, (1, 77))
    
    print("  Tracing Text Encoder...")
    torch.onnx.export(
        text_encoder,
        input_ids,
        "tinysd_text_encoder.onnx",
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "batch"}},
        opset_version=14,
        do_constant_folding=True
    )
    print("✓ Exported Text Encoder to: tinysd_text_encoder.onnx")
except Exception as e:
    print(f"✗ Text Encoder export failed: {e}")

# Export VAE decoder to ONNX
print("\n--- Exporting VAE Decoder to ONNX ---")
try:
    vae = pipe.vae
    vae.eval()
    vae = vae.to("cpu")
    
    # Dummy latent
    latents = torch.randn(1, 4, 64, 64)
    
    print("  Tracing VAE Decoder...")
    torch.onnx.export(
        vae.decode,
        latents,
        "tinysd_vae_decoder.onnx",
        input_names=["latents"],
        output_names=["sample"],
        dynamic_axes={"latents": {0: "batch"}},
        opset_version=14,
        do_constant_folding=True
    )
    print("✓ Exported VAE Decoder to: tinysd_vae_decoder.onnx")
except Exception as e:
    print(f"✗ VAE Decoder export failed: {e}")

# Save tokenizer vocab for C++
print("\n--- Saving tokenizer data ---")
tokenizer = pipe.tokenizer
test_prompt = "a photo of an astronaut riding a horse on mars"
tokens = tokenizer(
    test_prompt,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="pt"
)

np.savetxt("tinysd_token_ids.txt", tokens.input_ids.numpy().flatten(), fmt='%d')
print(f"✓ Saved token IDs: tinysd_token_ids.txt")

# Save model config
with open("tinysd_config.txt", "w") as f:
    f.write(f"model_id: {model_id}\n")
    f.write(f"prompt: {prompt}\n")
    f.write(f"height: 512\n")
    f.write(f"width: 512\n")
    f.write(f"num_inference_steps: 20\n")
    f.write(f"guidance_scale: 7.5\n")
    f.write(f"vocab_size: {tokenizer.vocab_size}\n")

print("✓ Saved config: tinysd_config.txt")

print("\n=== Export Complete ===")
print("\nNote: Full SD pipeline requires orchestrating:")
print("  1. Text Encoder (prompt -> embeddings)")
print("  2. UNET (denoising)")
print("  3. VAE Decoder (latents -> image)")
print("\nFor C++, we'll demonstrate loading these ONNX models")
