import torch
from diffusers import (
    DiffusionPipeline, 
    DPMSolverMultistepScheduler, 
    AnimateDiffPipeline, 
    MotionAdapter,
    CogVideoXPipeline
)
from diffusers.utils import export_to_video
import os

def generate_video_reference():
    # Setup common parameters
    prompt = "A small orange cat playing with a ball of yarn, high quality, cinematic"
    negative_prompt = "distorted, blurry, low quality, static, noisy"
    num_frames = 16
    # Detect device: priority CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # 1. Configuration for 4 distinct tiny/efficient models
    model_configs = [
        {
            "name": "zeroscope_v2",
            "id": "cerspense/zeroscope_v2_576w",
            "type": "t2v"
        },
        {
            "name": "modelscope",
            "id": "damo-vilab/text-to-video-ms-1.7b",
            "type": "t2v"
        },
        {
            "name": "animatediff_lightning",
            "id": "ByteDance/AnimateDiff-Lightning",
            "type": "animatediff"
        },
        {
            "name": "cogvideox_2b",
            "id": "THUDM/CogVideoX-2b",
            "type": "cogvideo"
        }
    ]

    for config in model_configs:
        print(f"--- Processing Model: {config['name']} ---")
        try:
            dtype = torch.float16 if device != "cpu" else torch.float32
            
            if config["type"] == "t2v":
                # Standard Text-to-Video Pipeline
                pipe = DiffusionPipeline.from_pretrained(
                    config["id"], 
                    torch_dtype=dtype
                ).to(device)
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                
            elif config["type"] == "animatediff":
                # AnimateDiff requires a base SD model and a motion adapter
                adapter = MotionAdapter.from_pretrained(config["id"], torch_dtype=dtype)
                pipe = AnimateDiffPipeline.from_pretrained(
                    "emilianJR/epiCRealism", # Example base model
                    motion_adapter=adapter,
                    torch_dtype=dtype
                ).to(device)
                
            elif config["type"] == "cogvideo":
                # CogVideoX specific optimized pipeline
                pipe = CogVideoXPipeline.from_pretrained(
                    config["id"], 
                    torch_dtype=dtype
                ).to(device)

            # Memory Optimization: Offload to CPU if VRAM is low
            if device != "cpu":
                pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()

            # Inference
            print(f"Generating {num_frames} frames...")
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                num_inference_steps=25,
                guidance_scale=7.5
            )

            # Save result
            video_path = f"output_{config['name']}.mp4"
            export_to_video(output.frames[0], video_path, fps=8)
            print(f"Saved: {video_path}")

            # Cleanup for next model
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        except Exception as e:
            print(f"Failed to run {config['name']}: {e}")

if __name__ == "__main__":
    generate_video_reference()
