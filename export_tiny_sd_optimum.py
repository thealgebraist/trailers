from optimum.onnxruntime import ORTModelForStableDiffusion
import os

def export():
    model_id = "segmind/tiny-sd"
    save_path = "tiny_sd_onnx"
    
    print(f"Loading and exporting {model_id} via Optimum...")
    # This command downloads and converts UNet, VAE, CLIP, and Tokenizer automatically
    model = ORTModelForStableDiffusion.from_pretrained(model_id, export=True)
    
    print(f"Saving to {save_path}...")
    model.save_pretrained(save_path)
    print("Export successful!")

if __name__ == "__main__":
    export()
