import torch
from diffusers import StableDiffusionPipeline
import os

def export():
    model_id = "segmind/tiny-sd"
    save_path = "tiny_sd_onnx"
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Loading {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    
    # 1. Export UNet
    print("Exporting UNet to ONNX...")
    unet = pipe.unet
    dummy_input = (torch.randn(1, 4, 64, 64), torch.tensor([1.0]), torch.randn(1, 77, 768))
    torch.onnx.export(
        unet, 
        dummy_input, 
        os.path.join(save_path, "unet.onnx"),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["out_sample"],
        dynamic_axes={"sample": {0: "batch"}, "encoder_hidden_states": {0: "batch"}, "out_sample": {0: "batch"}},
        opset_version=14
    )

    # 2. Export VAE Decoder
    print("Exporting VAE Decoder to ONNX...")
    vae = pipe.vae
    dummy_latent = torch.randn(1, 4, 64, 64)
    # We wrap the decode call
    class VaeDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
        def forward(self, x):
            return self.vae.decode(x).sample

    vae_model = VaeDecoder(vae)
    torch.onnx.export(
        vae_model,
        dummy_latent,
        os.path.join(save_path, "vae_decoder.onnx"),
        input_names=["latent_sample"],
        output_names=["sample"],
        dynamic_axes={"latent_sample": {0: "batch"}, "sample": {0: "batch"}},
        opset_version=14
    )
    
    print(f"Export Complete! Files in {save_path}")

if __name__ == "__main__":
    export()
