import torch
from diffusers import StableDiffusionPipeline
import os

def export():
    model_id = "segmind/tiny-sd"
    save_path = "tiny_sd_pt"
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Loading {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    
    # 1. Export UNet
    print("Tracing UNet...")
    class UnetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet
        def forward(self, sample, timestep, encoder_hidden_states):
            # Return only the first element of the UNet output (the sample)
            return self.unet(sample, timestep, encoder_hidden_states, return_dict=False)[0]

    unet = UnetWrapper(pipe.unet)
    sample = torch.randn(1, 4, 64, 64)
    timestep = torch.tensor([1.0])
    encoder_hidden_states = torch.randn(1, 77, 768)
    
    traced_unet = torch.jit.trace(unet, (sample, timestep, encoder_hidden_states), check_trace=False)
    traced_unet.save(os.path.join(save_path, "unet.pt"))

    # 2. Export VAE Decoder
    print("Tracing VAE Decoder...")
    class VaeDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
        def forward(self, x):
            return self.vae.decode(x, return_dict=False)[0]
    
    vae_dec = VaeDecoder(pipe.vae)
    latent = torch.randn(1, 4, 64, 64)
    traced_vae = torch.jit.trace(vae_dec, (latent,), check_trace=False)
    traced_vae.save(os.path.join(save_path, "vae.pt"))

    # 3. Export Text Encoder (CLIP)
    print("Tracing Text Encoder...")
    class ClipWrapper(torch.nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.clip = clip
        def forward(self, tokens):
            return self.clip(tokens, return_dict=False)[0]

    text_enc = ClipWrapper(pipe.text_encoder)
    tokens = torch.zeros((1, 77), dtype=torch.long)
    traced_clip = torch.jit.trace(text_enc, (tokens,), check_trace=False)
    traced_clip.save(os.path.join(save_path, "clip.pt"))
    
    print(f"Success! TorchScript models saved to {save_path}")

if __name__ == "__main__":
    export()