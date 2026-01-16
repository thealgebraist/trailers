import torch
from diffusers import StableAudioPipeline

try:
    pipe = StableAudioPipeline.from_pretrained(
        "facebook/stable-audio-open-1.0", torch_dtype=torch.float16
    )
    print(f"Type: {type(pipe)}")
    print(f"Has components: {hasattr(pipe, 'components')}")
    if hasattr(pipe, 'components'):
        print(f"Components type: {type(pipe.components)}")
    
    print(f"Has named_modules: {hasattr(pipe, 'named_modules')}")
except Exception as e:
    print(f"Error: {e}")
