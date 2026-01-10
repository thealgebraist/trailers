import torch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from .core import get_device, flush

def load_parler(repo_id="parler-tts/parler-tts-mini-v1"):
    device = get_device()
    print(f"Loading Parler-TTS model: {repo_id}...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return model, tokenizer

def generate_parler_audio(model, tokenizer, text, description, device=None):
    if device is None:
        device = get_device()
    
    desc = tokenizer(description, return_tensors="pt", return_attention_mask=True)
    prompt = tokenizer(text, return_tensors="pt", return_attention_mask=True)

    input_ids = desc.input_ids.to(device)
    attention_mask = desc.attention_mask.to(device)
    prompt_input_ids = prompt.input_ids.to(device)
    prompt_attention_mask = prompt.attention_mask.to(device)

    generation = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
    )
    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr

def save_audio(path, audio_arr, sample_rate):
    sf.write(path, audio_arr, sample_rate)
