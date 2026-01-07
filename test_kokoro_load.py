import torch
from transformers import AutoModel
try:
    model = AutoModel.from_pretrained('hexgrad/Kokoro-82M', trust_remote_code=True)
    print('Successfully loaded Kokoro model')
except Exception as e:
    print(f'Failed to load: {e}')
