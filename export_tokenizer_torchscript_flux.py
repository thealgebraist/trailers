from transformers import AutoTokenizer
import torch
import numpy as np

model_id = "openai/clip-vit-large-patch14"
text = "A photorealistic cinematic shot of a living room, 8k."

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
except Exception as e:
    print(f"Failed to load tokenizer {model_id}, falling back to gpt2: {e}
")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Tokenize
tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

class TokenizerModule(torch.nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.register_buffer('tokens', tokens)

    def forward(self, dummy):
        return self.tokens.unsqueeze(0)

module = TokenizerModule(tokens)
scripted = torch.jit.trace(module, torch.zeros(1))
scripted.save('flux_tokenizer.pt')

# save tokens as numpy and text
np.save('flux_token_ids.npy', tokens.numpy())
with open('flux_token_ids.txt','w') as f:
    f.write(' '.join(map(str, tokens.tolist())))
print('Saved flux_tokenizer.pt, flux_token_ids.npy, flux_token_ids.txt')
