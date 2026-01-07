from transformers import AutoTokenizer
import torch
import numpy as np

model_id = "facebook/musicgen-small"
text = "Cinematic trailer music for a movie."

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
except Exception as e:
    print(f"Failed to load tokenizer {model_id}, falling back to gpt2: {e}")
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
scripted.save('ldm2_tokenizer.pt')

# save tokens as numpy and text
np.save('ldm2_token_ids.npy', tokens.numpy())
with open('ldm2_token_ids.txt','w') as f:
    f.write(' '.join(map(str, tokens.tolist())))
print('Saved ldm2_tokenizer.pt, ldm2_token_ids.npy, ldm2_token_ids.txt')
