from transformers import AutoTokenizer
import torch

# Load tokenizer from cache
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
text = "In a world that was once crisp and dry, a strange phenomenon has begun."

# Tokenize using HuggingFace tokenizer
outputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
tokens = outputs["input_ids"].squeeze(0)  # shape (seq_len,)
attention_mask = outputs["attention_mask"].squeeze(0)
# Save attention mask alongside tokens
torch.save(attention_mask, 'bark_token_attention_mask.pt')
import numpy as np
np.save('bark_token_attention_mask.npy', attention_mask.numpy())
with open('bark_token_attention_mask.txt','w') as f: f.write(' '.join(map(str, attention_mask.tolist())))

class TokenizerModule(torch.nn.Module):
    def __init__(self, tokens):
        super().__init__()
        # register token ids as a buffer so they are part of the module
        self.register_buffer('tokens', tokens)

    def forward(self, dummy):
        # ignore dummy input; return token ids as batch of 1
        return self.tokens.unsqueeze(0)

# Create a TorchScript module that returns the token ids for this example text
module = TokenizerModule(tokens)
# Use tracing with a dummy input since the module has no string inputs
scripted = torch.jit.trace(module, torch.zeros(1))
scripted.save('bark_tokenizer.pt')

# Also save the raw token ids for reference
torch.save(tokens, 'bark_token_ids.pt')
import numpy as np
np.save('bark_token_ids.npy', tokens.numpy())
with open('bark_token_ids.txt','w') as f:
    f.write(' '.join(map(str, tokens.tolist())))
print('Saved bark_tokenizer.pt, bark_token_ids.pt, bark_token_ids.npy, bark_token_ids.txt')
