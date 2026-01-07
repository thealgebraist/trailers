from transformers import AutoTokenizer
import numpy as np
import onnx
from onnx import helper, numpy_helper

# Load tokenizer and produce token ids for a sample text
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
text = "In a world that was once crisp and dry, a strange phenomenon has begun."
outputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, return_attention_mask=True)
tokens = outputs["input_ids"]  # shape (1, seq_len)
attention_mask = outputs["attention_mask"]
# Save attention mask
np.save('bark_token_attention_mask.npy', attention_mask)
with open('bark_token_attention_mask.txt','w') as f: f.write(' '.join(map(str, attention_mask.flatten().tolist())))

# Create an ONNX model with no inputs that outputs the constant token ids
output_name = 'output'
output_array = np.asarray(tokens, dtype=np.int64)
initializer = numpy_helper.from_array(output_array, name=output_name)

output_tensor = helper.make_tensor_value_info(output_name, onnx.TensorProto.INT64, output_array.shape)

graph = helper.make_graph(
    nodes=[],
    name='TokenizerGraph',
    inputs=[],
    outputs=[output_tensor],
    initializer=[initializer]
)

model = helper.make_model(graph)
onnx.save(model, 'bark_tokenizer.onnx')
import numpy as np
np.save('bark_token_ids.npy', tokens)
with open('bark_token_ids.txt','w') as f:
    f.write(' '.join(map(str, tokens.flatten().tolist())))
print('Saved bark_tokenizer.onnx, bark_token_ids.npy, bark_token_ids.txt')
