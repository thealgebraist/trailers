from transformers import AutoTokenizer
import numpy as np
import onnx
from onnx import helper, numpy_helper

model_id = "openai/clip-vit-large-patch14"
text = "A photorealistic cinematic shot of a living room, 8k."

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
except Exception as e:
    print(f"Failed to load tokenizer {model_id}, falling back to gpt2: {e}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokens = tokenizer(text, return_tensors="np")["input_ids"]
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
onnx.save(model, 'flux_tokenizer.onnx')
np.save('flux_token_ids.npy', output_array)
with open('flux_token_ids.txt','w') as f:
    f.write(' '.join(map(str, output_array.flatten().tolist())))
print('Saved flux_tokenizer.onnx, flux_token_ids.npy, flux_token_ids.txt')
