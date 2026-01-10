from transformers import AutoTokenizer
import numpy as np
import onnx
from onnx import helper, numpy_helper

model_id = "facebook/musicgen-small"
text = "Cinematic trailer music for a movie."

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
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
model.ir_version = 10
onnx.save(model, 'ldm2_tokenizer.onnx')
np.save('ldm2_token_ids.npy', output_array)
with open('ldm2_token_ids.txt','w') as f:
    f.write(' '.join(map(str, output_array.flatten().tolist())))
print('Saved ldm2_tokenizer.onnx, ldm2_token_ids.npy, ldm2_token_ids.txt')
