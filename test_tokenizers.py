import torch

print('Loading TorchScript tokenizer (bark_tokenizer.pt)')
try:
    mod = torch.jit.load('bark_tokenizer.pt')
    # module expects a dummy input due to tracing
    out = mod(torch.zeros(1))
    print('TorchScript tokenizer output shape:', out.shape)
    print('TorchScript tokens:', out.tolist())
except Exception as e:
    print('Failed to load/run TorchScript tokenizer:', e)

print('\nAttempting ONNX tokenizer (bark_tokenizer.onnx) via onnxruntime if available')
try:
    import onnxruntime as ort
    sess = ort.InferenceSession('bark_tokenizer.onnx')
    outputs = sess.run(None, {})
    print('ONNX tokenizer output shape:', outputs[0].shape)
    print('ONNX tokens:', outputs[0].tolist())
except Exception as e:
    print('onnxruntime not available or ONNX test failed:', e)
    # Fallback: print saved token ids if present
    try:
        tokens = torch.load('bark_token_ids.pt')
        print('Fallback: loaded bark_token_ids.pt, shape:', tokens.shape)
        print(tokens.tolist())
    except Exception as e2:
        print('Fallback failed to load bark_token_ids.pt:', e2)
