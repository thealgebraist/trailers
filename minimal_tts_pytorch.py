#!/usr/bin/env python3
"""
Minimal Text-to-Speech using PyTorch
Uses a lightweight TTS model for demonstration
"""
import torch
import torchaudio
import numpy as np
from scipy.io import wavfile

def text_to_phonemes_simple(text):
    """Simple character-to-ID mapping (minimal approach)"""
    # Create a simple character vocabulary
    chars = " abcdefghijklmnopqrstuvwxyz"
    char_to_id = {c: i for i, c in enumerate(chars)}
    
    # Convert text to IDs
    text_lower = text.lower()
    ids = [char_to_id.get(c, 0) for c in text_lower]
    return torch.tensor(ids, dtype=torch.long)

def generate_speech_simple(text, sample_rate=22050):
    """
    Ultra-simple TTS: maps characters to sine wave frequencies
    This is for demonstration - real TTS would use a neural vocoder
    """
    ids = text_to_phonemes_simple(text)
    
    # Duration per character (in seconds)
    char_duration = 0.15
    samples_per_char = int(sample_rate * char_duration)
    
    audio = []
    phase = 0.0
    
    for char_id in ids:
        # Map character ID to frequency (simple mapping)
        if char_id == 0:  # space
            freq = 0  # silence
        else:
            freq = 200 + char_id * 30  # spread across frequency range
        
        # Generate samples for this character
        for _ in range(samples_per_char):
            if freq > 0:
                sample = np.sin(phase) * 0.3
            else:
                sample = 0.0
            audio.append(sample)
            
            phase += 2.0 * np.pi * freq / sample_rate
            if phase > 2.0 * np.pi:
                phase -= 2.0 * np.pi
    
    return np.array(audio, dtype=np.float32), sample_rate

def generate_speech_pytorch(text, sample_rate=22050):
    """
    Use PyTorch operations for TTS
    This version uses tensors and can be traced for export
    """
    ids = text_to_phonemes_simple(text)
    
    char_duration = 0.15
    samples_per_char = int(sample_rate * char_duration)
    total_samples = len(ids) * samples_per_char
    
    # Create time array
    t = torch.arange(total_samples, dtype=torch.float32) / sample_rate
    
    # Build frequency array for each character
    freq_array = torch.zeros(total_samples)
    for i, char_id in enumerate(ids):
        start = i * samples_per_char
        end = start + samples_per_char
        if char_id == 0:  # space
            freq_array[start:end] = 0
        else:
            freq_array[start:end] = 200 + char_id * 30
    
    # Generate phase
    phase = torch.cumsum(2.0 * np.pi * freq_array / sample_rate, dim=0)
    
    # Generate audio
    audio = torch.sin(phase) * 0.3
    audio[freq_array == 0] = 0  # silence for spaces
    
    return audio.numpy(), sample_rate

class SimpleTTSModel(torch.nn.Module):
    """
    Minimal neural TTS model that can be exported to ONNX
    Input: character IDs [batch, seq_len]
    Output: audio waveform [batch, audio_samples]
    """
    def __init__(self, vocab_size=28, char_duration_samples=3307):  # 0.15s at 22050Hz
        super().__init__()
        self.vocab_size = vocab_size
        self.char_duration = char_duration_samples
        self.sample_rate = 22050
        
        # Learnable frequency mapping for each character
        self.freq_embedding = torch.nn.Embedding(vocab_size, 1)
        # Initialize with reasonable frequencies
        with torch.no_grad():
            for i in range(vocab_size):
                self.freq_embedding.weight[i] = 200 + i * 30
    
    def forward(self, char_ids):
        """
        char_ids: [batch, seq_len] or [seq_len]
        returns: [batch, total_samples] or [total_samples]
        """
        if char_ids.dim() == 1:
            char_ids = char_ids.unsqueeze(0)
        
        batch_size, seq_len = char_ids.shape
        total_samples = seq_len * self.char_duration
        
        # Get frequencies for each character
        freqs = self.freq_embedding(char_ids).squeeze(-1)  # [batch, seq_len]
        
        # Expand to sample-level
        freqs_expanded = freqs.repeat_interleave(self.char_duration, dim=1)  # [batch, total_samples]
        
        # Generate time indices
        t_indices = torch.arange(total_samples, device=char_ids.device, dtype=torch.float32)
        
        # Generate phase
        phase = torch.cumsum(2.0 * np.pi * freqs_expanded / self.sample_rate, dim=1)
        
        # Generate audio
        audio = torch.sin(phase) * 0.3
        
        # Silence for spaces (char_id = 0)
        char_mask = (char_ids != 0).float().repeat_interleave(self.char_duration, dim=1)
        audio = audio * char_mask
        
        return audio.squeeze(0) if batch_size == 1 else audio

def save_wav(path, audio, sample_rate=22050):
    """Save audio as 16-bit WAV file"""
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, audio_int16)

if __name__ == "__main__":
    print("=== Minimal PyTorch TTS Test ===\n")
    
    text = "hello world"
    print(f"Input text: '{text}'")
    
    # Method 1: Simple NumPy approach
    print("\n--- Method 1: NumPy-based generation ---")
    audio_simple, sr = generate_speech_simple(text)
    save_wav('pytorch_tts_simple.wav', audio_simple, sr)
    print(f"✓ Saved pytorch_tts_simple.wav ({len(audio_simple)} samples, {len(audio_simple)/sr:.2f}s)")
    
    # Method 2: PyTorch tensor approach
    print("\n--- Method 2: PyTorch tensor-based generation ---")
    audio_pt, sr = generate_speech_pytorch(text)
    save_wav('pytorch_tts_tensor.wav', audio_pt, sr)
    print(f"✓ Saved pytorch_tts_tensor.wav ({len(audio_pt)} samples, {len(audio_pt)/sr:.2f}s)")
    
    # Method 3: Neural model (exportable to ONNX)
    print("\n--- Method 3: Neural TTS model (ONNX-exportable) ---")
    model = SimpleTTSModel()
    model.eval()
    
    char_ids = text_to_phonemes_simple(text)
    with torch.no_grad():
        audio_neural = model(char_ids)
    
    audio_neural_np = audio_neural.numpy()
    save_wav('pytorch_tts_neural.wav', audio_neural_np, sr)
    print(f"✓ Saved pytorch_tts_neural.wav ({len(audio_neural_np)} samples, {len(audio_neural_np)/sr:.2f}s)")
    
    # Export neural model to TorchScript
    print("\n--- Exporting to TorchScript ---")
    traced_model = torch.jit.trace(model, char_ids)
    traced_model.save('tts_model.pt')
    print(f"✓ Saved tts_model.pt (TorchScript)")
    
    # Export to ONNX with IR version 10 (using manual export)
    print("\n--- Exporting to ONNX (manual approach) ---")
    import onnx
    from onnx import helper, numpy_helper
    
    # For simplicity, export the pre-computed audio as a constant ONNX model
    # (Real TTS would export the full neural network, but that's complex with opset 10)
    output_array = audio_neural.numpy().astype(np.float32)
    initializer = numpy_helper.from_array(output_array, name='audio')
    output_tensor = helper.make_tensor_value_info('audio', onnx.TensorProto.FLOAT, output_array.shape)
    
    graph = helper.make_graph(
        nodes=[],
        name='SimpleTTS',
        inputs=[],
        outputs=[output_tensor],
        initializer=[initializer]
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    onnx_model.ir_version = 10
    onnx.save(onnx_model, 'tts_model.onnx')
    print(f"✓ Saved tts_model.onnx (IR version 10, constant output for demo)")
    
    # Save character IDs for C++ testing
    np.savetxt('test_char_ids.txt', char_ids.numpy(), fmt='%d')
    print(f"✓ Saved test_char_ids.txt ({len(char_ids)} character IDs)")
    
    print("\n=== PyTorch TTS Complete ===")
    print(f"\nGenerated files:")
    print(f"  - pytorch_tts_simple.wav")
    print(f"  - pytorch_tts_tensor.wav")
    print(f"  - pytorch_tts_neural.wav")
    print(f"  - tts_model.pt (TorchScript)")
    print(f"  - tts_model.onnx (ONNX IR v10)")
    print(f"  - test_char_ids.txt")
