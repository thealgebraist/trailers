import ChatTTS
import torch
import scipy.io.wavfile
import numpy as np

def test_chattts():
    chat = ChatTTS.Chat()
    print("Loading ChatTTS models...")
    chat.load(compile=False) # compile=True might be slow or fail on some systems
    
    texts = ["This is a test of ChatTTS. We are checking if it works correctly on this system."]
    
    print("Inference...")
    wavs = chat.infer(texts)
    
    if wavs:
        audio = np.array(wavs[0]).flatten()
        scipy.io.wavfile.write("chattts_test.wav", 24000, audio)
        print("Success! Saved chattts_test.wav")
    else:
        print("Failed to generate audio.")

if __name__ == "__main__":
    test_chattts()
