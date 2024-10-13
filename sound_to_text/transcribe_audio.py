import torch
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np

# Set parameters
samplerate = 16000  # Whisper expects 16kHz input
duration = 10       # Duration to record in seconds
output_file = "input_audio.wav"  # Temp file for the audio recording

# Record audio on the spot
print(f"Recording for {duration} seconds...")
audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
sd.wait()  # Wait for the recording to finish

# Save the recorded audio to a WAV file
write(output_file, samplerate, audio_data)

# See https://huggingface.co/openai/whisper-large-v3
# Load the Whisper model and processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")


# Load the audio into a tensor (Whisper requires float32)
audio_input = processor(audio_data.flatten(), sampling_rate=samplerate, return_tensors="pt").input_features

# Transcribe the audio
with torch.no_grad():
    predicted_ids = model.generate(audio_input, language='el')
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# Output the transcription
print("-"*90)
print("Transcription:", transcription)
