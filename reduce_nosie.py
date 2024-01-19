import librosa
from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.utils.reduce_noise import NoiseReducer
y, sr = librosa.load('aaa.wav', sr=16000)
reducer = NoiseReducer(sr=sr)
output_path = reducer.reduce_audio_amplify('aaa.wav',y)
print(output_path)