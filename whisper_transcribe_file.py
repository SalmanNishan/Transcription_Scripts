import whisper
import os 

dir_path = r"/home/cmdadmin/KL Audios/"
filename = "growing landscape o -ai for code.mp3"
file_path = os.path.join(dir_path, filename)

model = whisper.load_model("base")
result = model.transcribe(file_path)

transcript = filename.split(".")[0] + ".txt"
with open(transcript, 'w') as f:
	f.write(result["text"])

print(result["text"])

# # load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio(r"1628619083.1166868.wav")
# audio = whisper.pad_or_trim(audio)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)

# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# # decode the audio
# options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print(result.text)
