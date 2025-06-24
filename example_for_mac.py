import torch
import torchaudio as ta
import os
import random
from chatterbox.tts import ChatterboxTTS

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)
text = "Today is the day. I want to move like a titan at dawn, sweat like a god forging lightning. No more excuses. From now on, my mornings will be temples of discipline. I am going to work out like the gods… every damn day."

# Randomly select a voice from the voices directory
voices_dir = "./voices"
voice_files = [f for f in os.listdir(voices_dir) if f.endswith('.wav')]
random_voice = random.choice(voice_files)
audio_prompt_path = os.path.join(voices_dir, random_voice)

print(f"Using voice: {random_voice}")

wav = model.generate(
    text, 
    audio_prompt_path=audio_prompt_path,
    exaggeration=2.0,
    cfg_weight=0.5
    )
ta.save("test-2.wav", wav, model.sr)
