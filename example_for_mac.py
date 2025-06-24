import torch
import torchaudio as ta
import os
import random
import argparse
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

# Define emotional presets
EMOTION_PRESETS = {
    "normal": {"exaggeration": 1.0, "cfg_weight": 0.7, "temperature": 0.7},
    "excited": {"exaggeration": 2.5, "cfg_weight": 0.3, "temperature": 0.9},
    "sad": {"exaggeration": 0.5, "cfg_weight": 0.8, "temperature": 0.5},
    "angry": {"exaggeration": 3.0, "cfg_weight": 0.2, "temperature": 1.0},
    "dramatic": {"exaggeration": 3.5, "cfg_weight": 0.1, "temperature": 1.1},
    "whisper": {"exaggeration": 0.1, "cfg_weight": 0.95, "temperature": 0.3},
    "mysterious": {"exaggeration": 1.5, "cfg_weight": 0.6, "temperature": 0.6},
    "dreamy": {"exaggeration": 0.8, "cfg_weight": 0.75, "temperature": 0.8},
    "villain": {"exaggeration": 2.8, "cfg_weight": 0.25, "temperature": 0.95},
    "childlike": {"exaggeration": 2.2, "cfg_weight": 0.4, "temperature": 1.0},
    "seductive": {"exaggeration": 1.8, "cfg_weight": 0.5, "temperature": 0.85},
    "authoritative": {"exaggeration": 1.2, "cfg_weight": 0.8, "temperature": 0.6},
    "maniacal": {"exaggeration": 4.0, "cfg_weight": 0.05, "temperature": 1.2}
}

# Get available voices
voices_dir = "./voices"
voice_files = [f for f in os.listdir(voices_dir) if f.endswith('.wav')]

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate speech with different emotional tones')
parser.add_argument('--emotion', '-e', choices=list(EMOTION_PRESETS.keys()), 
                    default='normal', help='Emotional tone for the speech')
parser.add_argument('--voice', '-v', choices=voice_files + ['random'], 
                    default='random', help='Voice to use for synthesis')
parser.add_argument('--text', '-t', type=str, 
                    default="Today is the day. I want to move like a titan.",
                    help='Text to synthesize')
parser.add_argument('--output', '-o', type=str, default='test-2.wav', 
                    help='Output filename')
parser.add_argument('--render-all', '-r', type=str, metavar='VOICE', 
                    choices=voice_files, help='Render all emotions using specified voice as voice_emotion.wav')
args = parser.parse_args()

model = ChatterboxTTS.from_pretrained(device=device)

# Check if render-all mode
if args.render_all:
    voice_name = args.render_all.replace('.wav', '')
    audio_prompt_path = os.path.join(voices_dir, args.render_all)
    
    # Create output directory for this voice
    output_dir = voice_name
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Rendering all emotions using voice: {args.render_all}")
    print(f"Text format: hello this is {voice_name} and I am reading in [emotion] style voice.")
    print(f"Output directory: {output_dir}/")
    print()
    
    for emotion, params in EMOTION_PRESETS.items():
        output_filename = os.path.join(output_dir, f"{emotion}.wav")
        print(f"Generating {emotion} emotion... ({output_filename})")
        
        # Use custom text format for render-all mode
        render_text = f"hello this is {voice_name} and I am reading in {emotion} style voice."
        
        wav = model.generate(
            render_text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=params["exaggeration"],
            cfg_weight=params["cfg_weight"],
            temperature=params["temperature"]
        )
        ta.save(output_filename, wav, model.sr)
    
    print(f"\nCompleted! Generated {len(EMOTION_PRESETS)} files.")
else:
    # Single emotion mode
    emotion_params = EMOTION_PRESETS[args.emotion]
    
    # Select voice (random or specified)
    if args.voice == 'random':
        selected_voice = random.choice(voice_files)
    else:
        selected_voice = args.voice
    
    audio_prompt_path = os.path.join(voices_dir, selected_voice)
    
    print(f"Using voice: {selected_voice}")
    print(f"Emotion: {args.emotion}")
    print(f"Parameters: {emotion_params}")
    
    wav = model.generate(
        args.text, 
        audio_prompt_path=audio_prompt_path,
        exaggeration=emotion_params["exaggeration"],
        cfg_weight=emotion_params["cfg_weight"],
        temperature=emotion_params["temperature"]
        )
    ta.save(args.output, wav, model.sr)
    print(f"Saved to: {args.output}")
