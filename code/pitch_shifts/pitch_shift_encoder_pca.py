import os
import json
import torch
import torchaudio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from encodec import EncodecModel

# --- Configuration ---
# AUDIO_DIR = "aaron_xai4ae/approach_3/attempt_3/pitch_shifts/sine_wave/"
# AUDIO_DIR = "aaron_xai4ae/common/dataset/vctk_sub_dataset/"
AUDIO_DIR = "aaron_xai4ae/approach_3/attempt_2/vctk_sub_dataset-attempt_2/"
AUDIO_FILE = "440_sine_wave.wav"
AUDIO_PATH = AUDIO_DIR+AUDIO_FILE
OUTPUT_DIR = "aaron_xai4ae/approach_3/attempt_3/pitch_latent_space_geometry/speaker_audio-quarter_tones"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_AUDIOS_PER_SPEAKER = 10
NUM_PCA_ITERATIONS = 10  # Outer loop for PCA calculations
os.makedirs(OUTPUT_DIR, exist_ok=True)

def resample_audio(file_path, target_sample_rate=24000):
    """Loads and resamples an audio file to the target sample rate."""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
    return waveform

import torchaudio.functional as F
def shift_pitch_quarter_tone(waveform, sample_rate, quarter_tones):
    """Shifts pitch by a fractional number of semitones (quarter tones)."""
    semitones = quarter_tones * 0.25  # Convert quarter tones to semitones
    return F.pitch_shift(waveform, sample_rate, semitones)

def shift_pitch_half_tone(waveform, sample_rate, quarter_tones):
    """Shifts pitch by a fractional number of semitones (quarter tones)."""
    semitones = quarter_tones * 0.5  # Convert quarter tones to semitones
    return F.pitch_shift(waveform, sample_rate, semitones)

def shift_pitch(waveform, sample_rate, semitones):
    """Shifts the pitch of the waveform by a given number of semitones."""
    return torchaudio.transforms.PitchShift(sample_rate, semitones)(waveform)

def extract_latent_representation(model, waveform, device):
    """Extracts the latent representation from the EnCodec model."""
    waveform = waveform.to(device)
    with torch.no_grad():
        latent_rep = model.encoder(waveform.unsqueeze(0)).cpu().numpy()
    return latent_rep.squeeze(0)  # Remove batch dim -> (128, frames)

def normalize_latents(latents):
    """Flattens and normalizes latent representations for PCA."""
    latents = np.array([latent.flatten() for latent in latents])
    scaler = StandardScaler()
    return scaler.fit_transform(latents)

def plot_pca(latents, pitch_shifts, output_dir):
    """Performs PCA and plots the 2D and 3D visualizations."""
    pca = PCA(n_components=3)
    transformed = pca.fit_transform(latents)
    file_name = AUDIO_FILE[5:-4]
    
    # 2D Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed[:, 0], transformed[:, 1], c=pitch_shifts, cmap='coolwarm', edgecolors='k')
    plt.colorbar(label="Pitch Shift (Semitones)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"2D PCA - Pitch Shift Trajectory - {file_name}")
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"{file_name}_pca_2d.png"))
    plt.show()
    
    # 3D Plot
    df = pd.DataFrame({
        'PC1': transformed[:, 0],
        'PC2': transformed[:, 1],
        'PC3': transformed[:, 2],
        'Pitch Shift': pitch_shifts
    })
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Pitch Shift', title=f"3D PCA - Pitch Shift Trajectory - {AUDIO_FILE}")
    fig.write_html(os.path.join(output_dir, f"{file_name}_pca_3d.html"))

    return transformed

def analyze_pitch_variation(audio_path, output_dir, device):
    """Runs the analysis by shifting pitch and extracting latent representations."""
    model = EncodecModel.encodec_model_24khz().to(device).eval()
    waveform = resample_audio(audio_path)
    
    latents = []
    pitch_shifts = []
    
    for tone_shift in range(-20, 20):  # Does each range plus and minus
        print(f"running for pitch shifted {tone_shift}")
        # shifted_audio = shift_pitch(waveform, sample_rate=24000, semitones=semitone) # semitone
        shifted_audio = shift_pitch_quarter_tone(waveform, sample_rate=24000, quarter_tones=tone_shift) # quartertone
        latent = extract_latent_representation(model, shifted_audio, device)
        latents.append(latent)
        pitch_shifts.append(tone_shift)
    
    print("Normalizing latents")
    latents = normalize_latents(latents)
    print("Plotting PCA")
    transformed = plot_pca(latents, pitch_shifts, output_dir)
    

if __name__ == "__main__":
    for i in range(3,4):
        AUDIO_FILE = f"p236/p236_00{i}_mic2.flac"
        print(f"Starting {AUDIO_FILE}")
        AUDIO_PATH = AUDIO_DIR+AUDIO_FILE
        analyze_pitch_variation(AUDIO_PATH, OUTPUT_DIR, DEVICE)
    # for i in range(2,10):
    #     AUDIO_FILE = f"p374/p374_00{i}_mic2.flac"
    #     print(f"Starting {AUDIO_FILE}")
    #     AUDIO_PATH = AUDIO_DIR+AUDIO_FILE
    #     analyze_pitch_variation(AUDIO_PATH, OUTPUT_DIR, DEVICE)

    # Sine Wave
    # print(f"Starting {AUDIO_FILE}")
    # AUDIO_PATH = AUDIO_DIR+AUDIO_FILE
    # analyze_pitch_variation(AUDIO_PATH, OUTPUT_DIR, DEVICE)

    print("Finished")