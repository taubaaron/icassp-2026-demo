import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from encodec import EncodecModel

# --- Configuration ---
AUDIO_DIR = "aaron_xai4ae/approach_3/results-attempt_2/vctk_sub_dataset-attempt_2/p236/"
OUTPUT_DIR = "aaron_xai4ae/approach_3/results-attempt_3/manipulation/pitch_shift_manipulation-results/pca"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_COMPONENTS = 20  # Changeable number of PCA components
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---
def resample_audio(file_path, target_sample_rate=24000):
    """Loads and resamples an audio file to the target sample rate."""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
    return waveform

def shift_pitch(waveform, sample_rate, semitones):
    """Shifts the pitch of the waveform by a given number of semitones."""
    return torchaudio.transforms.PitchShift(sample_rate, semitones)(waveform)

def extract_latent_representation(model, waveform, device):
    """Extracts the latent representation from the EnCodec model."""
    waveform = waveform.to(device)
    with torch.no_grad():
        latent_rep = model.encoder(waveform.unsqueeze(0)).cpu().numpy()
    return latent_rep.squeeze(0).T  # Shape: (frames, 128)

def normalize_latents(latents):
    """Normalizes each frame's latent representation independently."""
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)
    return latents_scaled, scaler

def apply_pca(latents, n_components=NUM_COMPONENTS):
    """Applies PCA to each frame independently."""
    pca_model = PCA(n_components=n_components)
    transformed = pca_model.fit_transform(latents)  # Shape: (frames, n_components)
    return transformed, pca_model

def plot_pca(latents, pitch_shifts, output_dir, file_name):
    """Plots the PCA-reduced latent representations."""
    plt.figure(figsize=(8, 6))
    plt.scatter(latents[:, 0], latents[:, 1], c=pitch_shifts, cmap='coolwarm', edgecolors='k')
    plt.colorbar(label="Pitch Shift (Semitones)")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title(f"2D PCA - Pitch Shift Trajectory - {file_name}")
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"{file_name}_pca_2d.png"))
    plt.show()

def analyze_pitch_variation(audio_path, output_dir, device):
    """Runs pitch shifting and extracts latent representations."""
    model = EncodecModel.encodec_model_24khz().to(device).eval()
    waveform = resample_audio(audio_path)

    latents = []
    pitch_shifts = []

    for tone_shift in range(-20, 21):  
        print(f"Processing pitch shift: {tone_shift}")
        shifted_audio = shift_pitch(waveform, sample_rate=24000, semitones=tone_shift)
        latent = extract_latent_representation(model, shifted_audio, device)
        latents.append(latent)
        pitch_shifts.append(tone_shift)

    latents = np.array(latents)  # Shape: (41, frames, 128)

    print("Normalizing latents...")
    latents = latents.reshape(-1, 128)  # Merge all pitch shifts to apply PCA across all frames
    latents_scaled, scaler = normalize_latents(latents)

    print("Applying PCA per frame...")
    transformed, pca_model = apply_pca(latents_scaled)

    file_name = os.path.basename(audio_path)[:-4]
    repetition = transformed.shape[0] // len(pitch_shifts)
    expanded_pitch_shifts = pitch_shifts * repetition
    plot_pca(transformed, expanded_pitch_shifts, output_dir, file_name)

    num_shifts = len(pitch_shifts)
    num_frames = transformed.shape[0] // num_shifts
    transformed_reshaped = transformed.reshape(num_shifts, num_frames, -1)  # (41, frames, components)

    return transformed_reshaped, pca_model, scaler, pitch_shifts, model, latents


# --- Step 2: Move in Latent Space ---
def fit_pitch_shift_function(pitch_shifts, transformed_reshaped, degree=3, frame_idx=None):
    """Fits polynomial regression for each PCA component at a specific frame index."""
    num_shifts, num_frames, num_components = transformed_reshaped.shape
    if frame_idx is None:
        frame_idx = num_frames // 2  # Middle frame

    # Take the values at that frame across all shifts
    values_across_shifts = transformed_reshaped[:, frame_idx, :]  # Shape: (41, components)
    fitted_functions = [np.poly1d(np.polyfit(pitch_shifts, values_across_shifts[:, i], degree)) for i in range(num_components)]
    return fitted_functions

def move_in_latent_space(target_shift, pitch_shifts, transformed, fitted_functions):
    """Predicts new PCA coordinates using polynomial regression."""
    num_components = len(fitted_functions)  # Instead of relying on shape
    new_pca_coords = np.array([f(target_shift) for f in fitted_functions])
    return new_pca_coords


# --- Step 3: Inverse PCA and Decode ---
def inverse_pca_and_decode(model, pca_model, scaler, transformed_point, device):
    """Reconstructs full latent space using PCA inverse transformation and decodes it."""
    transformed_point = np.array(transformed_point).reshape(1, -1)

    # Use PCA inverse transform
    full_latent = pca_model.inverse_transform(transformed_point)

    # Reverse normalization
    full_latent = scaler.inverse_transform(full_latent)

    # Convert to PyTorch tensor
    full_latent = torch.tensor(full_latent, dtype=torch.float32).to(device)
    full_latent = full_latent.view(1, full_latent.shape[0], 128).permute(0, 2, 1)  # Reshape to (1, 128, frames)

    # Decode back to waveform
    with torch.no_grad():
        decoded_audio = model.decoder(full_latent).cpu()

    return decoded_audio

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting")

    # Step 1: Extract Latents and Perform PCA Per Frame
    AUDIO_FILE = "p236_004_mic2.flac"
    print(f"Processing {AUDIO_FILE}")
    AUDIO_PATH = os.path.join(AUDIO_DIR, AUDIO_FILE)

    transformed, pca_model, scaler, pitch_shifts, model, original_latents = analyze_pitch_variation(AUDIO_PATH, OUTPUT_DIR, DEVICE)

    # Step 2: Move in Latent Space
    target_pitch_shift = 0  # Change to desired shift
    fitted_functions = fit_pitch_shift_function(pitch_shifts, transformed)
    new_pca_coords = move_in_latent_space(target_pitch_shift, pitch_shifts, transformed, fitted_functions)

    # Step 3: Inverse PCA and Decode
    print("Reconstructing audio from latent space...")
    reconstructed_audio = inverse_pca_and_decode(model, pca_model, scaler, new_pca_coords, DEVICE)

    # Save reconstructed audio
    output_audio_path = os.path.join(OUTPUT_DIR, f"reconstructed_{target_pitch_shift}.wav")
    torchaudio.save(output_audio_path, reconstructed_audio.squeeze(0), 24000)
    print(f"Saved reconstructed audio: {output_audio_path}")

    print("Finished")
