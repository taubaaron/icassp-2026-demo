import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
import random
import librosa
from scipy.signal import butter, filtfilt

class ReliableMorpher:
    def __init__(self):
        self.instrument_eqs = {}
        self.is_trained = False
        
    def preprocess_audio(self, waveform, sample_rate):
        if sample_rate != 24000:
            waveform = torchaudio.transforms.Resample(sample_rate, 24000)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
        
        target_samples = int(3.0 * 24000)
        if waveform.shape[1] != target_samples:
            if waveform.shape[1] < target_samples:
                waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[1]))
            else:
                waveform = waveform[:, :target_samples]
        
        return waveform.squeeze(0).numpy()
    
    def extract_eq_curve(self, audio):
        # Simple frequency analysis
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        frequencies = np.fft.rfftfreq(len(audio), 1/24000)
        
        # Define EQ bands
        bands = [
            (50, 150),    # Bass
            (150, 400),   # Low-mid
            (400, 1000),  # Mid
            (1000, 3000), # High-mid
            (3000, 8000), # Presence
            (8000, 12000) # Brilliance
        ]
        
        eq_curve = []
        for low_freq, high_freq in bands:
            band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            if np.any(band_mask):
                band_energy = np.mean(magnitude[band_mask])
                eq_curve.append(band_energy)
            else:
                eq_curve.append(0.0)
        
        # Normalize
        max_energy = max(eq_curve) + 1e-8
        eq_curve = [e / max_energy for e in eq_curve]
        
        return eq_curve, bands
    
    def train(self, csv_path, audio_dir, max_samples=5):
        print("Training reliable morpher...")
        
        df = pd.read_csv(csv_path)
        instruments = ['Sound_Guitar', 'Sound_Drum', 'Sound_Piano']
        
        for instrument_class in instruments:
            instrument_name = instrument_class.replace('Sound_', '')
            files = df[df['Class'] == instrument_class]['FileName'].tolist()
            existing_files = [f for f in files if (Path(audio_dir) / f).exists()]
            random.shuffle(existing_files)
            selected_files = existing_files[:max_samples]
            
            print(f"Processing {instrument_name}: {len(selected_files)} samples")
            
            eq_curves = []
            for audio_file in selected_files:
                try:
                    audio_path = Path(audio_dir) / audio_file
                    waveform, sr = torchaudio.load(audio_path)
                    audio = self.preprocess_audio(waveform, sr)
                    eq_curve, bands = self.extract_eq_curve(audio)
                    eq_curves.append(eq_curve)
                except Exception as e:
                    print(f"Error with {audio_file}: {e}")
                    continue
            
            if eq_curves:
                avg_eq = np.mean(eq_curves, axis=0)
                self.instrument_eqs[instrument_name] = {
                    'eq_curve': avg_eq,
                    'bands': bands
                }
                print(f"Created EQ for {instrument_name}: {[f'{x:.2f}' for x in avg_eq]}")
        
        self.is_trained = True
        return len(self.instrument_eqs)
    
    def morph_with_eq(self, source_audio_path, target_instrument, strength=0.5):
        if not self.is_trained or target_instrument not in self.instrument_eqs:
            raise ValueError(f"Model not trained or unknown instrument: {target_instrument}")
        
        # Load source
        waveform, sr = torchaudio.load(source_audio_path)
        audio = self.preprocess_audio(waveform, sr)
        
        # Get target EQ
        target_eq = self.instrument_eqs[target_instrument]['eq_curve']
        bands = self.instrument_eqs[target_instrument]['bands']
        
        # Apply EQ morphing
        morphed_audio = self._apply_eq_morphing(audio, target_eq, bands, strength)
        
        return torch.tensor(morphed_audio).unsqueeze(0)
    
    def _apply_eq_morphing(self, audio, target_eq, bands, strength):
        morphed_audio = audio.copy()
        
        # Apply strong EQ for each band
        for i, ((low_freq, high_freq), target_gain) in enumerate(zip(bands, target_eq)):
            try:
                # Create aggressive gain
                base_gain = self._get_base_gain(i, target_gain)
                final_gain = 1.0 + (base_gain - 1.0) * strength
                
                # Apply bandpass filter and gain
                nyquist = 12000
                low_norm = max(0.01, min(0.98, low_freq / nyquist))
                high_norm = max(low_norm + 0.01, min(0.99, high_freq / nyquist))
                
                b, a = butter(4, [low_norm, high_norm], btype='band')
                band_signal = filtfilt(b, a, audio)
                
                # Mix the gained band signal back
                morphed_audio = morphed_audio + band_signal * (final_gain - 1.0) * 0.5
                
            except Exception as e:
                print(f"Error in band {i}: {e}")
                continue
        
        # Apply instrument-specific processing
        morphed_audio = self._apply_instrument_processing(morphed_audio, target_eq, strength)
        
        return morphed_audio
    
    def _get_base_gain(self, band_index, target_gain):
        # More aggressive base gains
        gains = [2.5, 2.0, 1.8, 2.2, 2.8, 2.3]  # Per band
        return gains[band_index] * (0.5 + target_gain)
    
    def _apply_instrument_processing(self, audio, target_eq, strength):
        # Determine instrument type from EQ signature
        bass_heavy = target_eq[0] > 0.7  # Strong bass
        mid_heavy = target_eq[2] > 0.6   # Strong mids
        high_heavy = target_eq[4] > 0.6  # Strong highs
        
        try:
            if bass_heavy and high_heavy:
                # Drum-like: boost bass and highs, cut mids
                audio = self._boost_frequency_range(audio, 60, 120, 1.5 + strength, strength)
                audio = self._boost_frequency_range(audio, 5000, 10000, 1.3 + strength * 0.8, strength)
                audio = self._boost_frequency_range(audio, 800, 1500, 0.7 - strength * 0.3, strength)
                
            elif mid_heavy:
                # Guitar-like: boost mids and presence
                audio = self._boost_frequency_range(audio, 400, 800, 1.4 + strength * 0.6, strength)
                audio = self._boost_frequency_range(audio, 1500, 3000, 1.6 + strength * 0.8, strength)
                
            else:
                # Piano-like: balanced with sparkle
                audio = self._boost_frequency_range(audio, 200, 400, 1.2 + strength * 0.4, strength)
                audio = self._boost_frequency_range(audio, 2000, 6000, 1.3 + strength * 0.5, strength)
                
        except Exception as e:
            print(f"Error in instrument processing: {e}")
        
        return audio
    
    def _boost_frequency_range(self, audio, low_freq, high_freq, gain, strength):
        try:
            nyquist = 12000
            low_norm = max(0.01, min(0.98, low_freq / nyquist))
            high_norm = max(low_norm + 0.01, min(0.99, high_freq / nyquist))
            
            b, a = butter(3, [low_norm, high_norm], btype='band')
            filtered = filtfilt(b, a, audio)
            
            # Apply gain
            boosted = audio + filtered * (gain - 1.0) * strength * 0.4
            return boosted
            
        except:
            return audio
    
    def create_morphing_sequence(self, source_path, target_instrument, output_dir, steps=4):
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating morphing sequence: {Path(source_path).name} -> {target_instrument}")
        
        # Save original
        try:
            original_audio, sr = torchaudio.load(source_path)
            original_processed = torch.tensor(self.preprocess_audio(original_audio, sr)).unsqueeze(0)
            original_path = os.path.join(output_dir, "00_original.wav")
            torchaudio.save(original_path, original_processed, 24000)
            print(f"Saved original: {original_path}")
        except Exception as e:
            print(f"Error saving original: {e}")
            return []
        
        # Create morphing sequence
        strengths = [0.4, 0.6, 0.8, 1.0]
        results = []
        
        for i, strength in enumerate(strengths):
            output_path = os.path.join(output_dir, f"morph_{i+1:02d}_strength_{strength:.1f}.wav")
            
            try:
                print(f"Generating morph at strength {strength}...")
                morphed_audio = self.morph_with_eq(source_path, target_instrument, strength)
                
                # Normalize carefully
                max_val = morphed_audio.abs().max().item()
                if max_val > 0:
                    morphed_audio = morphed_audio / max_val * 0.8
                
                torchaudio.save(output_path, morphed_audio, 24000)
                results.append((strength, output_path))
                print(f"SUCCESS: {output_path}")
                
            except Exception as e:
                print(f"FAILED at strength {strength}: {e}")
                import traceback
                traceback.print_exc()
        
        return results

def main():
    CSV_PATH = "aaron_xai4ae/approach_3/musical_instruments/dataset/4_instruments_data/Metadata_Train.csv"
    AUDIO_DIR = "aaron_xai4ae/approach_3/musical_instruments/dataset/4_instruments_data/Train_submission/Train_submission"
    OUTPUT_DIR = "aaron_xai4ae/approach_3/musical_instruments/results/reliable_morphing"
    
    print("RELIABLE MORPHING TEST")
    print("=" * 40)
    
    morpher = ReliableMorpher()
    
    try:
        eq_count = morpher.train(CSV_PATH, AUDIO_DIR, max_samples=3)
        print(f"Training result: {eq_count} instruments")
        
        if eq_count == 0:
            print("Training failed - no EQ curves created")
            return
            
    except Exception as e:
        print(f"Training error: {e}")
        return
    
    # Find test files
    df = pd.read_csv(CSV_PATH)
    
    def find_file(class_name):
        files = df[df['Class'] == class_name]['FileName'].tolist()
        for f in files[:2]:
            path = Path(AUDIO_DIR) / f
            if path.exists():
                return str(path)
        return None
    
    guitar_path = find_file('Sound_Guitar')
    piano_path = find_file('Sound_Piano')
    
    if not guitar_path or not piano_path:
        print("Could not find test files")
        return
    
    print(f"\nTest files:")
    print(f"Guitar: {Path(guitar_path).name}")
    print(f"Piano: {Path(piano_path).name}")
    
    # Test scenarios
    scenarios = [
        (guitar_path, 'Piano', 'Guitar_to_Piano_RELIABLE'),
        (piano_path, 'Guitar', 'Piano_to_Guitar_RELIABLE')
    ]
    
    for source_path, target_instrument, scenario_name in scenarios:
        print(f"\n" + "="*50)
        print(f"SCENARIO: {scenario_name}")
        print("="*50)
        
        scenario_dir = os.path.join(OUTPUT_DIR, scenario_name)
        
        try:
            results = morpher.create_morphing_sequence(
                source_path, target_instrument, scenario_dir, steps=4
            )
            
            if results:
                print(f"SUCCESS: Generated {len(results)} files")
                for strength, path in results:
                    print(f"  - {Path(path).name}")
            else:
                print("No files generated")
                
        except Exception as e:
            print(f"SCENARIO FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nResults should be in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
