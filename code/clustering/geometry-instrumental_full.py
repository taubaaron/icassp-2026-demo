import os
import random
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import gaussian_kde

from encodec import EncodecModel

warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Add path for audiocraft
import sys
sys.path.append("/cs/labs/adiyoss/aarontaub/thesis/audiocraft/code")

from audiocraft.solvers import CompressionSolver
from audiocraft.utils import checkpoint
from audiocraft import models
from omegaconf import DictConfig
from audiocraft.models import builders

# --- Configuration ---
AUDIO_DIR = "aaron_xai4ae/approach_3/musical_instruments-full/dataset/music_dataset_30clips"
OUTPUT_DIR = "aaron_xai4ae/approach_3/musical_instruments-full/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_AUDIOS_PER_INSTRUMENT = 30
AUDIO_DURATION = 3.0

# Model paths
MODEL_PATHS = {
    "VAE": "aaron_xai4ae/trained_models/encodec_30hz_44100_vae_200k_steps.th",
    "3CB": "aaron_xai4ae/trained_models/encodec_30hz_44100_3cb_4096_200k_steps.th", 
    "AE": "aaron_xai4ae/trained_models/encodec_30hz_44100_autoencoder_200k_steps.th"
}

# Professional color schemes
MODEL_COLORS = {
    'VAE': '#9B59B6',
    '3CB': '#2ECC71', 
    'AE': '#E67E22'
}

# *** EXCLUSION LIST - Add instruments you want to exclude here ***
EXCLUDED_INSTRUMENTS = [
    'cowbell',          
    'Harmonium',        
    'Drum_set',         
    # 'Hi_Hats',        
    # 'Floor_Tom',      
]

# *** INCLUSION LIST - Alternative: specify only instruments you want to include ***
# If this list is not empty, only these instruments will be processed
INCLUDED_INSTRUMENTS = [
    # 'Piano',
    # 'Violin', 
    # 'Acoustic_Guitar',
    # 'Trumpet',
    # 'flute'
]

# Extended color palette for instruments
EXTENDED_COLOR_PALETTE = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD',
    '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA',
    '#F1948A', '#85C1E9', '#D7BDE2', '#A3E4D7', '#FAD7A0', '#D5DBDB',
    '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43', '#EE5A24',
    '#0ABDE3', '#006BA6', '#A3CB38', '#C44569', '#F8B500', '#6C5CE7',
    '#FD79A8', '#FDCB6E', '#E84393', '#74B9FF', '#00B894', '#E17055'
]


class CustomVAEWrapper(nn.Module):
    """Wrapper to make VAE model compatible with standard interface"""
    def __init__(self, model_state, cfg):
        super().__init__()
        self.cfg = cfg
        self.sample_rate = cfg.get('sample_rate', 44100)
        
        # Extract encoder from the model state
        self.encoder = self.build_encoder_from_state(model_state, cfg)
        
    def build_encoder_from_state(self, model_state, cfg):
        """Build encoder from model state dict"""
        try:
            # Create a base encodec model to get the encoder structure
            base_model = EncodecModel.encodec_model_24khz()
            
            # Extract encoder parameters from VAE state
            encoder_state = {}
            for key, value in model_state.items():
                if 'encoder' in key.lower():
                    # Try to map VAE encoder keys to base encoder keys
                    base_key = self.map_vae_key_to_base(key)
                    if base_key and base_key in base_model.encoder.state_dict():
                        if base_model.encoder.state_dict()[base_key].shape == value.shape:
                            encoder_state[base_key] = value
            
            if encoder_state:
                print(f"  Mapped {len(encoder_state)} encoder parameters")
                base_model.encoder.load_state_dict(encoder_state, strict=False)
                return base_model.encoder
            else:
                print("  No compatible encoder parameters found, using base encoder")
                return base_model.encoder
                
        except Exception as e:
            print(f"  Error building encoder: {e}")
            # Fallback to base encoder
            return EncodecModel.encodec_model_24khz().encoder
    
    def map_vae_key_to_base(self, vae_key):
        """Map VAE parameter names to base encodec parameter names"""
        # Remove common prefixes that might differ
        base_key = vae_key
        
        # Remove model prefix if present
        if base_key.startswith('model.'):
            base_key = base_key[6:]
        
        # Remove compression_model prefix if present
        if base_key.startswith('compression_model.'):
            base_key = base_key[18:]
            
        return base_key
    
    def forward(self, x):
        return self.encoder(x)


def load_vae_model(checkpoint_path, device):
    """Load VAE model with custom wrapper"""
    try:
        print("  Loading VAE model with custom wrapper...")
        
        # Load checkpoint
        state = torch.load(checkpoint_path, map_location=device)
        
        if 'xp.cfg' not in state or 'best_state' not in state:
            print("  Missing required checkpoint components")
            return None
            
        cfg = DictConfig(state['xp.cfg'])
        model_state = state['best_state']['model']
        
        print(f"  Found {len(model_state)} parameters in checkpoint")
        
        # Create custom wrapper
        wrapper = CustomVAEWrapper(model_state, cfg)
        wrapper = wrapper.to(device).eval()
        
        print("  VAE model wrapped successfully")
        return wrapper
        
    except Exception as e:
        print(f"  VAE loading failed: {e}")
        return None


def get_model(checkpoint_path, device="cpu"):
    """Enhanced model loading with VAE support"""
    try:
        state = torch.load(checkpoint_path, map_location=device)
        assert state is not None and 'xp.cfg' in state, f"Could not load compression model from ckpt: {checkpoint_path}"

        cfg = DictConfig(state['xp.cfg'])
        cfg.device = device

        if hasattr(cfg, 'compression_model'):
            print(f"  Model type: {cfg.compression_model}")
            
            # Handle VAE model with custom loader
            if cfg.compression_model == 'flat_codec_w_transformer':
                print("  Detected VAE model (flat_codec_w_transformer)")
                return load_vae_model(checkpoint_path, device)

        # For compatible models, use standard loading
        compression_model = builders.get_compression_model(cfg).to(device)
        assert compression_model.sample_rate == cfg.sample_rate, "Compression model sample rate should match"

        assert 'best_state' in state and state['best_state'] != {}
        assert 'exported' not in state, "When loading an exported checkpoint, use the //pretrained/ prefix."

        compression_model.load_state_dict(state['best_state']['model'])
        compression_model.eval()
        return compression_model
        
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None


def load_custom_model(model_path):
    """Load custom model with VAE support"""
    if not os.path.exists(model_path):
        print(f"  Model file does not exist: {model_path}")
        return None
        
    try:
        model = get_model(checkpoint_path=model_path, device=str(DEVICE))
        if model is not None:
            return model.eval()
        return None
    except Exception as e:
        print(f"  Failed to load model from {model_path}: {e}")
        return None


def test_model_encoding(model):
    """Test if model can encode audio properly"""
    try:
        dummy_audio = torch.randn(1, 1, 24000 * 3).to(DEVICE)
        with torch.no_grad():
            encoded = model.encoder(dummy_audio)
        print(f"  Model encoding test successful. Output shape: {encoded.shape}")
        return True
    except Exception as e:
        print(f"  Model encoding test failed: {e}")
        return False


def get_model_architecture_description(model_name):
    """Get detailed architecture description for each model"""
    descriptions = {
        'VAE': 'Variational Autoencoder with flat_codec_w_transformer architecture',
        '3CB': 'EnCodec with 3-level codebook quantization (4096 codes per level)',
        'AE': 'Standard autoencoder with EnCodec backbone'
    }
    return descriptions.get(model_name, 'Unknown architecture')


def filter_instruments(all_instruments, excluded_list, included_list):
    """Filter instruments based on inclusion/exclusion lists."""
    
    print(f"\nFiltering instruments...")
    print(f"Found {len(all_instruments)} total instruments: {all_instruments}")
    
    # If inclusion list is specified, use only those instruments
    if included_list:
        filtered_instruments = [instr for instr in all_instruments if instr in included_list]
        print(f"Using INCLUSION list: {included_list}")
        print(f"Instruments found from inclusion list: {filtered_instruments}")
        
        # Check for instruments in inclusion list that weren't found
        missing = [instr for instr in included_list if instr not in all_instruments]
        if missing:
            print(f"WARNING: These instruments from inclusion list were not found: {missing}")
    
    # Otherwise, use all instruments except those in exclusion list
    else:
        filtered_instruments = [instr for instr in all_instruments if instr not in excluded_list]
        if excluded_list:
            print(f"Using EXCLUSION list: {excluded_list}")
            excluded_found = [instr for instr in excluded_list if instr in all_instruments]
            excluded_not_found = [instr for instr in excluded_list if instr not in all_instruments]
            
            if excluded_found:
                print(f"Excluded instruments: {excluded_found}")
            if excluded_not_found:
                print(f"Instruments in exclusion list but not found: {excluded_not_found}")
        else:
            print("No exclusion list specified, using all instruments")
    
    print(f"Final instrument list ({len(filtered_instruments)}): {filtered_instruments}")
    return filtered_instruments


def discover_instruments(audio_dir, excluded_list=None, included_list=None):
    """Automatically discover instrument classes from subdirectories with filtering."""
    print(f"Scanning directory: {audio_dir}")
    
    if not os.path.exists(audio_dir):
        print(f"Error: Directory {audio_dir} does not exist!")
        return []
    
    all_instruments = []
    for item in os.listdir(audio_dir):
        item_path = os.path.join(audio_dir, item)
        if os.path.isdir(item_path):
            try:
                files = os.listdir(item_path)
                audio_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
                if audio_files:
                    all_instruments.append(item)
            except PermissionError:
                print(f"Permission denied accessing {item_path}")
                continue
    
    all_instruments = sorted(all_instruments)
    
    # Apply filtering
    filtered_instruments = filter_instruments(all_instruments, excluded_list or [], included_list or [])
    
    return filtered_instruments


def generate_color_map(instruments):
    """Generate a professional color map for instruments."""
    return {instrument: EXTENDED_COLOR_PALETTE[i % len(EXTENDED_COLOR_PALETTE)] 
            for i, instrument in enumerate(instruments)}


def categorize_instruments(instruments):
    """Categorize instruments into families for better analysis."""
    categories = {
        'String': ['Acoustic_Guitar', 'Bass_Guitar', 'Electro_Guitar', 'Banjo', 'Mandolin', 
                  'Ukulele', 'Violin', 'Dobro'],
        'Wind': ['flute', 'Clarinet', 'Horn', 'Trumpet', 'Trombone', 'Saxophone', 'Harmonica'],
        'Keyboard': ['Piano', 'Keyboard', 'Organ', 'Accordion', 'Harmonium'],
        'Percussion': ['Drum_set', 'Cymbals', 'Floor_Tom', 'Hi_Hats', 'Tambourine', 
                      'Shakers', 'cowbell', 'vibraphone']
    }
    
    instrument_to_category = {}
    for category, instr_list in categories.items():
        for instr in instr_list:
            if instr in instruments:
                instrument_to_category[instr] = category
    
    for instr in instruments:
        if instr not in instrument_to_category:
            instrument_to_category[instr] = 'Other'
    
    return instrument_to_category


def resample_audio(file_path, target_sr=24000):
    """Resample audio to target sample rate and normalize."""
    try:
        waveform, sr = torchaudio.load(file_path)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
        return waveform
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_balanced_metadata(audio_dir, num_clips_per_class, excluded_list=None, included_list=None):
    """Load metadata from directory structure with filtering."""
    
    # Discover and filter instruments
    instruments = discover_instruments(audio_dir, excluded_list, included_list)
    
    if not instruments:
        print("No instruments found after filtering!")
        return [], {}, []
    
    metadata = []
    sample_counts = {}
    
    for instrument in instruments:
        instrument_dir = os.path.join(audio_dir, instrument)
        
        try:
            all_files = os.listdir(instrument_dir)
            audio_files = [f for f in all_files 
                          if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
            
            valid_files = []
            for f in audio_files:
                file_path = os.path.join(instrument_dir, f)
                if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
                    valid_files.append(f)
            
            sample_counts[instrument] = len(valid_files)
            print(f"{instrument}: {len(valid_files)} valid files found")
            
            if len(valid_files) > 0:
                random.shuffle(valid_files)
                selected_files = valid_files[:min(num_clips_per_class, len(valid_files))]
                
                for f in selected_files:
                    metadata.append({
                        "audio_path": os.path.join(instrument, f),
                        "instrument": instrument,
                        "instrument_short": instrument,
                    })
            
        except Exception as e:
            print(f"Error processing {instrument}: {e}")
            sample_counts[instrument] = 0
            continue
    
    # Filter out instruments with no samples
    valid_instruments = [instr for instr in instruments if sample_counts.get(instr, 0) > 0]
    
    print(f"\nSummary:")
    print(f"Instruments after filtering: {len(instruments)}")
    print(f"Instruments with valid files: {len(valid_instruments)}")
    print(f"Total samples selected for analysis: {len(metadata)}")
    
    return metadata, sample_counts, valid_instruments


def extract_latents(metadata, audio_dir, model):
    """Extract latent representations from audio files with progress tracking."""
    cached = []
    failures = 0
    target_samples = int(AUDIO_DURATION * 24000)
    
    print("Extracting latent representations...")
    print(f"Processing {len(metadata)} audio files...")
    
    for i, entry in enumerate(metadata):
        try:
            path = os.path.join(audio_dir, entry["audio_path"])
            waveform = resample_audio(path)
            
            if waveform is None:
                failures += 1
                continue
                
            waveform = waveform.to(DEVICE)
            
            if waveform.shape[1] < target_samples:
                waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[1]))
            else:
                mid = (waveform.shape[1] - target_samples) // 2
                waveform = waveform[:, mid:mid + target_samples]
            
            with torch.no_grad():
                latent = model.encoder(waveform.unsqueeze(0)).cpu().numpy()
            
            entry["latent_representation"] = latent.tolist()
            cached.append(entry)
            
            if (i + 1) % 50 == 0:
                progress = (i + 1) / len(metadata) * 100
                print(f"Progress: {progress:.1f}% ({i + 1}/{len(metadata)} files)")
                
        except Exception as e:
            print(f"Error processing {entry['audio_path']}: {e}")
            failures += 1
            if failures > 50:
                print("Too many failures, stopping...")
                break
    
    print(f"Successfully processed {len(cached)} files, {failures} failures")
    return cached


def create_feature_representations(cached_data):
    """Create different feature representations from latent codes."""
    print("Creating feature representations...")
    
    reps = {}
    simple, weighted, statistical, temporal = [], [], [], []
    max_len = 0

    for e in cached_data:
        latent = np.array(e["latent_representation"]).squeeze(0)
        simple.append(latent.mean(axis=-1))

        n_frames = latent.shape[1]
        w = np.exp(-0.5 * ((np.arange(n_frames) - n_frames//2) / (n_frames//4))**2)
        w /= w.sum()
        weighted.append(np.average(latent, axis=1, weights=w))

        stats = np.concatenate([
            latent.mean(axis=-1),
            latent.std(axis=-1),
            np.percentile(latent, 25, axis=-1),
            np.percentile(latent, 75, axis=-1)
        ])
        statistical.append(stats)

        flat = latent.flatten()
        max_len = max(max_len, flat.shape[0])
        temporal.append(flat)

    max_len = min(max_len, len(cached_data) // 3, 100)

    temporal_fixed = []
    for x in temporal:
        if len(x) >= max_len:
            temporal_fixed.append(x[:max_len])
        else:
            temporal_fixed.append(np.pad(x, (0, max_len - len(x))))

    reps["Simple Mean"] = np.stack(simple)
    reps["Weighted Mean"] = np.stack(weighted)
    reps["Statistical"] = np.stack(statistical)
    temporal = np.stack(temporal_fixed)

    if temporal.shape[1] > 50:
        pca = PCA(n_components=min(50, temporal.shape[0] - 1))
        reps["Temporal PCA"] = pca.fit_transform(temporal)
    else:
        reps["Temporal Raw"] = temporal

    print(f"Created {len(reps)} feature representations:")
    for name, features in reps.items():
        print(f"  {name}: {features.shape}")
    
    return reps


def test_preprocessing(features, labels, instruments):
    """Test different preprocessing techniques."""
    scalers = {
        'Standard Scaler': StandardScaler(),
        'Robust Scaler': RobustScaler(),
        'Power Transformer': PowerTransformer(method='yeo-johnson'),
    }
    
    present_instruments = sorted(list(set(labels)))
    label_to_idx = {instr: i for i, instr in enumerate(present_instruments)}
    y_true = [label_to_idx[l] for l in labels]
    
    results = {}
    n_clusters = len(present_instruments)

    for name, scaler in scalers.items():
        try:
            X = scaler.fit_transform(features)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300).fit(X)
            results[name] = {
                'features': X,
                'silhouette': silhouette_score(X, kmeans.labels_),
                'adjusted_rand': adjusted_rand_score(y_true, kmeans.labels_)
            }
        except Exception as e:
            print(f"Error in preprocessing {name}: {e}")
            continue

    return results


def reduce_dimensions(features, labels, instruments):
    """Apply dimensionality reduction techniques."""
    techniques = {}
    
    present_instruments = sorted(list(set(labels)))
    label_to_idx = {instr: i for i, instr in enumerate(present_instruments)}
    y_true = [label_to_idx[l] for l in labels]
    
    n_samples, n_features = features.shape
    n_clusters = len(present_instruments)
    n_classes = len(present_instruments)

    # LDA
    try:
        n_components = min(n_classes - 1, n_features, n_samples - 1)
        if n_components >= 2:
            lda = LinearDiscriminantAnalysis(n_components=n_components)
            X = lda.fit_transform(features, y_true)
            techniques['LDA'] = {
                'features': X, 
                'silhouette': silhouette_score(X, KMeans(n_clusters, random_state=42, n_init=5).fit_predict(X)),
                'transformer': lda
            }
            print(f"LDA successful: {n_components} components")
    except Exception as e:
        print(f"LDA failed: {e}")

    # Feature Selection + PCA
    try:
        k = min(n_features//2, n_samples//2, 100)
        if 20 < k < n_features:
            selector = SelectKBest(f_classif, k=k).fit(features, y_true)
            X_sel = selector.transform(features)
            pca_components = min(25, X_sel.shape[1], X_sel.shape[0]-1)
            if pca_components >= 2:
                pca = PCA(n_components=pca_components).fit(X_sel)
                X = pca.transform(X_sel)
                techniques['Feature Selection + PCA'] = {
                    'features': X, 
                    'silhouette': silhouette_score(X, KMeans(n_clusters, random_state=42, n_init=5).fit_predict(X)),
                    'selector': selector,
                    'transformer': pca
                }
    except Exception as e:
        print(f"Feature Selection + PCA failed: {e}")

    # ICA
    try:
        k = min(30, n_features, n_samples - 1)
        if k >= 2:
            ica = FastICA(n_components=k, random_state=42, max_iter=500)
            X = ica.fit_transform(features)
            techniques['ICA'] = {
                'features': X, 
                'silhouette': silhouette_score(X, KMeans(n_clusters, random_state=42, n_init=5).fit_predict(X)),
                'transformer': ica
            }
    except Exception as e:
        print(f"ICA failed: {e}")

    # PCA
    try:
        pca_components = min(25, n_features, n_samples - 1)
        if pca_components >= 2:
            pca = PCA(n_components=pca_components).fit(features)
            X = pca.transform(features)
            techniques['PCA'] = {
                'features': X, 
                'silhouette': silhouette_score(X, KMeans(n_clusters, random_state=42, n_init=5).fit_predict(X)),
                'transformer': pca
            }
    except Exception as e:
        print(f"PCA failed: {e}")

    return techniques


def create_comprehensive_report(all_results, best_results, instruments, output_dir):
    """Create comprehensive comparison report with visualizations"""
    
    # Create results directory structure
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    
    # 1. Create detailed results table
    detailed_results = []
    for model_name, model_results in all_results.items():
        for result in model_results:
            detailed_results.append({
                'Model': model_name,
                'Feature Representation': result['feature_rep'],
                'Preprocessing': result['preprocessing'],
                'Dimensionality Reduction': result['dim_reduction'],
                'Silhouette Score': result['silhouette'],
                'Adjusted Rand Index': result['adjusted_rand']
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df = detailed_df.sort_values(['Model', 'Silhouette Score'], ascending=[True, False])
    detailed_df.to_csv(os.path.join(output_dir, 'data', 'detailed_results.csv'), index=False)
    
    # 2. Create summary statistics
    summary_stats = detailed_df.groupby('Model')['Silhouette Score'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).round(4)
    summary_stats.columns = ['Mean Score', 'Std Dev', 'Min Score', 'Max Score', 'Configurations Tested']
    summary_stats.to_csv(os.path.join(output_dir, 'data', 'model_summary_statistics.csv'))
    
    # 3. Create best results comparison
    best_df = pd.DataFrame([
        {
            'Model': model,
            'Architecture': get_model_architecture_description(model),
            'Best Configuration': result['config'],
            'Best Silhouette Score': result['silhouette'],
            'Feature Representation': result['feature_rep'],
            'Preprocessing': result['preprocessing'],
            'Dimensionality Reduction': result['dim_reduction'],
            'Latent Dimensions': result['latent_shape']
        }
        for model, result in best_results.items()
    ])
    best_df = best_df.sort_values('Best Silhouette Score', ascending=False)
    best_df.to_csv(os.path.join(output_dir, 'data', 'best_results_comparison.csv'), index=False)
    
    # 4. Create visualizations
    create_performance_visualizations(detailed_df, best_df, output_dir)
    create_side_by_side_scatter_plots(best_results, output_dir)
    create_comprehensive_dashboard(detailed_df, best_df, best_results, output_dir)
    
    # 5. Generate HTML report
    generate_html_report(detailed_df, best_df, summary_stats, best_results, instruments, output_dir)
    
    print(f"\nComprehensive report generated in: {output_dir}")
    print("Files created:")
    print("- index.html (Main report)")
    print("- data/detailed_results.csv")
    print("- data/model_summary_statistics.csv") 
    print("- data/best_results_comparison.csv")
    print("- visualizations/ (Multiple charts and plots)")


def create_performance_visualizations(detailed_df, best_df, output_dir):
    """Create comprehensive performance visualizations"""
    
    # Set style for professional plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Box plot of all configurations by model
    plt.figure(figsize=(12, 8))
    box_plot = sns.boxplot(data=detailed_df, x='Model', y='Silhouette Score', palette=MODEL_COLORS)
    plt.title('Distribution of Silhouette Scores Across All Configurations', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    
    # Add statistical annotations
    for i, model in enumerate(detailed_df['Model'].unique()):
        model_data = detailed_df[detailed_df['Model'] == model]['Silhouette Score']
        mean_score = model_data.mean()
        plt.text(i, mean_score + 0.01, f'μ={mean_score:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'score_distribution_boxplot.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap of preprocessing vs dimensionality reduction techniques
    pivot_table = detailed_df.pivot_table(
        values='Silhouette Score', 
        index='Preprocessing', 
        columns='Dimensionality Reduction',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap='RdYlBu_r', center=0.5, 
                square=True, linewidths=0.5, fmt='.3f')
    plt.title('Average Silhouette Scores by Preprocessing and Dimensionality Reduction', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Dimensionality Reduction Technique', fontsize=12)
    plt.ylabel('Preprocessing Method', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'technique_performance_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature representation comparison
    feature_performance = detailed_df.groupby(['Model', 'Feature Representation'])['Silhouette Score'].mean().reset_index()
    
    plt.figure(figsize=(14, 8))
    bars = sns.barplot(data=feature_performance, x='Feature Representation', y='Silhouette Score', hue='Model', palette=MODEL_COLORS)
    plt.title('Performance Comparison by Feature Representation Method', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Feature Representation', fontsize=12)
    plt.ylabel('Average Silhouette Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model', loc='upper right')
    
    # Add value labels on bars
    for container in bars.containers:
        bars.bar_label(container, fmt='%.3f', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'feature_representation_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_side_by_side_scatter_plots(best_results, output_dir):
    """Create side-by-side scatter plots of best separations"""
    
    n_models = len(best_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    # Generate color map for instruments
    all_labels = []
    for result in best_results.values():
        all_labels.extend(result['labels'])
    unique_instruments = sorted(list(set(all_labels)))
    instrument_color_map = generate_color_map(unique_instruments)
    
    for idx, (model_name, result) in enumerate(best_results.items()):
        features = result['features']
        labels = result['labels']
        
        # Apply PCA for 2D visualization if needed
        if features.shape[1] > 2:
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(features)
            explained_var = pca.explained_variance_ratio_
            xlabel = f'PC1 ({explained_var[0]:.1%} variance)'
            ylabel = f'PC2 ({explained_var[1]:.1%} variance)'
        else:
            coords_2d = features
            xlabel = 'Dimension 1'
            ylabel = 'Dimension 2'
        
        # Create scatter plot
        for inst in unique_instruments:
            mask = np.array(labels) == inst
            if np.any(mask):
                axes[idx].scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                                color=instrument_color_map[inst], label=inst, 
                                alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        axes[idx].set_title(f'{model_name} Model\nSilhouette Score: {result["silhouette"]:.3f}', 
                           fontsize=14, fontweight='bold')
        axes[idx].set_xlabel(xlabel, fontsize=12)
        axes[idx].set_ylabel(ylabel, fontsize=12)
        axes[idx].legend(title='Instrument', loc='best', fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Best Instrument Separation by Model Architecture', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'best_separations_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_comprehensive_dashboard(detailed_df, best_df, best_results, output_dir):
    """Create interactive dashboard using plotly"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Performance Comparison', 'Configuration Analysis', 
                       'Score Distribution', 'Best Separations'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "box"}, {"type": "scatter"}]]
    )
    
    # 1. Best scores comparison
    fig.add_trace(
        go.Bar(x=best_df['Model'], y=best_df['Best Silhouette Score'],
               marker_color=[MODEL_COLORS[model] for model in best_df['Model']],
               name='Best Scores', showlegend=False),
        row=1, col=1
    )
    
    # 2. All configurations scatter
    for model in detailed_df['Model'].unique():
        model_data = detailed_df[detailed_df['Model'] == model]
        fig.add_trace(
            go.Scatter(x=model_data.index, y=model_data['Silhouette Score'],
                      mode='markers', name=f'{model} Configs',
                      marker=dict(color=MODEL_COLORS[model], size=6)),
            row=1, col=2
        )
    
    # 3. Box plots
    for model in detailed_df['Model'].unique():
        model_data = detailed_df[detailed_df['Model'] == model]['Silhouette Score']
        fig.add_trace(
            go.Box(y=model_data, name=model, 
                  marker_color=MODEL_COLORS[model]),
            row=2, col=1
        )
    
    # 4. Best separation visualization (using first model as example)
    if best_results:
        first_model = list(best_results.keys())[0]
        features = best_results[first_model]['features']
        labels = best_results[first_model]['labels']
        
        if features.shape[1] > 2:
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(features)
        else:
            coords_2d = features
            
        unique_instruments = sorted(list(set(labels)))
        instrument_color_map = generate_color_map(unique_instruments)
        
        for inst in unique_instruments:
            mask = np.array(labels) == inst
            if np.any(mask):
                fig.add_trace(
                    go.Scatter(x=coords_2d[mask, 0], y=coords_2d[mask, 1],
                              mode='markers', name=inst,
                              marker=dict(color=instrument_color_map[inst], size=8)),
                    row=2, col=2
                )
    
    # Update layout
    fig.update_layout(
        title_text="Neural Audio Codec Model Comparison Dashboard",
        title_x=0.5,
        title_font_size=20,
        showlegend=True,
        height=800,
        template="plotly_white"
    )
    
    fig.write_html(os.path.join(output_dir, 'visualizations', 'interactive_dashboard.html'))


def generate_html_report(detailed_df, best_df, summary_stats, best_results, instruments, output_dir):
    """Generate comprehensive HTML report"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Neural Audio Codec Model Comparison Report - Multi-Instrument Analysis</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .summary-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .summary-card h3 {{
                margin: 0 0 10px 0;
                font-size: 1.2em;
            }}
            .summary-card .score {{
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .summary-card .config {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background-color: white;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .winner {{
                background-color: #2ecc71 !important;
                color: white;
                font-weight: bold;
            }}
            .image-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .image-container {{
                text-align: center;
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .methodology {{
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .key-findings {{
                background-color: #e8f5e8;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #27ae60;
                margin: 20px 0;
            }}
            .instrument-list {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
                margin: 10px 0;
            }}
            .instrument-item {{
                background-color: #f8f9fa;
                padding: 8px;
                border-radius: 5px;
                text-align: center;
                font-size: 0.9em;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Neural Audio Codec Model Comparison Report<br>Multi-Instrument Analysis</h1>
            
            <div class="key-findings">
                <h3>Executive Summary</h3>
                <p>This report presents a comprehensive comparison of three neural audio codec architectures for musical instrument clustering in latent space. The analysis evaluates {len(detailed_df)} different configurations across VAE, 3CB (3-level CodeBook), and AutoEncoder models using {MAX_AUDIOS_PER_INSTRUMENT} samples per instrument from {len(instruments)} instrument classes.</p>
                
                <h4>Analyzed Instruments ({len(instruments)} total):</h4>
                <div class="instrument-list">
    """
    
    # Add instrument list
    for instrument in instruments:
        html_content += f'<div class="instrument-item">{instrument}</div>'
    
    html_content += f"""
                </div>
            </div>
            
            <h2>Performance Overview</h2>
            <div class="summary-grid">
    """
    
    # Add summary cards for each model
    for _, row in best_df.iterrows():
        color = MODEL_COLORS.get(row['Model'], '#3498db')
        winner_class = 'winner' if row['Model'] == best_df.iloc[0]['Model'] else ''
        
        html_content += f"""
                <div class="summary-card {winner_class}" style="background: linear-gradient(135deg, {color} 0%, {color}aa 100%);">
                    <h3>{row['Model']} Model</h3>
                    <div class="score">{row['Best Silhouette Score']:.4f}</div>
                    <div class="config">{row['Best Configuration']}</div>
                    <div style="font-size: 0.8em; margin-top: 10px;">
                        {row['Architecture']}
                    </div>
                </div>
        """
    
    html_content += f"""
            </div>
            
            <h2>Detailed Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Silhouette Score</th>
                        <th>Feature Representation</th>
                        <th>Preprocessing</th>
                        <th>Dimensionality Reduction</th>
                        <th>Latent Shape</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add table rows
    for idx, (_, row) in enumerate(best_df.iterrows()):
        winner_class = 'winner' if idx == 0 else ''
        html_content += f"""
                    <tr class="{winner_class}">
                        <td>{idx + 1}</td>
                        <td>{row['Model']}</td>
                        <td>{row['Best Silhouette Score']:.4f}</td>
                        <td>{row['Feature Representation']}</td>
                        <td>{row['Preprocessing']}</td>
                        <td>{row['Dimensionality Reduction']}</td>
                        <td>{row['Latent Dimensions']}</td>
                    </tr>
        """
    
    html_content += f"""
                </tbody>
            </table>
            
            <h2>Statistical Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Mean Score</th>
                        <th>Std Dev</th>
                        <th>Min Score</th>
                        <th>Max Score</th>
                        <th>Configurations Tested</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for model, stats in summary_stats.iterrows():
        html_content += f"""
                    <tr>
                        <td>{model}</td>
                        <td>{stats['Mean Score']:.4f}</td>
                        <td>{stats['Std Dev']:.4f}</td>
                        <td>{stats['Min Score']:.4f}</td>
                        <td>{stats['Max Score']:.4f}</td>
                        <td>{int(stats['Configurations Tested'])}</td>
                    </tr>
        """
    
    html_content += f"""
                </tbody>
            </table>
            
            <div class="methodology">
                <h3>Methodology</h3>
                <p><strong>Dataset:</strong> {MAX_AUDIOS_PER_INSTRUMENT} samples per instrument class across {len(instruments)} instruments</p>
                <p><strong>Audio Processing:</strong> {AUDIO_DURATION}s segments, resampled to 24kHz, normalized</p>
                <p><strong>Feature Representations:</strong> Simple Mean, Weighted Mean, Statistical (mean/std/percentiles), Temporal PCA</p>
                <p><strong>Preprocessing:</strong> Standard Scaler, Robust Scaler, Power Transformer</p>
                <p><strong>Dimensionality Reduction:</strong> LDA, PCA, ICA, Feature Selection + PCA</p>
                <p><strong>Evaluation:</strong> Silhouette Score for clustering quality assessment</p>
            </div>
            
            <h2>Visualizations</h2>
            <div class="image-grid">
                <div class="image-container">
                    <h4>Best Instrument Separations</h4>
                    <img src="visualizations/best_separations_comparison.png" alt="Best Separations Comparison">
                </div>
                <div class="image-container">
                    <h4>Score Distribution by Model</h4>
                    <img src="visualizations/score_distribution_boxplot.png" alt="Score Distribution">
                </div>
                <div class="image-container">
                    <h4>Feature Representation Performance</h4>
                    <img src="visualizations/feature_representation_comparison.png" alt="Feature Representation Comparison">
                </div>
                <div class="image-container">
                    <h4>Technique Performance Heatmap</h4>
                    <img src="visualizations/technique_performance_heatmap.png" alt="Technique Performance Heatmap">
                </div>
            </div>
            
            <h2>Key Findings</h2>
            <div class="key-findings">
                <ul>
                    <li><strong>Best Performing Model:</strong> {best_df.iloc[0]['Model']} achieved the highest silhouette score of {best_df.iloc[0]['Best Silhouette Score']:.4f}</li>
                    <li><strong>Optimal Configuration:</strong> {best_df.iloc[0]['Best Configuration']} proved most effective</li>
                    <li><strong>Feature Representation:</strong> {best_df.iloc[0]['Feature Representation']} features provided the best discrimination</li>
                    <li><strong>Instruments Analyzed:</strong> {len(instruments)} different musical instruments</li>
                    <li><strong>Consistency:</strong> {summary_stats.loc[best_df.iloc[0]['Model'], 'Mean Score']:.4f} ± {summary_stats.loc[best_df.iloc[0]['Model'], 'Std Dev']:.4f} across all configurations</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>For detailed data and interactive visualizations, see the accompanying files in the data/ and visualizations/ directories.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'index.html'), 'w') as f:
        f.write(html_content)


def main():
    """Main execution function with multi-model comparison."""
    print("=" * 70)
    print("COMPREHENSIVE NEURAL AUDIO CODEC MODEL COMPARISON")
    print("=" * 70)
    
    # Print filtering configuration
    if INCLUDED_INSTRUMENTS:
        print(f"INCLUSION MODE: Only analyzing {len(INCLUDED_INSTRUMENTS)} instruments:")
        for instr in INCLUDED_INSTRUMENTS:
            print(f"  - {instr}")
    elif EXCLUDED_INSTRUMENTS:
        print(f"EXCLUSION MODE: Excluding {len(EXCLUDED_INSTRUMENTS)} instruments:")
        for instr in EXCLUDED_INSTRUMENTS:
            print(f"  - {instr}")
    else:
        print("NO FILTERING: Analyzing all available instruments")
    
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load metadata with filtering (same for all models)
    metadata, all_sample_counts, valid_instruments = load_balanced_metadata(
        AUDIO_DIR, MAX_AUDIOS_PER_INSTRUMENT, EXCLUDED_INSTRUMENTS, INCLUDED_INSTRUMENTS
    )
    
    if not metadata:
        print("No valid audio files found!")
        return
    
    if len(valid_instruments) < 2:
        print("Need at least 2 instruments with data for analysis!")
        return

    print(f"\nFinal instrument list for analysis ({len(valid_instruments)} instruments):")
    for i, instr in enumerate(valid_instruments):
        print(f"  {i+1:2d}. {instr}")

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    all_results = {}
    best_results = {}
    successful_models = 0
    failed_models = []

    print(f"\n{'='*70}")
    print("PROCESSING MODELS")
    print(f"{'='*70}")

    for model_name, model_path in MODEL_PATHS.items():
        print(f"\nProcessing model: {model_name}")
        print(f"{'─'*40}")
        
        try:
            # Load model
            model = load_custom_model(model_path)
            if model is None:
                print(f"Failed to load {model_name}, skipping...")
                failed_models.append(model_name)
                continue
            
            # Test model
            if not test_model_encoding(model):
                print(f"{model_name} failed encoding test, skipping...")
                failed_models.append(model_name)
                continue

            # Extract latents
            data = extract_latents(metadata, AUDIO_DIR, model)
            if not data:
                print(f"No data extracted for {model_name}, skipping...")
                failed_models.append(model_name)
                continue

            labels = [d['instrument_short'] for d in data]
            reps = create_feature_representations(data)

            model_results = []
            best_result = None

            print(f"\nTesting feature processing pipelines for {model_name}...")
            combination_count = 0
            
            # Test all combinations and store detailed results
            for r_name, features in reps.items():
                print(f"\nProcessing representation: {r_name} (shape: {features.shape})")
                preproc = test_preprocessing(features, labels, valid_instruments)
                
                for p_name, p_result in preproc.items():
                    dimred = reduce_dimensions(p_result["features"], labels, valid_instruments)
                    
                    for d_name, d_result in dimred.items():
                        combination_count += 1
                        score = d_result["silhouette"]
                        result = {
                            "feature_rep": r_name,
                            "preprocessing": p_name,
                            "dim_reduction": d_name,
                            "silhouette": score,
                            "adjusted_rand": p_result["adjusted_rand"],
                            "features": d_result["features"],
                            "labels": labels,
                            "config": f"{r_name} + {p_name} + {d_name}",
                            "latent_shape": str(d_result["features"].shape)
                        }
                        model_results.append(result)
                        
                        print(f"  [{combination_count:2d}] {result['config']}: {score:.3f}")
                        
                        if not best_result or score > best_result["silhouette"]:
                            best_result = result

            all_results[model_name] = model_results
            best_results[model_name] = best_result
            
            print(f"\nBest config for {model_name}: {best_result['config']} | Silhouette: {best_result['silhouette']:.3f}")
            successful_models += 1
                
            # Clean up GPU memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"Unexpected error processing {model_name}: {e}")
            failed_models.append(model_name)
            continue

    # Generate comprehensive report
    if best_results:
        print(f"\n{'='*70}")
        print("GENERATING COMPREHENSIVE REPORT")
        print(f"{'='*70}")
        
        create_comprehensive_report(all_results, best_results, valid_instruments, OUTPUT_DIR)
        
        # Print final summary
        sorted_results = sorted(best_results.items(), key=lambda x: x[1]['silhouette'], reverse=True)
        print(f"\nFINAL RANKING:")
        for i, (model, result) in enumerate(sorted_results, 1):
            print(f"{i}. {model:>8s} | Score: {result['silhouette']:.4f} | Config: {result['config']}")
        
        print(f"\nWINNER: {sorted_results[0][0]} with score {sorted_results[0][1]['silhouette']:.4f}")
        print(f"Successfully processed: {successful_models}/{len(MODEL_PATHS)} models")
        if failed_models:
            print(f"Failed models: {', '.join(failed_models)}")
        
        print(f"\nComprehensive report available at: {OUTPUT_DIR}/index.html")
        print(f"Total instruments analyzed: {len(valid_instruments)}")
        print(f"Total samples processed per model: {len(metadata)}")
        
    else:
        print("No models processed successfully!")


if __name__ == "__main__":
    main()
