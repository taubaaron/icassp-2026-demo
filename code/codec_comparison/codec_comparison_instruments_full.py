import os
import sys
import random
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter

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
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings('ignore')

# ===========================
# Style
# ===========================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 220, "font.size": 11})

# ===========================
# Hugging Face cache (avoid home quota)
# ===========================
DEFAULT_HF_HOME = "/cs/labs/adiyoss/aarontaub/hf_cache"
for k, v in {
    "HF_HOME": DEFAULT_HF_HOME,
    "HF_HUB_CACHE": os.path.join(DEFAULT_HF_HOME, "hub"),
    "TRANSFORMERS_CACHE": os.path.join(DEFAULT_HF_HOME, "transformers"),
}.items():
    os.environ.setdefault(k, v)
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

# ===========================
# Configuration
# ===========================
AUDIO_DIR = "aaron_xai4ae/approach_3/musical_instruments-full/dataset/music_dataset_30clips"
OUTPUT_DIR = "aaron_xai4ae/approach_3/other_models/codec_comparison/musical_instruments-full/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_AUDIOS_PER_INSTRUMENT = 30
AUDIO_DURATION = 3.0  # seconds

MODEL_ORDER = ["DAC-24k", "EnCodec-24k", "SpeechTokenizer-16k"]
MODEL_COLORS = {
    "DAC-24k": "#3B82F6",
    "EnCodec-24k": "#10B981",
    "SpeechTokenizer-16k": "#F59E0B",
}

# *** EXCLUSION / INCLUSION ***
EXCLUDED_INSTRUMENTS = [
    'cowbell', 'Harmonium', 'Drum_set',
    # 'Hi_Hats', 'Floor_Tom',
]
INCLUDED_INSTRUMENTS = [
    # 'Piano', 'Violin', 'Acoustic_Guitar', 'Trumpet', 'flute'
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

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

# ===========================
# Dataset utilities
# ===========================
def discover_instruments(audio_dir, excluded_list=None, included_list=None):
    print(f"[{timestamp()}] Scanning directory: {audio_dir}")
    if not os.path.exists(audio_dir):
        print(f"[{timestamp()}]  ! Directory {audio_dir} does not exist!")
        return []

    all_instruments = []
    for item in os.listdir(audio_dir):
        p = os.path.join(audio_dir, item)
        if os.path.isdir(p):
            try:
                files = os.listdir(p)
                audio_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
                if audio_files:
                    all_instruments.append(item)
            except PermissionError:
                print(f"[{timestamp()}]  ! Permission denied: {p}")
    all_instruments = sorted(all_instruments)

    # Filter
    filtered = filter_instruments(all_instruments, excluded_list or [], included_list or [])
    return filtered

def filter_instruments(all_instruments, excluded_list, included_list):
    print(f"\n[{timestamp()}] Filtering instruments…")
    print(f"Found {len(all_instruments)} total instruments: {all_instruments}")
    if included_list:
        filtered = [i for i in all_instruments if i in included_list]
        print(f"Using INCLUSION list: {included_list}")
        missing = [i for i in included_list if i not in all_instruments]
        if missing:
            print(f"WARNING: Not found from inclusion list: {missing}")
    else:
        filtered = [i for i in all_instruments if i not in excluded_list]
        if excluded_list:
            print(f"Using EXCLUSION list: {excluded_list}")
    print(f"Final instrument list ({len(filtered)}): {filtered}")
    return filtered

def load_balanced_metadata(audio_dir, num_clips_per_class, excluded_list=None, included_list=None):
    instruments = discover_instruments(audio_dir, excluded_list, included_list)
    if not instruments:
        print(f"[{timestamp()}]  ! No instruments after filtering")
        return [], {}, []

    metadata, counts = [], {}
    for instr in instruments:
        idir = os.path.join(audio_dir, instr)
        try:
            files = os.listdir(idir)
            audio_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
            valid = []
            for f in audio_files:
                fp = os.path.join(idir, f)
                if os.path.exists(fp) and os.path.getsize(fp) > 1000:
                    valid.append(f)
            counts[instr] = len(valid)
            random.shuffle(valid)
            chosen = valid[:min(num_clips_per_class, len(valid))]
            for f in chosen:
                metadata.append({
                    "audio_path": os.path.join(instr, f),
                    "instrument": instr,
                    "instrument_short": instr,
                })
        except Exception as e:
            print(f"[{timestamp()}]  ! Error listing {idir}: {e}")
            counts[instr] = 0

    valid_instruments = [i for i in instruments if counts.get(i, 0) > 0]
    print(f"\n[{timestamp()}] Summary:")
    print(f"Instruments after filtering: {len(instruments)}")
    print(f"Instruments with valid files: {len(valid_instruments)}")
    print(f"Total samples selected: {len(metadata)}")
    return metadata, counts, valid_instruments

def generate_color_map(instruments):
    return {instrument: EXTENDED_COLOR_PALETTE[i % len(EXTENDED_COLOR_PALETTE)]
            for i, instrument in enumerate(instruments)}

# ===========================
# Audio helpers
# ===========================
def resample_audio(file_path, target_sr):
    try:
        wav, sr = torchaudio.load(file_path)
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        peak = wav.abs().max()
        if float(peak) > 0:
            wav = wav / peak
        return wav
    except Exception as e:
        print(f"[{timestamp()}]  ! Error loading {file_path}: {e}")
        return None

def center_crop_or_pad(wav, target_len):
    T = wav.shape[-1]
    if T < target_len:
        return torch.nn.functional.pad(wav, (0, target_len - T))
    if T > target_len:
        mid = (T - target_len) // 2
        return wav[:, mid:mid + target_len]
    return wav

# ===========================
# Feature builders
# ===========================
def create_feature_representations_continuous(cached_data):
    print(f"[{timestamp()}]  Building continuous feature representations…")
    reps = {}
    simple, weighted, statistical, temporal = [], [], [], []
    max_len = 0

    for e in cached_data:
        latent = np.array(e["latent_representation"])  # (C, T')
        simple.append(latent.mean(axis=-1))

        n_frames = latent.shape[1]
        w = np.exp(-0.5 * ((np.arange(n_frames) - n_frames//2) / (n_frames//4 + 1e-8))**2)
        w = w / w.sum()
        weighted.append(np.average(latent, axis=1, weights=w))

        stats = np.concatenate([
            latent.mean(axis=-1),
            latent.std(axis=-1),
            np.percentile(latent, 25, axis=-1),
            np.percentile(latent, 75, axis=-1),
        ])
        statistical.append(stats)

        flat = latent.flatten()
        max_len = max(max_len, flat.shape[0])
        temporal.append(flat)

    max_len = min(max_len, len(cached_data)//2 if len(cached_data) > 1 else max_len, 128)
    t_fix = []
    for x in temporal:
        t_fix.append(x[:max_len] if len(x) >= max_len else np.pad(x, (0, max_len - len(x))))
    temporal_arr = np.stack(t_fix)

    reps["Simple Mean"] = np.stack(simple)
    reps["Weighted Mean"] = np.stack(weighted)
    reps["Statistical"] = np.stack(statistical)
    if temporal_arr.shape[1] > 64:
        n_comp = int(min(64, temporal_arr.shape[0] - 1, temporal_arr.shape[1]))
        n_comp = max(n_comp, 2)
        reps["Temporal PCA"] = PCA(n_components=n_comp, random_state=42).fit_transform(temporal_arr)
    else:
        reps["Temporal Raw"] = temporal_arr
    return reps

def _entropy(p, eps=1e-12): return float(-np.sum(p * np.log(p + eps)))

def create_feature_representations_discrete(code_arrays, n_q, codebook_size):
    print(f"[{timestamp()}]  Building discrete feature representations…")
    semantic_hists, acoustic_hists, all_layer_hists, stats_feats = [], [], [], []
    for codes in code_arrays:
        Tprime = codes.shape[1]
        sem = codes[0]
        sem_hist = np.bincount(sem, minlength=codebook_size).astype(np.float32) / max(1, Tprime)
        semantic_hists.append(sem_hist)

        if n_q > 1:
            ac = codes[1:].reshape(-1)
            ac_hist = np.bincount(ac, minlength=codebook_size).astype(np.float32) / max(1, ac.shape[0])
        else:
            ac_hist = np.zeros(codebook_size, dtype=np.float32)
        acoustic_hists.append(ac_hist)

        per, st = [], []
        for q in range(n_q):
            cq = codes[q]
            hq = np.bincount(cq, minlength=codebook_size).astype(np.float32) / max(1, Tprime)
            per.append(hq)
            sorted_p = np.sort(hq)[::-1]
            top1 = float(sorted_p[0]) if sorted_p.size else 0.0
            top3 = float(sorted_p[:3].sum()) if sorted_p.size >= 3 else float(sorted_p.sum())
            uniq = float((hq > 0).sum()) / codebook_size
            ent = _entropy(hq)
            st.extend([ent, top1, top3, uniq])
        all_layer_hists.append(np.concatenate(per))
        stats_feats.append(np.array(st, dtype=np.float32))

    reps = {
        "Semantic Hist": np.stack(semantic_hists),
        "Acoustic Hist": np.stack(acoustic_hists),
        "Code Stats": np.stack(stats_feats),
    }
    concat = np.stack(all_layer_hists)
    if concat.shape[1] > 8:
        n_comp = int(min(128, concat.shape[0]-1, concat.shape[1]))
        n_comp = max(n_comp, 2)
        reps["Code Hist PCA"] = PCA(n_components=n_comp, random_state=42).fit_transform(concat)
    else:
        reps["Code Hist PCA"] = concat
    return reps

# ===========================
# Preprocess + DR + eval
# ===========================
def test_preprocessing(features, labels, instruments):
    scalers = {
        'Standard Scaler': StandardScaler(),
        'Robust Scaler': RobustScaler(),
        'Power Transformer': PowerTransformer(method='yeo-johnson'),
    }
    present_instruments = sorted(set(labels))
    label_to_idx = {instr: i for i, instr in enumerate(present_instruments)}
    y_true = [label_to_idx[l] for l in labels]

    results = {}
    n_clusters = len(present_instruments)
    for name, scaler in scalers.items():
        try:
            X = scaler.fit_transform(features)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit(X)
            results[name] = {
                'features': X,
                'silhouette': silhouette_score(X, kmeans.labels_),
                'adjusted_rand': adjusted_rand_score(y_true, kmeans.labels_)
            }
        except Exception as e:
            print(f"[{timestamp()}]  ! Preprocessing '{name}' failed: {e}")
    return results

def reduce_dimensions(features, labels, instruments):
    techniques = {}
    present_instruments = sorted(set(labels))
    label_to_idx = {instr: i for i, instr in enumerate(present_instruments)}
    y_true = [label_to_idx[l] for l in labels]

    n_samples, n_features = features.shape
    n_clusters = len(present_instruments)

    # LDA
    try:
        n_components = min(len(present_instruments) - 1, n_features, n_samples - 1)
        if n_components >= 2:
            lda = LinearDiscriminantAnalysis(n_components=n_components)
            X = lda.fit_transform(features, y_true)
            techniques['LDA'] = {
                'features': X,
                'silhouette': silhouette_score(X, KMeans(n_clusters, random_state=42).fit_predict(X)),
                'transformer': lda
            }
            print(f"[{timestamp()}]   LDA OK ({n_components} comps)")
    except Exception as e:
        print(f"[{timestamp()}]  ! LDA failed: {e}")

    # Feature Selection + PCA
    try:
        k = min(n_features//2, n_samples//2, 50)
        if 10 < k < n_features:
            selector = SelectKBest(f_classif, k=k).fit(features, y_true)
            X_sel = selector.transform(features)
            pca_components = min(10, X_sel.shape[1], X_sel.shape[0]-1)
            if pca_components >= 2:
                pca = PCA(n_components=pca_components, random_state=42).fit(X_sel)
                X = pca.transform(X_sel)
                techniques['Feature Selection + PCA'] = {
                    'features': X,
                    'silhouette': silhouette_score(X, KMeans(n_clusters, random_state=42).fit_predict(X)),
                    'selector': selector,
                    'transformer': pca
                }
    except Exception as e:
        print(f"[{timestamp()}]  ! FS+PCA failed: {e}")

    # ICA
    try:
        k = min(20, n_features, n_samples - 1)
        if k >= 4:
            ica = FastICA(n_components=k, random_state=42, max_iter=1000)
            X = ica.fit_transform(features)
            techniques['ICA'] = {
                'features': X,
                'silhouette': silhouette_score(X, KMeans(n_clusters, random_state=42).fit_predict(X)),
                'transformer': ica
            }
    except Exception as e:
        print(f"[{timestamp()}]  ! ICA failed: {e}")

    # PCA
    try:
        pca_components = min(10, n_features, n_samples - 1)
        if pca_components >= 2:
            pca = PCA(n_components=pca_components, random_state=42).fit(features)
            X = pca.transform(features)
            techniques['PCA'] = {
                'features': X,
                'silhouette': silhouette_score(X, KMeans(n_clusters, random_state=42).fit_predict(X)),
                'transformer': pca
            }
    except Exception as e:
        print(f"[{timestamp()}]  ! PCA failed: {e}")

    return techniques

# ===========================
# Model runners
# ===========================
def run_dac(metadata):
    print(f"\n[{timestamp()}] === Running DAC-24k ===")
    out = {}
    try:
        import dac
    except Exception as e:
        print(f"[{timestamp()}]  ! Could not import dac: {e}")
        return None

    target_sr = 24000
    T = int(AUDIO_DURATION * target_sr)

    print(f"[{timestamp()}]  Loading DAC weights…")
    weights = dac.utils.download(model_type="24khz")
    model = dac.DAC.load(weights).eval().to(DEVICE)

    print(f"[{timestamp()}]  Extracting continuous latents (z)…")
    cached, failures = [], 0
    for i, entry in enumerate(metadata):
        path = os.path.join(AUDIO_DIR, entry["audio_path"])
        wav = resample_audio(path, target_sr)
        if wav is None:
            failures += 1
            continue
        wav = center_crop_or_pad(wav, T).to(DEVICE)
        with torch.no_grad():
            x = model.preprocess(wav.unsqueeze(0), sample_rate=target_sr)
            z, _, _, _, _ = model.encode(x)
        latent = z.squeeze(0).detach().cpu().numpy()  # (C, T')
        rec = dict(entry); rec["latent_representation"] = latent.tolist()
        cached.append(rec)
        if (i+1) % 25 == 0:
            print(f"[{timestamp()}]   Processed {i+1}/{len(metadata)}")

    if not cached:
        print(f"[{timestamp()}]  ! No DAC data extracted")
        return None

    labels = [e["instrument_short"] for e in cached]
    features = create_feature_representations_continuous(cached)
    out["name"] = "DAC-24k"
    out["labels"] = labels
    out["features"] = features
    return out

def run_encodec(metadata):
    print(f"\n[{timestamp()}] === Running EnCodec-24k ===")
    try:
        from encodec import EncodecModel
        from encodec.utils import convert_audio
    except Exception as e:
        print(f"[{timestamp()}]  ! Could not import EnCodec: {e}")
        return None

    target_sr = 24000
    T = int(AUDIO_DURATION * target_sr)
    model = EncodecModel.encodec_model_24khz().to(DEVICE)
    model.set_target_bandwidth(6.0)

    print(f"[{timestamp()}]  Extracting discrete codes…")
    code_arrays, labels, failures = [], [], 0
    for i, entry in enumerate(metadata):
        path = os.path.join(AUDIO_DIR, entry["audio_path"])
        try:
            wav, sr = torchaudio.load(path)
            wav = convert_audio(wav, sr, target_sr, model.channels).to(DEVICE)
            wav = center_crop_or_pad(wav, T)
            with torch.no_grad():
                enc = model.encode(wav.unsqueeze(0))  # list of frames
                codes = torch.cat([f[0] for f in enc], dim=-1)  # (B, n_q, T')
                codes = codes[0].detach().cpu().numpy().astype(np.int32)
            code_arrays.append(codes)
            labels.append(entry["instrument_short"])
        except Exception:
            failures += 1
        if (i+1) % 25 == 0:
            print(f"[{timestamp()}]   Processed {i+1}/{len(metadata)}")

    if not code_arrays:
        print(f"[{timestamp()}]  ! No EnCodec data extracted")
        return None

    n_q = code_arrays[0].shape[0]
    codebook_size = 1024  # typical
    features = create_feature_representations_discrete(code_arrays, n_q, codebook_size)
    return {"name": "EnCodec-24k", "labels": labels, "features": features}

def run_speechtokenizer(metadata):
    print(f"\n[{timestamp()}] === Running SpeechTokenizer-16k ===")
    try:
        from huggingface_hub import snapshot_download
        from speechtokenizer import SpeechTokenizer
        import json
    except Exception as e:
        print(f"[{timestamp()}]  ! Could not import SpeechTokenizer: {e}")
        return None

    target_sr = 16000
    T = int(AUDIO_DURATION * target_sr)

    print(f"[{timestamp()}]  Downloading model files (local cache)…")
    repo_id = "fnlp/SpeechTokenizer"
    sub = "speechtokenizer_hubert_avg"
    base = os.environ.get("HF_HOME", DEFAULT_HF_HOME)
    local_dir = os.path.join(base, "models", "SpeechTokenizer")
    ddir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{sub}/config.json", f"{sub}/SpeechTokenizer.pt"],
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    cfg_path = os.path.join(ddir, sub, "config.json")
    ckpt_path = os.path.join(ddir, sub, "SpeechTokenizer.pt")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    print(f"[{timestamp()}]  Loading SpeechTokenizer…")
    model = SpeechTokenizer.load_from_checkpoint(cfg_path, ckpt_path).eval().to(DEVICE)
    n_q = int(cfg.get("n_q", 8))
    codebook_size = int(cfg.get("codebook_size", 1024))

    print(f"[{timestamp()}]  Extracting discrete tokens…")
    code_arrays, labels, failures = [], [], 0
    for i, entry in enumerate(metadata):
        path = os.path.join(AUDIO_DIR, entry["audio_path"])
        try:
            wav = resample_audio(path, target_sr)
            if wav is None:
                failures += 1
                continue
            wav = center_crop_or_pad(wav, T).to(DEVICE)
            with torch.no_grad():
                codes = model.encode(wav.unsqueeze(0))  # (n_q, B, T')
                codes = codes[:, 0, :].detach().cpu().numpy().astype(np.int32)
            code_arrays.append(codes)
            labels.append(entry["instrument_short"])
        except Exception:
            failures += 1
        if (i+1) % 25 == 0:
            print(f"[{timestamp()}]   Processed {i+1}/{len(metadata)}")

    if not code_arrays:
        print(f"[{timestamp()}]  ! No ST data extracted")
        return None

    features = create_feature_representations_discrete(code_arrays, n_q, codebook_size)
    return {"name": "SpeechTokenizer-16k", "labels": labels, "features": features}

# ===========================
# Evaluation orchestration
# ===========================
def evaluate_model(model_name, labels, feature_dict, instruments):
    print(f"\n[{timestamp()}] Evaluating {model_name} …")
    all_results = []
    best_result = None
    combo_id = 0

    for r_name, features in feature_dict.items():
        print(f"[{timestamp()}]   • Representation: {r_name} (shape {features.shape})")
        preproc = test_preprocessing(features, labels, instruments)
        for p_name, p_res in preproc.items():
            dimred = reduce_dimensions(p_res["features"], labels, instruments)
            for d_name, d_res in dimred.items():
                combo_id += 1
                score = d_res["silhouette"]
                result = {
                    "feature_rep": r_name,
                    "preprocessing": p_name,
                    "dim_reduction": d_name,
                    "silhouette": score,
                    "adjusted_rand": p_res["adjusted_rand"],
                    "features": d_res["features"],
                    "labels": labels,
                    "config": f"{r_name} + {p_name} + {d_name}",
                    "latent_shape": str(d_res["features"].shape)
                }
                all_results.append(result)
                print(f"[{timestamp()}]     [{combo_id:02d}] {result['config']}: Sil={score:.3f}, ARI={result['adjusted_rand']:.3f}")
                if (best_result is None) or (score > best_result["silhouette"]):
                    best_result = result
    return all_results, best_result

def get_model_architecture_description(model_name):
    desc = {
        "EnCodec-24k": "Meta EnCodec neural audio codec (24 kHz), RVQ discrete codes.",
        "DAC-24k": "Descript Audio Codec (24 kHz), continuous encoder latents z.",
        "SpeechTokenizer-16k": "FNLP SpeechTokenizer (16 kHz), multi-layer RVQ tokens."
    }
    return desc.get(model_name, "Unknown architecture")

# ===========================
# Reporting
# ===========================
def create_performance_visualizations(detailed_df, best_df, output_dir):
    plt.figure(figsize=(12, 8))
    box_plot = sns.boxplot(data=detailed_df, x='Model', y='Silhouette Score', palette=MODEL_COLORS)
    plt.title('Distribution of Silhouette Scores Across All Configurations', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    for i, model in enumerate(detailed_df['Model'].unique()):
        md = detailed_df[detailed_df['Model'] == model]['Silhouette Score']
        plt.text(i, md.mean() + 0.01, f'μ={md.mean():.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    Path(os.path.join(output_dir, 'visualizations')).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'score_distribution_boxplot.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    pivot = detailed_df.pivot_table(
        values='Silhouette Score', index='Preprocessing', columns='Dimensionality Reduction', aggfunc='mean'
    )
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, cmap='RdYlBu_r', center=0.5, square=True, linewidths=0.5, fmt='.3f')
    plt.title('Average Silhouette Scores by Preprocessing and Dimensionality Reduction',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Dimensionality Reduction Technique', fontsize=12)
    plt.ylabel('Preprocessing Method', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'technique_performance_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    feature_perf = detailed_df.groupby(['Model', 'Feature Representation'])['Silhouette Score'].mean().reset_index()
    plt.figure(figsize=(14, 8))
    bars = sns.barplot(data=feature_perf, x='Feature Representation', y='Silhouette Score',
                       hue='Model', palette=MODEL_COLORS)
    plt.title('Performance by Feature Representation', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Feature Representation', fontsize=12)
    plt.ylabel('Average Silhouette Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model', loc='upper right')
    for container in bars.containers:
        bars.bar_label(container, fmt='%.3f', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'feature_representation_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_side_by_side_scatter_plots(best_results, output_dir):
    n_models = len(best_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
    if n_models == 1:
        axes = [axes]

    all_labels = []
    for r in best_results.values(): all_labels.extend(r['labels'])
    uniq_instr = sorted(set(all_labels))
    instr_cmap = generate_color_map(uniq_instr)

    for idx, (model_name, result) in enumerate(best_results.items()):
        features = result['features']; labels = result['labels']
        if features.shape[1] > 2:
            pca = PCA(n_components=2); coords_2d = pca.fit_transform(features)
            xlabel = f'PC1'; ylabel = f'PC2'
        else:
            coords_2d = features; xlabel = 'Dim 1'; ylabel = 'Dim 2'

        for inst in uniq_instr:
            mask = np.array(labels) == inst
            if np.any(mask):
                axes[idx].scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                                  color=instr_cmap[inst], alpha=0.7, s=50,
                                  edgecolors='black', linewidth=0.5, label=inst)
        axes[idx].set_title(f'{model_name}\nSilhouette: {result["silhouette"]:.3f}',
                            fontsize=14, fontweight='bold')
        axes[idx].set_xlabel(xlabel); axes[idx].set_ylabel(ylabel); axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(title='Instrument', fontsize=8)

    plt.suptitle('Best Instrument Separation by Model', fontsize=16, fontweight='bold', y=1.02)
    Path(os.path.join(output_dir, 'visualizations')).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'best_separations_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_dashboard(detailed_df, best_df, best_results, output_dir):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Performance Comparison', 'Configuration Analysis',
                        'Score Distribution', 'Best Separations'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "box"}, {"type": "scatter"}]]
    )

    # Best scores comparison
    fig.add_trace(
        go.Bar(x=best_df['Model'], y=best_df['Best Silhouette Score'],
               marker_color=[MODEL_COLORS.get(m, '#6366F1') for m in best_df['Model']],
               name='Best Scores', showlegend=False),
        row=1, col=1
    )

    # All configurations scatter
    for model in detailed_df['Model'].unique():
        mdf = detailed_df[detailed_df['Model'] == model]
        fig.add_trace(
            go.Scatter(x=mdf.index, y=mdf['Silhouette Score'], mode='markers',
                       name=f'{model} Configs', marker=dict(color=MODEL_COLORS.get(model, '#6366F1'), size=6)),
            row=1, col=2
        )

    # Box plots
    for model in detailed_df['Model'].unique():
        fig.add_trace(
            go.Box(y=detailed_df[detailed_df['Model'] == model]['Silhouette Score'],
                   name=model, marker_color=MODEL_COLORS.get(model, '#6366F1')),
            row=2, col=1
        )

    # Best separation of first model
    if len(best_results):
        first_model = list(best_results.keys())[0]
        features = best_results[first_model]['features']
        labels = best_results[first_model]['labels']
        if features.shape[1] > 2:
            coords_2d = PCA(n_components=2).fit_transform(features)
        else:
            coords_2d = features
        uniq_instr = sorted(set(labels))
        instr_cmap = generate_color_map(uniq_instr)
        for inst in uniq_instr:
            mask = np.array(labels) == inst
            if np.any(mask):
                fig.add_trace(
                    go.Scatter(x=coords_2d[mask, 0], y=coords_2d[mask, 1], mode='markers',
                               name=inst, marker=dict(color=instr_cmap[inst], size=8)),
                    row=2, col=2
                )

    fig.update_layout(title_text="Neural Audio Codec Model Comparison Dashboard",
                      title_x=0.5, title_font_size=20, showlegend=True, height=800, template="plotly_white")
    Path(os.path.join(output_dir, 'visualizations')).mkdir(parents=True, exist_ok=True)
    fig.write_html(os.path.join(output_dir, 'visualizations', 'interactive_dashboard.html'))

def generate_html_report(detailed_df, best_df, summary_stats, best_results, instruments, output_dir):
    html = f"""
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neural Audio Codec Model Comparison Report - Multi-Instrument Analysis</title>
<style>
body {{font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height:1.6; margin:0; padding:20px; background:#f5f5f5; color:#333}}
.container {{max-width:1200px; margin:0 auto; background:white; padding:30px; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.1)}}
h1 {{color:#2c3e50; text-align:center; border-bottom:3px solid #3498db; padding-bottom:10px; margin-bottom:30px}}
h2 {{color:#34495e; border-left:4px solid #3498db; padding-left:15px; margin-top:30px}}
.image-grid {{display:grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap:20px; margin:20px 0}}
.image-container {{text-align:center; background:#f8f9fa; padding:15px; border-radius:8px}}
img {{max-width:100%; height:auto; border-radius:5px; box-shadow:0 2px 4px rgba(0,0,0,0.1)}}
.table {{width:100%; border-collapse:collapse; margin:20px 0; background:white}}
.table th, .table td {{padding:12px; text-align:left; border-bottom:1px solid #ddd}}
.table th {{background:#3498db; color:white; font-weight:bold}}
.instrument-list {{display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:10px 0}}
.instrument-item {{background:#f8f9fa; padding:8px; border-radius:5px; text-align:center; font-size:0.9em}}
.badge {{display:inline-block; background:#eef2ff; color:#3730a3; padding:4px 10px; border-radius:9999px; font-weight:600}}
.footer {{text-align:center; margin-top:40px; padding-top:20px; border-top:1px solid #ddd; color:#7f8c8d}}
</style></head><body><div class="container">
<h1>Neural Audio Codec Model Comparison<br>Multi-Instrument Analysis</h1>

<p><span class="badge">{AUDIO_DURATION}s segments · up to {MAX_AUDIOS_PER_INSTRUMENT} samples / instrument · {len(instruments)} instruments</span></p>

<h2>Instruments ({len(instruments)})</h2>
<div class="instrument-list">
{''.join(f'<div class="instrument-item">{ins}</div>' for ins in instruments)}
</div>

<h2>Best Results by Model</h2>
{best_df.to_html(index=False, classes='table')}

<h2>Statistical Summary</h2>
{summary_stats.to_html(classes='table')}

<h2>Visualizations</h2>
<div class="image-grid">
  <div class="image-container"><h4>Best Separations (Side by Side)</h4><img src="visualizations/best_separations_comparison.png"></div>
  <div class="image-container"><h4>Score Distribution by Model</h4><img src="visualizations/score_distribution_boxplot.png"></div>
  <div class="image-container"><h4>Feature Representation Performance</h4><img src="visualizations/feature_representation_comparison.png"></div>
  <div class="image-container"><h4>Technique Performance Heatmap</h4><img src="visualizations/technique_performance_heatmap.png"></div>
</div>

<p>Interactive dashboard: <code>visualizations/interactive_dashboard.html</code></p>

<div class="footer">
  <p>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
</div></body></html>
"""
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)

def create_comprehensive_report(all_results, best_results, instruments, output_dir):
    Path(os.path.join(output_dir, 'visualizations')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'data')).mkdir(parents=True, exist_ok=True)

    # Detailed table
    detailed_rows = []
    for model_name, model_results in all_results.items():
        for r in model_results:
            detailed_rows.append({
                'Model': model_name,
                'Feature Representation': r['feature_rep'],
                'Preprocessing': r['preprocessing'],
                'Dimensionality Reduction': r['dim_reduction'],
                'Silhouette Score': r['silhouette'],
                'Adjusted Rand Index': r['adjusted_rand'],
            })
    detailed_df = pd.DataFrame(detailed_rows).sort_values(['Model', 'Silhouette Score'], ascending=[True, False])
    detailed_df.to_csv(os.path.join(output_dir, 'data', 'detailed_results.csv'), index=False)

    # Summary stats
    summary_stats = detailed_df.groupby('Model')['Silhouette Score'].agg(['mean', 'std', 'min', 'max', 'count']).round(4)
    summary_stats.columns = ['Mean Score', 'Std Dev', 'Min Score', 'Max Score', 'Configurations Tested']
    summary_stats.to_csv(os.path.join(output_dir, 'data', 'model_summary_statistics.csv'))

    # Best results
    best_df = pd.DataFrame([{
        'Model': model,
        'Architecture': get_model_architecture_description(model),
        'Best Configuration': res['config'],
        'Best Silhouette Score': res['silhouette'],
        'Feature Representation': res['feature_rep'],
        'Preprocessing': res['preprocessing'],
        'Dimensionality Reduction': res['dim_reduction'],
        'Latent Shape': res['latent_shape'],
    } for model, res in best_results.items()]).sort_values('Best Silhouette Score', ascending=False)
    best_df.to_csv(os.path.join(output_dir, 'data', 'best_results_comparison.csv'), index=False)

    # Visuals
    print(f"[{timestamp()}]  Rendering visuals …")
    create_performance_visualizations(detailed_df, best_df, output_dir)
    create_side_by_side_scatter_plots(best_results, output_dir)
    create_comprehensive_dashboard(detailed_df, best_df, best_results, output_dir)

    # HTML
    print(f"[{timestamp()}]  Writing HTML report …")
    generate_html_report(detailed_df, best_df, summary_stats, best_results, instruments, output_dir)

    print(f"\nComprehensive report generated in: {output_dir}")
    print("Files created:")
    print("- index.html")
    print("- data/detailed_results.csv")
    print("- data/model_summary_statistics.csv")
    print("- data/best_results_comparison.csv")
    print("- visualizations/*")

# ===========================
# Main
# ===========================
def main():
    print("=" * 78)
    print(" MULTI-CODEC COMPARISON (EnCodec-24k, DAC-24k, SpeechTokenizer-16k)")
    print(" Dataset: musical_instruments-full")
    print("=" * 78)

    # Filtering announcement
    if INCLUDED_INSTRUMENTS:
        print(f"[{timestamp()}] INCLUSION MODE → {INCLUDED_INSTRUMENTS}")
    elif EXCLUDED_INSTRUMENTS:
        print(f"[{timestamp()}] EXCLUSION MODE → {EXCLUDED_INSTRUMENTS}")
    else:
        print(f"[{timestamp()}] NO FILTERS")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Metadata
    metadata, sample_counts, valid_instruments = load_balanced_metadata(
        AUDIO_DIR, MAX_AUDIOS_PER_INSTRUMENT, EXCLUDED_INSTRUMENTS, INCLUDED_INSTRUMENTS
    )
    if not metadata or len(valid_instruments) < 2:
        print(f"[{timestamp()}]  ! Need at least 2 instruments with valid audio")
        return

    print(f"\n[{timestamp()}] Final instruments ({len(valid_instruments)}):")
    for i, instr in enumerate(valid_instruments):
        print(f"  {i+1:2d}. {instr}")

    # Seeds
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    all_results = {}
    best_results = {}

    print(f"\n{'='*78}\n RUNNING MODELS\n{'='*78}")

    # DAC
    dac_bundle = run_dac(metadata)
    if dac_bundle:
        model_results, best = evaluate_model(dac_bundle["name"], dac_bundle["labels"], dac_bundle["features"], valid_instruments)
        all_results[dac_bundle["name"]] = model_results
        best_results[dac_bundle["name"]] = best

    # EnCodec
    enc_bundle = run_encodec(metadata)
    if enc_bundle:
        model_results, best = evaluate_model(enc_bundle["name"], enc_bundle["labels"], enc_bundle["features"], valid_instruments)
        all_results[enc_bundle["name"]] = model_results
        best_results[enc_bundle["name"]] = best

    # SpeechTokenizer
    st_bundle = run_speechtokenizer(metadata)
    if st_bundle:
        model_results, best = evaluate_model(st_bundle["name"], st_bundle["labels"], st_bundle["features"], valid_instruments)
        all_results[st_bundle["name"]] = model_results
        best_results[st_bundle["name"]] = best

    if not best_results:
        print(f"[{timestamp()}]  ! No models produced results.")
        return

    # Report
    print(f"\n{'='*78}\n GENERATING REPORT\n{'='*78}")
    create_comprehensive_report(all_results, best_results, valid_instruments, OUTPUT_DIR)

    # Final summary
    sorted_best = sorted(best_results.items(), key=lambda kv: kv[1]['silhouette'], reverse=True)
    print(f"\nFINAL RANKING:")
    for i, (m, res) in enumerate(sorted_best, 1):
        print(f"{i}. {m:>18s} | Score: {res['silhouette']:.4f} | Config: {res['config']}")
    print(f"\nWINNER: {sorted_best[0][0]}  ({sorted_best[0][1]['silhouette']:.4f})")
    print(f"\nReport: {OUTPUT_DIR}/index.html")
    print(f"Interactive dashboard: {OUTPUT_DIR}/visualizations/interactive_dashboard.html")
    print(f"Total instruments analyzed: {len(valid_instruments)}")
    print(f"Max samples per instrument: {MAX_AUDIOS_PER_INSTRUMENT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
