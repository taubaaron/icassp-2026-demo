import os
import sys
import random
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import torch
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 220, "font.size": 11})

# ──────────────────────────────────────────────────────────────────────────────
# Avoid home quota issues with HF cache
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_HF_HOME = "/cs/labs/adiyoss/aarontaub/hf_cache"
for k, v in {
    "HF_HOME": DEFAULT_HF_HOME,
    "HF_HUB_CACHE": os.path.join(DEFAULT_HF_HOME, "hub"),
    "TRANSFORMERS_CACHE": os.path.join(DEFAULT_HF_HOME, "transformers"),
}.items():
    os.environ.setdefault(k, v)
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (GENRES)
# ──────────────────────────────────────────────────────────────────────────────
AUDIO_DIR = "aaron_xai4ae/approach_3/musical_genre/dataset/genres_30clips"
OUTPUT_DIR = "aaron_xai4ae/approach_3/other_models/codec_comparison/musical_genres/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_AUDIOS_PER_GENRE = 50
AUDIO_DURATION = 3.0  # seconds

MODEL_COLORS = {
    "DAC-24k": "#3B82F6",
    "EnCodec-24k": "#10B981",
    "SpeechTokenizer-16k": "#F59E0B",
}

# Optional filters
EXCLUDED_GENRES = [
    # 'pop',
    # 'rock',
]
INCLUDED_GENRES = [
    # 'classical', 'jazz', 'blues', 'metal', 'reggae'
]

GENRE_COLOR_PALETTE = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F8C471', '#82E0AA', '#F1948A', '#74B9FF', '#E84393',
    '#00B894', '#6C5CE7', '#FDCB6E', '#C44569', '#0ABDE3',
]

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers (genres)
# ──────────────────────────────────────────────────────────────────────────────
def filter_genres(all_genres, excluded_list, included_list):
    print(f"\n[{timestamp()}] Filtering genres…")
    print(f"Found {len(all_genres)} total genres: {sorted(all_genres)}")
    if included_list:
        filtered = [g for g in all_genres if g in included_list]
        missing = [g for g in included_list if g not in all_genres]
        if missing:
            print(f"[{timestamp()}]  ! Not found from inclusion list: {missing}")
    else:
        filtered = [g for g in all_genres if g not in excluded_list]
        if excluded_list:
            print(f"[{timestamp()}] Using EXCLUSION list: {excluded_list}")
    print(f"[{timestamp()}] Final genre list ({len(filtered)}): {sorted(filtered)}")
    return sorted(filtered)

def discover_genres(audio_dir, excluded_list=None, included_list=None):
    print(f"[{timestamp()}] Scanning directory: {audio_dir}")
    if not os.path.exists(audio_dir):
        print(f"[{timestamp()}]  ! Directory does not exist")
        return []
    all_genres = []
    for item in os.listdir(audio_dir):
        p = os.path.join(audio_dir, item)
        if os.path.isdir(p):
            try:
                files = os.listdir(p)
                audio_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.au'))]
                if audio_files:
                    all_genres.append(item)
            except PermissionError:
                print(f"[{timestamp()}]  ! Permission denied: {p}")
    return filter_genres(all_genres, excluded_list or [], included_list or [])

def load_balanced_metadata(audio_dir, num_clips_per_genre, excluded_list=None, included_list=None):
    genres = discover_genres(audio_dir, excluded_list, included_list)
    if not genres:
        print(f"[{timestamp()}]  ! No genres after filtering")
        return [], {}, []

    metadata, counts = [], {}
    for genre in genres:
        gdir = os.path.join(audio_dir, genre)
        try:
            files = os.listdir(gdir)
            audio_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.au'))]
            valid = []
            for f in audio_files:
                fp = os.path.join(gdir, f)
                if os.path.exists(fp) and os.path.getsize(fp) > 1000:
                    valid.append(f)
            counts[genre] = len(valid)
            random.shuffle(valid)
            chosen = valid[:min(num_clips_per_genre, len(valid))]
            for f in chosen:
                metadata.append({
                    "audio_path": os.path.join(genre, f),
                    "genre": genre,
                    "genre_short": genre,
                })
        except Exception as e:
            print(f"[{timestamp()}]  ! Error listing {gdir}: {e}")
            counts[genre] = 0

    valid_genres = [g for g in genres if counts.get(g, 0) > 0]
    print(f"\n[{timestamp()}] Summary:")
    print(f"Genres after filtering: {len(genres)}")
    print(f"Genres with valid files: {len(valid_genres)}")
    print(f"Total samples selected: {len(metadata)}")
    return metadata, counts, valid_genres

def generate_color_map(genres):
    return {g: GENRE_COLOR_PALETTE[i % len(GENRE_COLOR_PALETTE)] for i, g in enumerate(genres)}

# ──────────────────────────────────────────────────────────────────────────────
# Audio utils
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Feature builders
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Preprocess + DR + eval
# ──────────────────────────────────────────────────────────────────────────────
def test_preprocessing(features, labels):
    scalers = {
        'Standard Scaler': StandardScaler(),
        'Robust Scaler': RobustScaler(),
        'Power Transformer': PowerTransformer(method='yeo-johnson'),
    }
    present = sorted(set(labels))
    label_to_idx = {g: i for i, g in enumerate(present)}
    y_true = [label_to_idx[l] for l in labels]

    results = {}
    n_clusters = len(present)
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

def reduce_dimensions(features, labels):
    techniques = {}
    present = sorted(set(labels))
    label_to_idx = {g: i for i, g in enumerate(present)}
    y_true = [label_to_idx[l] for l in labels]

    n_samples, n_features = features.shape
    n_clusters = len(present)

    # LDA
    try:
        n_components = min(len(present) - 1, n_features, n_samples - 1)
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

# ──────────────────────────────────────────────────────────────────────────────
# Model runners
# ──────────────────────────────────────────────────────────────────────────────
def run_dac(metadata):
    print(f"\n[{timestamp()}] === Running DAC-24k ===")
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
        latent = z.squeeze(0).detach().cpu().numpy()
        rec = dict(entry); rec["latent_representation"] = latent.tolist()
        cached.append(rec)
        if (i+1) % 25 == 0:
            print(f"[{timestamp()}]   Processed {i+1}/{len(metadata)}")

    if not cached:
        print(f"[{timestamp()}]  ! No DAC data extracted")
        return None

    labels = [e["genre_short"] for e in cached]
    features = create_feature_representations_continuous(cached)
    return {"name": "DAC-24k", "labels": labels, "features": features}

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
            labels.append(entry["genre_short"])
        except Exception as e:
            failures += 1
        if (i+1) % 25 == 0:
            print(f"[{timestamp()}]   Processed {i+1}/{len(metadata)}")

    if not code_arrays:
        print(f"[{timestamp()}]  ! No EnCodec data extracted")
        return None

    n_q = code_arrays[0].shape[0]
    codebook_size = 1024
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
    repo_id, sub = "fnlp/SpeechTokenizer", "speechtokenizer_hubert_avg"

    print(f"[{timestamp()}]  Downloading model files to local HF cache…")
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
            labels.append(entry["genre_short"])
        except Exception as e:
            failures += 1
        if (i+1) % 25 == 0:
            print(f"[{timestamp()}]   Processed {i+1}/{len(metadata)}")

    if not code_arrays:
        print(f"[{timestamp()}]  ! No SpeechTokenizer data extracted")
        return None

    features = create_feature_representations_discrete(code_arrays, n_q, codebook_size)
    return {"name": "SpeechTokenizer-16k", "labels": labels, "features": features}

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation orchestration
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(model_name, labels, feature_dict):
    print(f"\n[{timestamp()}] Evaluating {model_name} …")
    all_results = []
    best_result = None
    combo_id = 0

    for r_name, features in feature_dict.items():
        print(f"[{timestamp()}]   • Representation: {r_name} (shape {features.shape})")
        preproc = test_preprocessing(features, labels)
        for p_name, p_res in preproc.items():
            dimred = reduce_dimensions(p_res["features"], labels)
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
    return {
        "EnCodec-24k": "Meta EnCodec neural audio codec (24 kHz), RVQ discrete codes.",
        "DAC-24k": "Descript Audio Codec (24 kHz), continuous encoder latents z.",
        "SpeechTokenizer-16k": "FNLP SpeechTokenizer (16 kHz), multi-layer RVQ tokens."
    }.get(model_name, "Unknown architecture")

# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────
def create_performance_visualizations(detailed_df, best_df, output_dir):
    Path(os.path.join(output_dir, 'visualizations')).mkdir(parents=True, exist_ok=True)

    # Box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=detailed_df, x='Model', y='Silhouette Score', palette=MODEL_COLORS)
    plt.title('Distribution of Silhouette Scores Across All Configurations', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Model Architecture'); plt.ylabel('Silhouette Score')
    for i, model in enumerate(detailed_df['Model'].unique()):
        md = detailed_df[detailed_df['Model'] == model]['Silhouette Score']
        if len(md): plt.text(i, md.mean() + 0.01, f'μ={md.mean():.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'visualizations', 'score_distribution_boxplot.png'), dpi=300, bbox_inches='tight'); plt.close()

    # Heatmap
    pivot = detailed_df.pivot_table(values='Silhouette Score', index='Preprocessing', columns='Dimensionality Reduction', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, cmap='RdYlBu_r', center=0.5, square=True, linewidths=0.5, fmt='.3f')
    plt.title('Average Silhouette Scores by Preprocessing and Dimensionality Reduction', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Dimensionality Reduction Technique'); plt.ylabel('Preprocessing Method')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'visualizations', 'technique_performance_heatmap.png'), dpi=300, bbox_inches='tight'); plt.close()

    # Feature representation bars
    feature_perf = detailed_df.groupby(['Model', 'Feature Representation'])['Silhouette Score'].mean().reset_index()
    plt.figure(figsize=(14, 8))
    bars = sns.barplot(data=feature_perf, x='Feature Representation', y='Silhouette Score', hue='Model', palette=MODEL_COLORS)
    plt.title('Performance by Feature Representation', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Feature Representation'); plt.ylabel('Average Silhouette Score')
    plt.xticks(rotation=45, ha='right'); plt.legend(title='Model', loc='upper right')
    for container in bars.containers:
        bars.bar_label(container, fmt='%.3f', fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'visualizations', 'feature_representation_comparison.png'), dpi=300, bbox_inches='tight'); plt.close()

def create_side_by_side_scatter_plots(best_results, output_dir, genres_for_colors=None):
    n_models = len(best_results)
    if n_models == 0: return
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
    if n_models == 1: axes = [axes]

    # unified color map
    all_labels = []
    for r in best_results.values(): all_labels.extend(r['labels'])
    uniq = sorted(set(all_labels))
    cmap = generate_color_map(genres_for_colors or uniq)

    for idx, (model_name, result) in enumerate(best_results.items()):
        X = result['features']; y = result['labels']
        coords = PCA(n_components=2).fit_transform(X) if X.shape[1] > 2 else X
        for g in uniq:
            mask = (np.array(y) == g)
            if np.any(mask):
                axes[idx].scatter(coords[mask, 0], coords[mask, 1],
                                  color=cmap[g], alpha=0.75, s=50,
                                  edgecolors='black', linewidth=0.5, label=g.title())
        axes[idx].set_title(f'{model_name}\nSilhouette: {result["silhouette"]:.3f}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('PC1'); axes[idx].set_ylabel('PC2'); axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(title='Genre', fontsize=8)
    Path(os.path.join(output_dir, 'visualizations')).mkdir(parents=True, exist_ok=True)
    plt.suptitle('Best Genre Separation by Model', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'visualizations', 'best_separations_comparison.png'), dpi=300, bbox_inches='tight'); plt.close()

def create_comprehensive_dashboard(detailed_df, best_df, best_results, output_dir):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Performance Comparison', 'Configuration Analysis', 'Score Distribution', 'Best Separation (example)'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "box"}, {"type": "scatter"}]]
    )
    # Best scores
    fig.add_trace(go.Bar(x=best_df['Model'], y=best_df['Best Silhouette Score'],
                         marker_color=[MODEL_COLORS.get(m, '#6366F1') for m in best_df['Model']],
                         showlegend=False), row=1, col=1)
    # All configs scatter
    for m in detailed_df['Model'].unique():
        mdf = detailed_df[detailed_df['Model'] == m]
        fig.add_trace(go.Scatter(x=mdf.index, y=mdf['Silhouette Score'], mode='markers',
                                 name=f'{m} Configs', marker=dict(color=MODEL_COLORS.get(m, '#6366F1'), size=6)),
                      row=1, col=2)
    # Box plots
    for m in detailed_df['Model'].unique():
        fig.add_trace(go.Box(y=detailed_df[detailed_df['Model'] == m]['Silhouette Score'],
                             name=m, marker_color=MODEL_COLORS.get(m, '#6366F1')),
                      row=2, col=1)

    # one best separation
    if len(best_results):
        first = list(best_results.keys())[0]
        X = best_results[first]['features']; y = best_results[first]['labels']
        coords = PCA(n_components=2).fit_transform(X) if X.shape[1] > 2 else X
        uniq = sorted(set(y)); cmap = generate_color_map(uniq)
        for g in uniq:
            mask = (np.array(y) == g)
            if np.any(mask):
                fig.add_trace(go.Scatter(x=coords[mask, 0], y=coords[mask, 1],
                                         mode='markers', name=g.title(),
                                         marker=dict(color=cmap[g], size=8)),
                              row=2, col=2)

    fig.update_layout(title_text="Neural Audio Codec Model Comparison — Music Genres",
                      title_x=0.5, title_font_size=20, showlegend=True, height=800, template="plotly_white")
    Path(os.path.join(output_dir, 'visualizations')).mkdir(parents=True, exist_ok=True)
    fig.write_html(os.path.join(output_dir, 'visualizations', 'interactive_dashboard.html'))

def generate_html_report(detailed_df, best_df, summary_stats, genres, output_dir):
    html = f"""
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neural Audio Codec Model Comparison Report - Music Genre Analysis</title>
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
.genre-list {{display:grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap:8px; margin:10px 0}}
.genre-item {{background:#f8f9fa; padding:6px; border-radius:4px; text-align:center; font-size:0.85em; text-transform:capitalize}}
.badge {{display:inline-block; background:#eef2ff; color:#3730a3; padding:4px 10px; border-radius:9999px; font-weight:600}}
.footer {{text-align:center; margin-top:40px; padding-top:20px; border-top:1px solid #ddd; color:#7f8c8d}}
</style></head><body><div class="container">
<h1>Neural Audio Codec Model Comparison<br>Music Genre Analysis</h1>

<p><span class="badge">{AUDIO_DURATION}s segments · up to {MAX_AUDIOS_PER_GENRE} samples / genre · {len(genres)} genres</span></p>

<h2>Genres ({len(genres)})</h2>
<div class="genre-list">
{''.join(f'<div class="genre-item">{g.title()}</div>' for g in genres)}
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

def create_comprehensive_report(all_results, best_results, genres, output_dir):
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
    # use the union of genre names found to stabilize colors
    all_genres_for_colors = sorted(set([g for model in best_results.values() for g in model['labels']]))
    create_side_by_side_scatter_plots(best_results, output_dir, genres_for_colors=all_genres_for_colors)
    create_comprehensive_dashboard(detailed_df, best_df, best_results, output_dir)

    # HTML
    print(f"[{timestamp()}]  Writing HTML report …")
    generate_html_report(detailed_df, best_df, summary_stats, all_genres_for_colors, output_dir)

    print(f"\nComprehensive report generated in: {output_dir}")
    print("Files created:")
    print("- index.html")
    print("- data/detailed_results.csv")
    print("- data/model_summary_statistics.csv")
    print("- data/best_results_comparison.csv")
    print("- visualizations/*")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print(" MULTI-CODEC COMPARISON (EnCodec-24k, DAC-24k, SpeechTokenizer-16k)")
    print(" Dataset: musical_genre")
    print("=" * 78)

    # Filtering announce
    if INCLUDED_GENRES:
        print(f"[{timestamp()}] INCLUSION MODE → {INCLUDED_GENRES}")
    elif EXCLUDED_GENRES:
        print(f"[{timestamp()}] EXCLUSION MODE → {EXCLUDED_GENRES}")
    else:
        print(f"[{timestamp()}] NO FILTERS")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Metadata
    metadata, sample_counts, valid_genres = load_balanced_metadata(
        AUDIO_DIR, MAX_AUDIOS_PER_GENRE, EXCLUDED_GENRES, INCLUDED_GENRES
    )
    if not metadata or len(valid_genres) < 2:
        print(f"[{timestamp()}]  ! Need at least 2 genres with valid audio")
        return

    print(f"\n[{timestamp()}] Final genres ({len(valid_genres)}):")
    for i, g in enumerate(valid_genres):
        print(f"  {i+1:2d}. {g.title()}")

    # Seeds
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    all_results = {}
    best_results = {}

    print(f"\n{'='*78}\n RUNNING MODELS\n{'='*78}")

    # DAC
    dac_bundle = run_dac(metadata)
    if dac_bundle:
        model_results, best = evaluate_model(dac_bundle["name"], dac_bundle["labels"], dac_bundle["features"])
        all_results[dac_bundle["name"]] = model_results
        best_results[dac_bundle["name"]] = best

    # EnCodec
    enc_bundle = run_encodec(metadata)
    if enc_bundle:
        model_results, best = evaluate_model(enc_bundle["name"], enc_bundle["labels"], enc_bundle["features"])
        all_results[enc_bundle["name"]] = model_results
        best_results[enc_bundle["name"]] = best

    # SpeechTokenizer
    st_bundle = run_speechtokenizer(metadata)
    if st_bundle:
        model_results, best = evaluate_model(st_bundle["name"], st_bundle["labels"], st_bundle["features"])
        all_results[st_bundle["name"]] = model_results
        best_results[st_bundle["name"]] = best

    if not best_results:
        print(f"[{timestamp()}]  ! No models produced results.")
        return

    # Report
    print(f"\n{'='*78}\n GENERATING REPORT\n{'='*78}")
    create_comprehensive_report(all_results, best_results, valid_genres, OUTPUT_DIR)

    # Final summary
    sorted_best = sorted(best_results.items(), key=lambda kv: kv[1]['silhouette'], reverse=True)
    print(f"\nFINAL RANKING:")
    for i, (m, res) in enumerate(sorted_best, 1):
        print(f"{i}. {m:>18s} | Score: {res['silhouette']:.4f} | Config: {res['config']}")
    print(f"\nWINNER: {sorted_best[0][0]}  ({sorted_best[0][1]['silhouette']:.4f})")
    print(f"\nReport: {OUTPUT_DIR}/index.html")
    print(f"Interactive dashboard: {OUTPUT_DIR}/visualizations/interactive_dashboard.html")
    print(f"Total genres analyzed: {len(valid_genres)}")
    print(f"Max samples per genre: {MAX_AUDIOS_PER_GENRE}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
