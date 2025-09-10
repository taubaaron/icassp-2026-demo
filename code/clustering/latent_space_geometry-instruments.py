import os
import random
import pickle
import warnings
from pathlib import Path

import torch
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.manifold import TSNE

from encodec import EncodecModel

warnings.filterwarnings('ignore')

# --- Configuration ---
CSV_PATH = "aaron_xai4ae/approach_3/musical_instruments/dataset/4_instruments_data/Metadata_Train.csv"
AUDIO_DIR = "aaron_xai4ae/approach_3/musical_instruments/dataset/4_instruments_data/Train_submission/Train_submission"
OUTPUT_DIR = "aaron_xai4ae/approach_3/musical_instruments/enhanced_separation"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_AUDIOS_PER_INSTRUMENT = 60
AUDIO_DURATION = 3.0

INSTRUMENT_CLASSES = ['Sound_Guitar', 'Sound_Drum', 'Sound_Piano', 'Sound_Violin']
INSTRUMENT_COLOR_MAP = {
    'Guitar': '#FF0000',
    'Drum': '#00FF00',
    'Piano': '#0000FF',
    'Violin': '#FF8000'
}


def resample_audio(file_path, target_sr=24000):
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.abs().max() > 0:
        waveform = waveform / waveform.abs().max()
    return waveform


def load_balanced_metadata(csv_path, audio_dir, num_clips_per_class):
    df = pd.read_csv(csv_path)
    metadata = []

    for instr in INSTRUMENT_CLASSES:
        files = df[df['Class'] == instr]['FileName'].tolist()
        files = [f for f in files if (Path(audio_dir) / f).exists()]
        random.shuffle(files)
        for f in files[:num_clips_per_class]:
            metadata.append({
                "audio_path": f,
                "instrument": instr,
                "instrument_short": instr.replace('Sound_', ''),
            })

    return metadata


def extract_latents(metadata, audio_dir, model):
    cached = []
    failures = 0
    target_samples = int(AUDIO_DURATION * 24000)

    for entry in metadata:
        try:
            path = os.path.join(audio_dir, entry["audio_path"])
            waveform = resample_audio(path).to(DEVICE)
            if waveform.shape[1] < target_samples:
                waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[1]))
            else:
                mid = (waveform.shape[1] - target_samples) // 2
                waveform = waveform[:, mid:mid + target_samples]
            with torch.no_grad():
                latent = model.encoder(waveform.unsqueeze(0)).cpu().numpy()
            entry["latent_representation"] = latent.tolist()
            cached.append(entry)
        except Exception:
            failures += 1
            if failures > 5:
                break

    return cached


def create_feature_representations(cached_data):
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

    max_len = min(max_len, len(cached_data) // 2, 128)

    temporal_fixed = []
    for x in temporal:
        if len(x) >= max_len:
            temporal_fixed.append(x[:max_len])
        else:
            temporal_fixed.append(np.pad(x, (0, max_len - len(x))))
    temporal = np.stack(temporal_fixed)


    reps["simple_mean"] = np.stack(simple)
    reps["weighted_mean"] = np.stack(weighted)
    reps["statistical"] = np.stack(statistical)
    temporal = np.stack(temporal)

    if temporal.shape[1] > 64:
        pca = PCA(n_components=min(64, temporal.shape[0] - 1))
        reps["temporal_pca"] = pca.fit_transform(temporal)
    else:
        reps["temporal_raw"] = temporal

    return reps


def test_preprocessing(features, labels):
    scalers = {
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'PowerTransformer': PowerTransformer(method='yeo-johnson'),
    }
    y_true = [INSTRUMENT_CLASSES.index(f'Sound_{l}') for l in labels]
    results = {}

    for name, scaler in scalers.items():
        try:
            X = scaler.fit_transform(features)
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=20).fit(X)
            results[name] = {
                'features': X,
                'silhouette': silhouette_score(X, kmeans.labels_),
                'adjusted_rand': adjusted_rand_score(y_true, kmeans.labels_)
            }
        except Exception:
            continue

    return results


def reduce_dimensions(features, labels):
    techniques = {}
    y_true = [INSTRUMENT_CLASSES.index(f'Sound_{l}') for l in labels]
    n_samples, n_features = features.shape

    try:
        lda = LinearDiscriminantAnalysis(n_components=3)
        X = lda.fit_transform(features, y_true)
        techniques['LDA'] = {'features': X, 'silhouette': silhouette_score(X, KMeans(4).fit_predict(X))}
    except:
        pass

    try:
        k = min(n_features//2, n_samples//2, 50)
        if 10 < k < n_features:
            selector = SelectKBest(f_classif, k=k).fit(features, y_true)
            X_sel = selector.transform(features)
            pca = PCA(n_components=min(10, X_sel.shape[1], X_sel.shape[0]-1)).fit(X_sel)
            X = pca.transform(X_sel)
            techniques['FeatureSelect_PCA'] = {'features': X, 'silhouette': silhouette_score(X, KMeans(4).fit_predict(X))}
    except:
        pass

    try:
        k = min(20, n_features, n_samples - 1)
        if k >= 4:
            ica = FastICA(n_components=k, random_state=42, max_iter=1000)
            X = ica.fit_transform(features)
            techniques['ICA'] = {'features': X, 'silhouette': silhouette_score(X, KMeans(4).fit_predict(X))}
    except:
        pass

    try:
        pca = PCA(n_components=min(10, n_features, n_samples - 1)).fit(features)
        X = pca.transform(features)
        techniques['PCA'] = {'features': X, 'silhouette': silhouette_score(X, KMeans(4).fit_predict(X))}
    except:
        pass

    return techniques


def visualize(features, labels, method_name, out_dir):
    if len(features) < 10:
        return 0, 0

    pca_2d = PCA(n_components=2).fit(features)
    coords_2d = pca_2d.transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    for inst in INSTRUMENT_COLOR_MAP:
        mask = np.array(labels) == inst
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], color=INSTRUMENT_COLOR_MAP[inst], label=inst, alpha=0.7)
    ax.set_title(f'2D PCA - {method_name}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'pca_{method_name}.png'), dpi=300)
    plt.close()

    pca_3d = PCA(n_components=3).fit(features)
    coords_3d = pca_3d.transform(features)
    df = pd.DataFrame(coords_3d, columns=["PC1", "PC2", "PC3"])
    df["Instrument"] = labels

    fig3d = px.scatter_3d(df, x="PC1", y="PC2", z="PC3", color="Instrument", color_discrete_map=INSTRUMENT_COLOR_MAP)
    fig3d.write_html(os.path.join(out_dir, f'pca3d_{method_name}.html'), include_plotlyjs="cdn")

    return silhouette_score(features, KMeans(4).fit_predict(features)), 0  # placeholder


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = EncodecModel.encodec_model_24khz().eval().to(DEVICE)
    metadata = load_balanced_metadata(CSV_PATH, AUDIO_DIR, MAX_AUDIOS_PER_INSTRUMENT)
    data = extract_latents(metadata, AUDIO_DIR, model)
    if not data:
        return

    labels = [d['instrument_short'] for d in data]
    reps = create_feature_representations(data)

    best_result = None
    all_results = []

    for r_name, features in reps.items():
        preproc = test_preprocessing(features, labels)
        for p_name, p_result in preproc.items():
            dimred = reduce_dimensions(p_result["features"], labels)
            for d_name, d_result in dimred.items():
                score = d_result["silhouette"]
                result = {
                    "name": f"{r_name} + {p_name} + {d_name}",
                    "features": d_result["features"],
                    "silhouette": score
                }
                all_results.append(result)
                if not best_result or score > best_result["silhouette"]:
                    best_result = result

    if best_result:
        sil, sep = visualize(best_result["features"], labels, best_result["name"], OUTPUT_DIR)
        with open(os.path.join(OUTPUT_DIR, "results.pkl"), "wb") as f:
            pickle.dump(all_results, f)
        print(f"Best config: {best_result['name']} | Silhouette: {sil:.3f}")

if __name__ == "__main__":
    main()
