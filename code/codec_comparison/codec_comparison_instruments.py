# ==============================================================
# Multi-codec clustering benchmark: EnCodec, DAC, SpeechTokenizer
# --------------------------------------------------------------
# - Loads your instruments dataset and runs all three models.
# - Builds model-appropriate features:
#     * DAC: continuous latents (z)  -> mean/weighted/stat/temporal-PCA
#     * EnCodec: discrete codes      -> per-layer/global code hist features
#     * SpeechTokenizer: discrete    -> semantic/acoustic hist + code stats/PCA
# - Runs preprocessing, DR, KMeans, metrics (silhouette, ARI).
# - Saves per-model visualizations and a cross-model comparison pack.
#
# Requires (already in your env):
#   torch, torchaudio, numpy<2, pandas, scikit-learn, matplotlib, seaborn, plotly
#   encodec, descript-audio-codec (dac), speechtokenizer, huggingface_hub
#
# Outputs:
#   <OUT_DIR>/
#     ├─ encodec/ (plots, confusion heatmap, pkl)
#     ├─ dac/     (plots, confusion heatmap, pkl)
#     ├─ speechtokenizer/ (plots, confusion heatmap, pkl)
#     ├─ comparison_summary.csv
#     ├─ comparison_bars.png
#     ├─ comparison_bars.html
#     ├─ report.md
#     └─ report.html
# ==============================================================

import os
import json
import warnings
from pathlib import Path
import random
import pickle
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import torch
import torchaudio
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

warnings.filterwarnings("ignore")

# ---------- Paths & Config ----------
CSV_PATH = "aaron_xai4ae/approach_3/musical_instruments/dataset/4_instruments_data/Metadata_Train.csv"
AUDIO_DIR = "aaron_xai4ae/approach_3/musical_instruments/dataset/4_instruments_data/Train_submission/Train_submission"
OUT_DIR  = "aaron_xai4ae/approach_3/other_models/codec_comparison/musical_instruments/results"

# Avoid home-cache quota by default (respects env if already set)
DEFAULT_HF_HOME = "/cs/labs/adiyoss/aarontaub/hf_cache"
for k, v in {
    "HF_HOME": DEFAULT_HF_HOME,
    "HF_HUB_CACHE": os.path.join(DEFAULT_HF_HOME, "hub"),
    "TRANSFORMERS_CACHE": os.path.join(DEFAULT_HF_HOME, "transformers")
}.items():
    os.environ.setdefault(k, v)
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
AUDIO_DURATION = 3.0  # seconds
MAX_AUDIOS_PER_INSTRUMENT = 60

INSTRUMENT_CLASSES = ['Sound_Guitar', 'Sound_Drum', 'Sound_Piano', 'Sound_Violin']
COLOR_MAP = {'Guitar': '#FF0000', 'Drum': '#00B050', 'Piano': '#1F4E79', 'Violin': '#FF8000'}

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "font.size": 11,
})
sns.set_theme(style="whitegrid")

# ---------- Repro ----------
def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------- Data ----------
def load_balanced_metadata(csv_path, audio_dir, num_clips_per_class):
    df = pd.read_csv(csv_path)
    rows = []
    for instr in INSTRUMENT_CLASSES:
        files = df[df["Class"] == instr]["FileName"].tolist()
        files = [f for f in files if (Path(audio_dir) / f).exists()]
        random.shuffle(files)
        for f in files[:num_clips_per_class]:
            rows.append({
                "audio_path": str(Path(audio_dir) / f),
                "instrument_full": instr,
                "instrument": instr.replace("Sound_", "")
            })
    return rows

def load_wave(path):
    wav, sr = torchaudio.load(path)
    # mono + peak norm
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    peak = wav.abs().max()
    if float(peak) > 0:
        wav = wav / peak
    return wav, sr

def center_crop_or_pad(wav, target_len):
    T = wav.shape[-1]
    if T == target_len:
        return wav
    if T < target_len:
        return torch.nn.functional.pad(wav, (0, target_len - T))
    mid = (T - target_len) // 2
    return wav[:, mid:mid + target_len]

# ---------- Feature builders ----------
def build_continuous_reps(latents_list):
    """
    latents_list: list of np.ndarray with shape (C, T')
    Returns dict[str] -> (N, D)
    """
    simple, weighted, statistical, temporal = [], [], [], []
    max_len = 0
    for L in latents_list:
        # simple mean
        simple.append(L.mean(axis=-1))
        # center-weighted mean
        nT = L.shape[1]
        w = np.exp(-0.5 * ((np.arange(nT) - nT//2) / (nT//4 + 1e-8))**2)
        w = w / w.sum()
        weighted.append(np.average(L, axis=1, weights=w))
        # stats
        stats = np.concatenate([
            L.mean(axis=-1),
            L.std(axis=-1),
            np.percentile(L, 25, axis=-1),
            np.percentile(L, 75, axis=-1)
        ])
        statistical.append(stats)
        # temporal flatten
        flat = L.flatten()
        max_len = max(max_len, flat.shape[0])
        temporal.append(flat)

    # cap temporal length
    cap = min(max_len, len(latents_list)//2 if len(latents_list) > 1 else max_len, 256)
    temporal_fixed = []
    for x in temporal:
        if len(x) >= cap:
            temporal_fixed.append(x[:cap])
        else:
            temporal_fixed.append(np.pad(x, (0, cap - len(x))))
    temporal_arr = np.stack(temporal_fixed)

    reps = {
        "simple_mean": np.stack(simple),
        "weighted_mean": np.stack(weighted),
        "statistical": np.stack(statistical),
    }
    # temporal PCA to <=64
    if temporal_arr.shape[1] > 8:
        n_comp = int(min(64, temporal_arr.shape[0]-1, temporal_arr.shape[1]))
        n_comp = max(n_comp, 2)
        reps["temporal_pca"] = PCA(n_components=n_comp, random_state=SEED).fit_transform(temporal_arr)
    else:
        reps["temporal_raw"] = temporal_arr
    return reps

def entropy(p, eps=1e-12):
    return float(-np.sum(p * np.log(p + eps)))

def build_discrete_reps(code_arrays, n_q, codebook_size, merge_acoustic=True):
    """
    code_arrays: list of np.array shape (n_q, T')
    Returns dict[str] -> (N, D)
    """
    semantic_h, acoustic_h, all_layer_h, stats_list = [], [], [], []
    for codes in code_arrays:
        Tprime = codes.shape[1]
        # layer 0: semantic hist
        sem = codes[0]
        sem_hist = np.bincount(sem, minlength=codebook_size).astype(np.float32)
        sem_hist /= max(1, Tprime)
        semantic_h.append(sem_hist)

        # layers 1..n_q-1 merged acoustic
        if n_q > 1 and merge_acoustic:
            ac = codes[1:].reshape(-1)
            ac_hist = np.bincount(ac, minlength=codebook_size).astype(np.float32)
            ac_hist /= max(1, ac.shape[0])
        else:
            ac_hist = np.zeros(codebook_size, dtype=np.float32)
        acoustic_h.append(ac_hist)

        # per-layer hist concat + stats
        per_layer = []
        layer_stats = []
        for q in range(n_q):
            cq = codes[q]
            hq = np.bincount(cq, minlength=codebook_size).astype(np.float32)
            hq /= max(1, Tprime)
            per_layer.append(hq)
            s = np.sort(hq)[::-1]
            top1 = s[0] if s.size else 0.0
            top3 = s[:3].sum() if s.size >= 3 else s.sum()
            uniq = float((hq > 0).sum()) / codebook_size
            ent = entropy(hq)
            layer_stats += [ent, top1, top3, uniq]

        all_layer_h.append(np.concatenate(per_layer))
        stats_list.append(np.array(layer_stats, dtype=np.float32))

    reps = {
        "semantic_hist": np.stack(semantic_h),
        "acoustic_hist": np.stack(acoustic_h),
        "code_stats": np.stack(stats_list)
    }
    concat_hist = np.stack(all_layer_h)
    if concat_hist.shape[1] > 8:
        n_comp = int(min(128, concat_hist.shape[0]-1, concat_hist.shape[1]))
        n_comp = max(n_comp, 2)
        reps["code_hist_pca"] = PCA(n_components=n_comp, random_state=SEED).fit_transform(concat_hist)
    else:
        reps["code_hist_pca"] = concat_hist
    return reps

# ---------- Clustering pipeline ----------
SCALERS = {
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler(),
    "PowerTransformer": PowerTransformer(method="yeo-johnson"),
}

def reduce_dimensions(X, y_idx):
    out = {}
    n, d = X.shape
    # LDA
    try:
        lda = LinearDiscriminantAnalysis(n_components=min(3, len(set(y_idx))-1))
        X_lda = lda.fit_transform(X, y_idx)
        out["LDA"] = X_lda
    except Exception:
        pass
    # FS + PCA
    try:
        k = min(d//2, n//2, 50)
        if 10 < k < d:
            X_sel = SelectKBest(f_classif, k=k).fit(X, y_idx).transform(X)
            pc = min(10, X_sel.shape[1], X_sel.shape[0]-1)
            out["FeatureSelect_PCA"] = PCA(n_components=pc, random_state=SEED).fit(X_sel).transform(X_sel)
    except Exception:
        pass
    # ICA
    try:
        k = min(20, d, n-1)
        if k >= 4:
            out["ICA"] = FastICA(n_components=k, random_state=SEED, max_iter=1000).fit_transform(X)
    except Exception:
        pass
    # PCA
    try:
        pc = min(10, d, n-1)
        out["PCA"] = PCA(n_components=pc, random_state=SEED).fit(X).transform(X)
    except Exception:
        pass
    return out

def run_kmeans_eval(X, y_idx, n_cls=4):
    km = KMeans(n_clusters=n_cls, n_init=20, random_state=SEED).fit(X)
    sil = silhouette_score(X, km.labels_) if len(set(km.labels_)) > 1 else -1
    ari = adjusted_rand_score(y_idx, km.labels_)
    return sil, ari, km.labels_

def plot_pca_2d(X, labels, title, out_png):
    pc2 = PCA(n_components=2, random_state=SEED).fit_transform(X)
    df = pd.DataFrame(pc2, columns=["PC1", "PC2"])
    df["Instrument"] = labels
    plt.figure(figsize=(7.5, 5.8))
    for k, c in COLOR_MAP.items():
        mask = (df["Instrument"] == k)
        plt.scatter(df.loc[mask, "PC1"], df.loc[mask, "PC2"], s=24, alpha=0.8, label=k, color=c, edgecolor="white", linewidth=0.5)
    plt.title(title)
    plt.legend(frameon=True)
    plt.tight_layout()
    Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

def plot_pca_3d_html(X, labels, title, out_html):
    pc3 = PCA(n_components=3, random_state=SEED).fit_transform(X)
    df = pd.DataFrame(pc3, columns=["PC1", "PC2", "PC3"])
    df["Instrument"] = labels
    fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3", color="Instrument", color_discrete_map=COLOR_MAP, title=title)
    fig.write_html(out_html, include_plotlyjs="cdn")

def plot_confusion(km_labels, true_labels, title, out_png):
    # Map clusters to the majority true label (for readability only)
    clusters = sorted(set(km_labels))
    mapping = {}
    for c in clusters:
        idx = np.where(km_labels == c)[0]
        maj = Counter([true_labels[i] for i in idx]).most_common(1)[0][0]
        mapping[c] = maj
    mapped = [mapping[c] for c in km_labels]
    classes = sorted(set(true_labels), key=lambda s: s)
    cm = confusion_matrix(true_labels, mapped, labels=classes)
    plt.figure(figsize=(6.2, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted (cluster→majority label)")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

# ---------- Model wrappers ----------
def run_dac(rows, outdir):
    import dac
    # 24k model
    weights = dac.utils.download(model_type="24khz")
    model = dac.DAC.load(weights).to(DEVICE).eval()
    target_sr = 24000
    T = int(AUDIO_DURATION * target_sr)

    latents = []
    labels = []
    for r in rows:
        wav, sr = load_wave(r["audio_path"])
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        wav = center_crop_or_pad(wav, T).to(DEVICE)
        with torch.no_grad():
            x = model.preprocess(wav.unsqueeze(0), target_sr)  # (B,C,T)
            z, codes, _lat, _, _ = model.encode(x)  # z: (B,C,T')
        L = z.squeeze(0).detach().cpu().numpy()
        latents.append(L)
        labels.append(r["instrument"])

    reps = build_continuous_reps(latents)
    return {"features": reps, "labels": labels, "sr": target_sr, "kind": "continuous"}

def run_encodec(rows, outdir):
    # Use encodec pip package (no audiocraft dependency)
    from encodec import EncodecModel
    from encodec.utils import convert_audio

    model = EncodecModel.encodec_model_24khz()  # downloads weights if missing
    model.set_target_bandwidth(6.0)  # typical; not critical for codes shape
    model = model.to(DEVICE)
    target_sr = 24000
    T = int(AUDIO_DURATION * target_sr)

    code_arrays = []
    labels = []
    for r in rows:
        wav, sr = load_wave(r["audio_path"])
        wav = convert_audio(wav, sr, target_sr, model.channels).to(DEVICE)
        wav = center_crop_or_pad(wav, T)
        with torch.no_grad():
            # encode returns a list of frames; each element: (encoded, scale)
            enc = model.encode(wav.unsqueeze(0))  # B=1
            # Collect codes across frames -> shape (B, n_q, T')
            codes = torch.cat([f[0] for f in enc], dim=-1)  # list[(B,n_q,S)] -> (B,n_q,T')
            codes = codes[0].detach().cpu().numpy().astype(np.int32)  # (n_q, T')
        code_arrays.append(codes)
        labels.append(r["instrument"])

    # EnCodec: treat as discrete RVQ codes like ST
    n_q = code_arrays[0].shape[0]
    # EnCodec codebook size is typically 1024 per codebook
    codebook_size = 1024
    reps = build_discrete_reps(code_arrays, n_q=n_q, codebook_size=codebook_size)
    return {"features": reps, "labels": labels, "sr": target_sr, "kind": "discrete"}

def run_speechtokenizer(rows, outdir):
    # Use snapshot_download to avoid ~/.cache symlinks
    from huggingface_hub import snapshot_download
    from speechtokenizer import SpeechTokenizer

    base = os.environ.get("HF_HOME", DEFAULT_HF_HOME)
    local_dir = os.path.join(base, "models", "SpeechTokenizer")
    repo_id = "fnlp/SpeechTokenizer"
    sub = "speechtokenizer_hubert_avg"

    download_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{sub}/config.json", f"{sub}/SpeechTokenizer.pt"],
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    cfg_path = os.path.join(download_dir, sub, "config.json")
    ckpt_path = os.path.join(download_dir, sub, "SpeechTokenizer.pt")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    model = SpeechTokenizer.load_from_checkpoint(cfg_path, ckpt_path).eval().to(DEVICE)
    target_sr = int(getattr(model, "sample_rate", cfg.get("sample_rate", 16000)))
    T = int(AUDIO_DURATION * target_sr)
    n_q = int(cfg.get("n_q", 8))
    codebook_size = int(cfg.get("codebook_size", 1024))

    code_arrays = []
    labels = []
    for r in rows:
        wav, sr = load_wave(r["audio_path"])
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        wav = center_crop_or_pad(wav, T).to(DEVICE)
        with torch.no_grad():
            codes = model.encode(wav.unsqueeze(0))  # (n_q, B, T')
            codes = codes[:, 0, :].detach().cpu().numpy().astype(np.int32)
        code_arrays.append(codes)
        labels.append(r["instrument"])

    reps = build_discrete_reps(code_arrays, n_q=n_q, codebook_size=codebook_size)
    return {"features": reps, "labels": labels, "sr": target_sr, "kind": "discrete"}

# ---------- Orchestration ----------
def evaluate_model(name, bundle, outdir):
    """
    bundle: dict with keys {"features": dict[str->np.ndarray], "labels": list[str], "kind": "continuous|discrete"}
    Returns best row dict
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    labels = bundle["labels"]
    y_idx = [sorted(set(labels)).index(x) for x in labels]
    summary_rows = []
    best = None

    for rep_name, X0 in bundle["features"].items():
        for scaler_name, scaler in SCALERS.items():
            try:
                X = scaler.fit_transform(X0)
            except Exception:
                continue
            # Dimensionality reduction family
            variants = reduce_dimensions(X, y_idx)
            # Also include "None" variant = raw scaled features
            variants["None"] = X

            for dim_name, Xd in variants.items():
                sil, ari, y_pred = run_kmeans_eval(Xd, y_idx, n_cls=4)
                row = {
                    "model": name,
                    "rep": rep_name,
                    "scaler": scaler_name,
                    "dimred": dim_name,
                    "silhouette": sil,
                    "ari": ari,
                }
                summary_rows.append(row)
                # Track best by silhouette; tie-break by ARI
                if (best is None) or (sil > best["silhouette"] + 1e-6) or (abs(sil - best["silhouette"]) < 1e-6 and ari > best["ari"]):
                    best = row
                    # Save visuals for the best so far
                    plot_pca_2d(Xd, labels, f"{name} · {rep_name} + {scaler_name} + {dim_name}", os.path.join(outdir, "best_pca2d.png"))
                    plot_pca_3d_html(Xd, labels, f"{name} · {rep_name} + {scaler_name} + {dim_name}", os.path.join(outdir, "best_pca3d.html"))
                    plot_confusion(np.array(y_pred), np.array(labels), f"{name} Confusion (best config)", os.path.join(outdir, "confusion.png"))

    # Save raw table for this model
    dfm = pd.DataFrame(summary_rows).sort_values(["silhouette", "ari"], ascending=False)
    dfm.to_csv(os.path.join(outdir, f"results_{name}.csv"), index=False)
    with open(os.path.join(outdir, f"results_{name}.pkl"), "wb") as f:
        pickle.dump(summary_rows, f)

    # Write a tiny model report
    if best:
        with open(os.path.join(outdir, "README.txt"), "w") as f:
            f.write(f"Best config for {name}:\n")
            f.write(json.dumps(best, indent=2))

    return best, pd.DataFrame(summary_rows)

def render_comparison(df_all, outdir):
    # Aggregate: best per model
    best_per_model = df_all.sort_values(["silhouette","ari"], ascending=False).groupby("model", as_index=False).first()

    # Save summary CSV
    best_per_model.to_csv(os.path.join(outdir, "comparison_summary.csv"), index=False)

    # Bar plots (PNG + HTML)
    plt.figure(figsize=(7.5,4.6))
    ax = sns.barplot(data=best_per_model, x="model", y="silhouette", palette="deep")
    ax.set_ylabel("Silhouette (higher is better)")
    ax.set_xlabel("Model")
    ax.set_title("Codec comparison: best silhouette per model")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x()+p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, xytext=(0,3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_bars.png"))
    plt.close()

    fig = px.bar(best_per_model, x="model", y=["silhouette","ari"], barmode="group",
                 title="Codec comparison (best per model): Silhouette & ARI")
    fig.write_html(os.path.join(outdir, "comparison_bars.html"), include_plotlyjs="cdn")

    # Markdown report
    md = ["# Codec Clustering Comparison",
          "",
          "## Best per model",
          best_per_model.to_markdown(index=False),
          "",
          "## Notes",
          "- Silhouette is geometry-only; ARI checks agreement with true labels (label-invariant).",
          "- Confusion heatmaps are in each model's folder (clusters mapped to majority label for readability)."]
    with open(os.path.join(outdir, "report.md"), "w") as f:
        f.write("\n".join(md))

    # Simple HTML report
    html = f"""
    <html><head><meta charset="utf-8"><title>Codec Comparison</title>
    <style>
      body{{font-family:system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin:24px;}}
      table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px 10px}}
    </style></head><body>
      <h1>Codec Clustering Comparison</h1>
      <h2>Best per model</h2>
      {best_per_model.to_html(index=False)}
      <h2>Charts</h2>
      <img src="comparison_bars.png" style="max-width:720px;border:1px solid #eee"/>
      <p>Interactive bars: <a href="comparison_bars.html">comparison_bars.html</a></p>
      <h2>Per-model artifacts</h2>
      <ul>
        <li><b>encodec/</b>: best_pca2d.png, best_pca3d.html, confusion.png</li>
        <li><b>dac/</b>: best_pca2d.png, best_pca3d.html, confusion.png</li>
        <li><b>speechtokenizer/</b>: best_pca2d.png, best_pca3d.html, confusion.png</li>
      </ul>
      <hr/>
      <p style="color:#888">Silhouette measures cluster separability; ARI compares clusters vs. ground truth.</p>
    </body></html>
    """
    with open(os.path.join(outdir, "report.html"), "w") as f:
        f.write(html)

def main():
    set_seeds(SEED)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load data
    rows = load_balanced_metadata(CSV_PATH, AUDIO_DIR, MAX_AUDIOS_PER_INSTRUMENT)
    if not rows:
        print("No audio found. Check CSV_PATH/AUDIO_DIR.")
        return

    # Run ALL models, each in its own try/except (so one failure doesn't stop the rest)
    all_best = []
    all_tables = []

    # DAC
    try:
        print("[RUN] DAC …")
        best, dfm = evaluate_model("dac", run_dac(rows, os.path.join(OUT_DIR, "dac")), os.path.join(OUT_DIR, "dac"))
        if best: all_best.append(best)
        all_tables.append(dfm)
    except Exception as e:
        print(f"[WARN] DAC failed: {type(e).__name__}: {e}")

    # EnCodec
    try:
        print("[RUN] EnCodec …")
        best, dfm = evaluate_model("encodec", run_encodec(rows, os.path.join(OUT_DIR, "encodec")), os.path.join(OUT_DIR, "encodec"))
        if best: all_best.append(best)
        all_tables.append(dfm)
    except Exception as e:
        print(f"[WARN] EnCodec failed: {type(e).__name__}: {e}")

    # SpeechTokenizer
    try:
        print("[RUN] SpeechTokenizer …")
        best, dfm = evaluate_model("speechtokenizer", run_speechtokenizer(rows, os.path.join(OUT_DIR, "speechtokenizer")), os.path.join(OUT_DIR, "speechtokenizer"))
        if best: all_best.append(best)
        all_tables.append(dfm)
    except Exception as e:
        print(f"[WARN] SpeechTokenizer failed: {type(e).__name__}: {e}")

    # Comparison pack
    if all_tables:
        df_all = pd.concat(all_tables, ignore_index=True)
        df_all.to_csv(os.path.join(OUT_DIR, "all_results_raw.csv"), index=False)
        render_comparison(df_all, OUT_DIR)
        print("[DONE] Wrote comparison pack to", OUT_DIR)
    else:
        print("No model produced results.")

if __name__ == "__main__":
    main()
