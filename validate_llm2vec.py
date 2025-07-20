#!/usr/bin/env python3
import os
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import mteb
import numpy as np
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix
from sklearn.cluster import KMeans
from embedding_model import EmbeddingModel, print_hw_status


# === Paths and Tasks ===
base_model_path   = "./models/local-base"
lora_adapter_path = "./models/local-llm2vec-simcse"
task_names = [
    "ArXivHierarchicalClusteringP2P",
    "ArXivHierarchicalClusteringS2S",
    "MedrxivClusteringP2P.v2",
    "MedrxivClusteringS2S.v2",
    "BiorxivClusteringS2S.v2",
    "BiorxivClusteringP2P.v2"
]

def main():
    # === Load base + LoRA ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading base model from {base_model_path} on {device}...")
    base = AutoModel.from_pretrained(
        base_model_path, local_files_only=True, device_map="auto", torch_dtype=torch.float16
    )
    print(f"Loading LoRA adapter from {lora_adapter_path}...")
    model = PeftModel.from_pretrained(
        base, lora_adapter_path, local_files_only=True
    ).to(device)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    print("\n=== Hardware status ===")
    print_hw_status()
    print("=======================\n")

    # === MTEB setup ===
    task_names = [
        "ArxivClustering-p2p",
        "ArXivHierarchicalClusteringS2S",
        "MedrxivClusteringP2P.v2",
        "MedrxivClusteringS2S.v2",
        "BiorxivClusteringS2S.v2",
        "BiorxivClusteringP2P.v2"
    ]
    tasks = mteb.get_tasks(tasks=task_names)
    evaluator = mteb.MTEB(tasks=tasks)

    # === Run evaluation ===
    wrapped = EmbeddingModel(model, tokenizer, device)
    evaluator.run(wrapped, output_folder=output_folder)

    # === Visualization: ARI & NMI Bar Chart ===
    records = []
    for t in task_names:
        path = os.path.join(output_folder, t, "metrics.csv")
        df = pd.read_csv(path)
        records.append({"task": t, "ARI": df.loc[0, "ARI"], "NMI": df.loc[0, "NMI"]})
    metrics_df = pd.DataFrame(records)

    x = np.arange(len(metrics_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, metrics_df["ARI"], width, label="ARI")
    ax.bar(x + width/2, metrics_df["NMI"], width, label="NMI")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["task"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Clustering Metrics Across Tasks")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # === Visualization: t-SNE Scatter (First Task) ===
    first_task = tasks[0]
    dataset = first_task.dataset
    texts = dataset["text"][:1000]
    true_labels = dataset["label"][:1000]
    embs = wrapped.encode(texts)
    k = len(set(true_labels))
    preds = KMeans(n_clusters=k, random_state=0).fit_predict(embs)
    proj = TSNE(n_components=2).fit_transform(embs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(proj[:, 0], proj[:, 1], c=true_labels, s=5)
    axes[0].set_title("True Clusters")
    axes[1].scatter(proj[:, 0], proj[:, 1], c=preds, s=5)
    axes[1].set_title("Predicted Clusters")
    plt.tight_layout()
    plt.show()

    # === Visualization: Silhouette Plot ===
    sil_vals = silhouette_samples(embs, preds)
    avg_sil = silhouette_score(embs, preds)
    print(f"Average silhouette score: {avg_sil:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    y_lower = 10
    for i in range(k):
        ith_sil = sil_vals[preds == i]
        ith_sil.sort()
        size_i = len(ith_sil)
        y_upper = y_lower + size_i
        color_i = cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil, facecolor=color_i, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.axvline(x=avg_sil, color="red", linestyle="--")
    ax.set_title("Silhouette Plot")
    plt.tight_layout()
    plt.show()

    # === Visualization: Confusion Matrix ===
    cmat = confusion_matrix(true_labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cmat, cmap="Blues")
    fig.colorbar(cax)
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            ax.text(j, i, cmat[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()