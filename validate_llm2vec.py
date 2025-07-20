#!/usr/bin/env python3
import os
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import numpy as np
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix
from math import comb, log
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

if __name__ == '__main__':
    # === Load model + tokenizer ===
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

    print("\nHardware status:")
    print_hw_status()

    # === Wrap model ===
    wrapped = EmbeddingModel(model, tokenizer, device)

    # === Results container ===
    results = []

    # === Process tasks with streaming + incremental stats ===
    for task in task_names:
        print(f"\n=== Task: {task} ===")
        # Streaming load for fitting KMeans
        ds1 = load_dataset(
            "mteb", name=task, split="test", streaming=True, cache_dir="./hf_cache"
        )
        # Determine number of clusters
        K = ds1.features['label'].num_classes
        print(f"Detected {K} clusters.")

        # Incremental clustering
        kmeans = MiniBatchKMeans(n_clusters=K, batch_size=512, random_state=0)
        buffer = []
        for ex in ds1:
            buffer.append(ex['text'])
            if len(buffer) == 512:
                embs = wrapped.encode(buffer)
                kmeans.partial_fit(embs)
                buffer.clear()
        if buffer:
            embs = wrapped.encode(buffer)
            kmeans.partial_fit(embs)
            buffer.clear()

        # Streaming second pass to build contingency matrix
        ds2 = load_dataset(
            "mteb", name=task, split="test", streaming=True, cache_dir="./hf_cache"
        )
        cont = np.zeros((K, K), dtype=np.int64)
        buf_texts, buf_labels = [], []
        for ex in ds2:
            buf_texts.append(ex['text'])
            buf_labels.append(ex['label'])
            if len(buf_texts) == 512:
                embs = wrapped.encode(buf_texts)
                preds = kmeans.predict(embs)
                for true, pred in zip(buf_labels, preds):
                    cont[true, pred] += 1
                buf_texts.clear(); buf_labels.clear()
        if buf_texts:
            embs = wrapped.encode(buf_texts)
            preds = kmeans.predict(embs)
            for true, pred in zip(buf_labels, preds):
                cont[true, pred] += 1

        # Compute ARI/NMI from contingency matrix
        n = cont.sum()
        sum_cij = sum(comb(cont[i, j], 2) for i in range(K) for j in range(K))
        sum_ci = sum(comb(cont[i, :].sum(), 2) for i in range(K))
        sum_cj = sum(comb(cont[:, j].sum(), 2) for j in range(K))
        expected = (sum_ci * sum_cj) / comb(n, 2)
        max_index = 0.5 * (sum_ci + sum_cj)
        ari = (sum_cij - expected) / (max_index - expected)

        # NMI
        h_true = -sum((cont[i, :].sum() / n) * log(cont[i, :].sum() / n) for i in range(K))
        h_pred = -sum((cont[:, j].sum() / n) * log(cont[:, j].sum() / n) for j in range(K))
        mutual_info = sum(
            (cont[i, j] / n) * log((cont[i, j] * n) / (cont[i, :].sum() * cont[:, j].sum()))
            for i in range(K) for j in range(K) if cont[i, j] > 0
        )
        nmi = mutual_info / ((h_true + h_pred) / 2)

        print(f"{task} -> ARI: {ari:.4f}, NMI: {nmi:.4f}")
        results.append({'task': task, 'ARI': ari, 'NMI': nmi})

    # === Visualization: Bar chart across tasks ===
    df = pd.DataFrame(results)
    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, df['ARI'], width, label='ARI')
    ax.bar(x + width/2, df['NMI'], width, label='NMI')
    ax.set_xticks(x)
    ax.set_xticklabels(df['task'], rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Full-data Clustering Metrics')
    ax.legend()
    plt.tight_layout()
    plt.show()
