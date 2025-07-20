import numpy as np
import psutil
import torch


# === Helpers ===
def print_hw_status():
    vm = psutil.virtual_memory()
    print(f"[RAM] total: {vm.total/1024**3:.1f} GB, used: {vm.used/1024**3:.1f} GB")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"[GPU] {props.name}, VRAM: {props.total_memory/1024**3:.1f} GB")

class EmbeddingModel:
    def __init__(self, model, tokenizer, device, batch_size=16, max_length=512):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    def encode(self, texts):
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                tok = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)
                out = self.model(**tok).last_hidden_state
                mask = tok["attention_mask"].unsqueeze(-1)
                summed = (out * mask).sum(dim=1)
                counts = mask.sum(dim=1)
                pooled = summed / counts
                embeddings.append(pooled.cpu().numpy())
        return np.vstack(embeddings)