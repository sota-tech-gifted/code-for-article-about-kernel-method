"""
Utility function for efficiently extracting feature vectors from a full dataset.
"""
import torch
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    for imgs, lbls in tqdm(loader, desc='Extract features'):
        feat, _ = model(imgs.to(device))
        feats.append(feat.cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(feats), np.concatenate(labels)
