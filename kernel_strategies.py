"""
Kernel transformation strategies (NumPy and PyTorch implementations)
========================================
Contains 12 operators for generating simulated target-domain kernels from feature kernels.
All outputs are symmetric positive semidefinite matrices with numerical stabilization.
"""
import numpy as np
import torch
import torch.nn.functional as F

# ========== Utility functions ==========
def ensure_psd_np(K):
    """NumPy version: clamp eigenvalues at zero to ensure positive semidefiniteness."""
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def stabilize_pt(K, eps=1e-6):
    """PyTorch version: add diagonal loading to improve eigendecomposition stability."""
    return K + eps * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)

def ensure_psd_pt(K):
    """PyTorch version: stabilize the matrix and clamp eigenvalues at zero."""
    K = stabilize_pt(K)
    eigvals, eigvecs = torch.linalg.eigh(K)
    eigvals = F.relu(eigvals)
    return eigvecs @ torch.diag(eigvals) @ eigvecs.T

# ============= NumPy strategies (for offline analysis) =============
def spectrum_perturbation_np(K_bar, noise_std=0.1):
    """A: Inject correlated Gaussian noise into the eigenvalues."""
    eigvals, eigvecs = np.linalg.eigh(K_bar)
    eigvals = np.maximum(eigvals, 0)
    eps = np.random.randn(len(eigvals)) * noise_std * np.sqrt(eigvals + 1e-8)
    new_eigvals = np.maximum(eigvals + eps, 0)
    return eigvecs @ np.diag(new_eigvals) @ eigvecs.T

def manifold_rotation_np(K_bar):
    """B: Apply a random orthogonal rotation to the eigenvectors."""
    eigvals, eigvecs = np.linalg.eigh(K_bar)
    eigvals = np.maximum(eigvals, 0)
    A = np.random.randn(eigvecs.shape[0], eigvecs.shape[0])
    Q, _ = np.linalg.qr(A)
    rotated = eigvecs @ Q
    return rotated @ np.diag(eigvals) @ rotated.T

def kernel_mixup_np(K_bar, noise_std=0.1, beta_alpha=0.4, beta_beta=0.4):
    """C: Interpolate with a random noise kernel using a Beta-distributed weight."""
    n = K_bar.shape[0]
    G = np.random.randn(n, n) * noise_std
    K_noise = G @ G.T
    beta = np.random.beta(beta_alpha, beta_beta)
    K_i = beta * K_bar + (1 - beta) * K_noise
    return ensure_psd_np(K_i)

def random_masking_np(K_bar, mask_ratio=0.2):
    """D: Randomly mask 20% of entries and reconstruct a positive semidefinite matrix."""
    mask = np.random.binomial(1, 1 - mask_ratio, size=K_bar.shape)
    mask = np.tril(mask) + np.tril(mask, -1).T
    return ensure_psd_np(K_bar * mask)

def frequency_filter_np(K_bar, keep_ratio=0.8):
    """E: Retain only the largest 80% of eigenvalues."""
    eigvals, eigvecs = np.linalg.eigh(K_bar)
    eigvals = np.maximum(eigvals, 0)
    cutoff = int(len(eigvals) * keep_ratio)
    eigvals[cutoff:] = 0.0
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def additive_noise_np(K_bar, noise_level=0.05):
    """F: Add symmetric Gaussian noise and project to the positive semidefinite cone."""
    noise = np.random.randn(*K_bar.shape) * noise_level
    K_i = K_bar + noise + noise.T
    return ensure_psd_np(K_i)

def block_masking_np(K_bar, block_size_ratio=0.15, mask_prob=0.3):
    """G: Randomly zero contiguous blocks."""
    N = K_bar.shape[0]
    mask = np.ones((N, N), dtype=np.float32)
    block_size = max(1, int(N * block_size_ratio))
    for i in range(0, N, block_size):
        for j in range(0, N, block_size):
            if np.random.rand() < mask_prob:
                mask[i:i+block_size, j:j+block_size] = 0.0
    mask = np.minimum(mask, mask.T)
    return ensure_psd_np(K_bar * mask)

def spectral_phase_perturbation_np(K_bar, noise_std=0.08):
    """H: Perturb the eigenvectors and orthogonalize them again."""
    eigvals, eigvecs = np.linalg.eigh(K_bar)
    eigvals = np.maximum(eigvals, 0)
    perturb = eigvecs + np.random.randn(*eigvecs.shape).astype(np.float32) * noise_std
    Q, _ = np.linalg.qr(perturb)
    return (Q @ np.diag(eigvals) @ Q.T).astype(np.float64)

def contrast_scaling_np(K_bar, gamma_range=(0.5, 1.5), scale_range=(0.8, 1.2)):
    """I: Scale eigenvalues by a random power."""
    eigvals, eigvecs = np.linalg.eigh(K_bar)
    eigvals = np.maximum(eigvals, 0)
    gamma = np.random.uniform(*gamma_range)
    scale = np.random.uniform(*scale_range)
    new_eigvals = (eigvals ** gamma) * scale
    new_eigvals = np.maximum(new_eigvals, 0)
    return eigvecs @ np.diag(new_eigvals) @ eigvecs.T

def pca_subspace_replace_np(K_bar, keep_ratio_range=(0.5, 0.95)):
    """J: Keep the first k principal components and fill the remainder with a random orthogonal complement."""
    eigvals, eigvecs = np.linalg.eigh(K_bar)
    eigvals = np.maximum(eigvals, 0)
    N = len(eigvals)
    keep_ratio = np.random.uniform(*keep_ratio_range)
    k = max(1, int(N * keep_ratio))
    U_k = eigvecs[:, :k]
    Lambda_k = eigvals[:k]
    random_mat = np.random.randn(N, N - k)
    random_mat = random_mat - U_k @ (U_k.T @ random_mat)
    Q, _ = np.linalg.qr(random_mat)
    U_perp = Q
    Lambda_perp = np.random.uniform(0, np.median(Lambda_k) * 0.1, size=N-k)
    full_U = np.hstack([U_k, U_perp])
    full_Lambda = np.concatenate([Lambda_k, Lambda_perp])
    return full_U @ np.diag(full_Lambda) @ full_U.T

def class_conditioned_perturbation_np(K_bar, labels, noise_std=0.1):
    """K: Perturb each within-class submatrix independently."""
    N = K_bar.shape[0]
    K_new = K_bar.copy()
    unique_labels = np.unique(labels)
    for l in unique_labels:
        idx = np.where(labels == l)[0]
        if len(idx) <= 1: continue
        block = K_bar[np.ix_(idx, idx)]
        eigvals, eigvecs = np.linalg.eigh(block)
        eigvals = np.maximum(eigvals, 0)
        eps = np.random.randn(len(eigvals)) * noise_std * np.sqrt(eigvals + 1e-8)
        new_eigvals = np.maximum(eigvals + eps, 0)
        K_new[np.ix_(idx, idx)] = eigvecs @ np.diag(new_eigvals) @ eigvecs.T
    return ensure_psd_np(K_new)

def kernel_dropout_np(K_bar, drop_prob=0.2, scale_range=(0.8, 1.2)):
    """L: Randomly drop and rescale kernel entries."""
    mask = np.random.binomial(1, 1 - drop_prob, size=K_bar.shape).astype(np.float32)
    scale = np.random.uniform(*scale_range, size=K_bar.shape)
    K_drop = K_bar * mask * scale
    K_drop = (K_drop + K_drop.T) / 2
    return ensure_psd_np(K_drop)

STRATEGIES_NP = { 'A': spectrum_perturbation_np, 'B': manifold_rotation_np, 'C': kernel_mixup_np,
                  'D': random_masking_np, 'E': frequency_filter_np, 'F': additive_noise_np,
                  'G': block_masking_np, 'H': spectral_phase_perturbation_np, 'I': contrast_scaling_np,
                  'J': pca_subspace_replace_np, 'K': class_conditioned_perturbation_np, 'L': kernel_dropout_np }

# ============= PyTorch strategies (for online training) =============
def spectrum_perturbation_pt(K_bar, noise_std=0.1):
    K = stabilize_pt(K_bar)
    eigvals, eigvecs = torch.linalg.eigh(K)
    eigvals = F.relu(eigvals)
    perturbation = torch.randn(eigvals.shape[0], device=K.device) * noise_std * torch.sqrt(eigvals + 1e-8)
    new_eigvals = F.relu(eigvals + perturbation)
    return eigvecs @ torch.diag(new_eigvals) @ eigvecs.T

def manifold_rotation_pt(K_bar):
    K = stabilize_pt(K_bar)
    eigvals, eigvecs = torch.linalg.eigh(K)
    eigvals = F.relu(eigvals)
    A = torch.randn(eigvecs.shape[0], eigvecs.shape[0], device=K.device)
    Q, _ = torch.linalg.qr(A)
    rotated = eigvecs @ Q
    return rotated @ torch.diag(eigvals) @ rotated.T

def kernel_mixup_pt(K_bar, noise_std=0.1, beta_alpha=0.4, beta_beta=0.4):
    K = stabilize_pt(K_bar)
    n = K.shape[0]
    random_feats = torch.randn(n, n, device=K.device) * noise_std
    K_noise = random_feats @ random_feats.T
    beta = np.random.beta(beta_alpha, beta_beta)
    K_i = beta * K + (1.0 - beta) * K_noise
    K_i = stabilize_pt(K_i)
    eigvals, eigvecs = torch.linalg.eigh(K_i)
    eigvals = F.relu(eigvals)
    return eigvecs @ torch.diag(eigvals) @ eigvecs.T

def random_masking_pt(K_bar, mask_ratio=0.2):
    K = stabilize_pt(K_bar)
    mask = torch.bernoulli(torch.full_like(K, 1 - mask_ratio))
    K_i = K * mask
    K_i = stabilize_pt(K_i)
    eigvals, eigvecs = torch.linalg.eigh(K_i)
    eigvals = F.relu(eigvals)
    return eigvecs @ torch.diag(eigvals) @ eigvecs.T

def frequency_filter_pt(K_bar, keep_ratio=0.8):
    K = stabilize_pt(K_bar)
    eigvals, eigvecs = torch.linalg.eigh(K)
    eigvals = F.relu(eigvals)
    n = eigvals.shape[0]
    cutoff = int(n * keep_ratio)
    eigvals = eigvals.clone()
    eigvals[cutoff:] = 0.0
    return eigvecs @ torch.diag(eigvals) @ eigvecs.T

def additive_noise_pt(K_bar, noise_level=0.05):
    K = stabilize_pt(K_bar)
    noise = torch.randn_like(K) * noise_level
    K_i = K + noise + noise.T
    K_i = stabilize_pt(K_i)
    eigvals, eigvecs = torch.linalg.eigh(K_i)
    eigvals = F.relu(eigvals)
    return eigvecs @ torch.diag(eigvals) @ eigvecs.T

def block_masking_pt(K_bar, block_size_ratio=0.15, mask_prob=0.3):
    N = K_bar.shape[0]
    mask = torch.ones((N, N), dtype=torch.float32, device=K_bar.device)
    block_size = max(1, int(N * block_size_ratio))
    for i in range(0, N, block_size):
        for j in range(0, N, block_size):
            if torch.rand(1).item() < mask_prob:
                mask[i:i+block_size, j:j+block_size] = 0.0
    mask = torch.minimum(mask, mask.T)
    mask = mask + torch.eye(N, device=K_bar.device) * 1e-6
    K_masked = K_bar * mask
    K_masked = stabilize_pt(K_masked)
    eigvals, eigvecs = torch.linalg.eigh(K_masked)
    eigvals = F.relu(eigvals)
    return eigvecs @ torch.diag(eigvals) @ eigvecs.T

def spectral_phase_perturbation_pt(K_bar, noise_std=0.08):
    K = stabilize_pt(K_bar)
    eigvals, eigvecs = torch.linalg.eigh(K)
    eigvals = F.relu(eigvals)
    perturb = eigvecs + torch.randn_like(eigvecs) * noise_std
    Q, _ = torch.linalg.qr(perturb)
    return (Q @ torch.diag(eigvals) @ Q.T)

def contrast_scaling_pt(K_bar, gamma_range=(0.5, 1.5), scale_range=(0.8, 1.2)):
    K = stabilize_pt(K_bar)
    eigvals, eigvecs = torch.linalg.eigh(K)
    eigvals = F.relu(eigvals) + 1e-8
    gamma = torch.empty(1).uniform_(*gamma_range).item()
    scale = torch.empty(1).uniform_(*scale_range).item()
    new_eigvals = (eigvals ** gamma) * scale
    new_eigvals = F.relu(new_eigvals) + 1e-8
    return eigvecs @ torch.diag(new_eigvals) @ eigvecs.T

def pca_subspace_replace_pt(K_bar, keep_ratio_range=(0.5, 0.95)):
    K = stabilize_pt(K_bar)
    eigvals, eigvecs = torch.linalg.eigh(K)
    eigvals = F.relu(eigvals) + 1e-8
    N = eigvals.shape[0]
    keep_ratio = torch.empty(1).uniform_(*keep_ratio_range).item()
    k = max(1, int(N * keep_ratio))
    U_k = eigvecs[:, :k]
    Lambda_k = eigvals[:k]
    random_mat = torch.randn(N, N - k, device=K_bar.device)
    random_mat = random_mat - U_k @ (U_k.T @ random_mat)
    Q, _ = torch.linalg.qr(random_mat)
    U_perp = Q
    Lambda_perp = torch.empty(N - k, device=K_bar.device).uniform_(0.01, torch.median(Lambda_k).item() * 0.1)
    full_U = torch.cat([U_k, U_perp], dim=1)
    full_Lambda = torch.cat([Lambda_k, Lambda_perp])
    return full_U @ torch.diag(full_Lambda) @ full_U.T

def class_conditioned_perturbation_pt(K_bar, labels, noise_std=0.1):
    N = K_bar.shape[0]
    K_new = K_bar.clone()
    unique_labels = torch.unique(labels)
    for l in unique_labels:
        idx = (labels == l).nonzero(as_tuple=True)[0]
        if len(idx) <= 1: continue
        block = K_bar[idx][:, idx]
        block = stabilize_pt(block)
        eigvals, eigvecs = torch.linalg.eigh(block)
        eigvals = F.relu(eigvals)
        eps = torch.randn(len(eigvals), device=K_bar.device) * noise_std * torch.sqrt(eigvals + 1e-8)
        new_eigvals = F.relu(eigvals + eps)
        K_new[idx.unsqueeze(1), idx.unsqueeze(0)] = eigvecs @ torch.diag(new_eigvals) @ eigvecs.T
    K_new = stabilize_pt(K_new)
    eigvals, eigvecs = torch.linalg.eigh(K_new)
    eigvals = F.relu(eigvals)
    return eigvecs @ torch.diag(eigvals) @ eigvecs.T

def kernel_dropout_pt(K_bar, drop_prob=0.2, scale_range=(0.8, 1.2)):
    K = stabilize_pt(K_bar)
    mask = torch.bernoulli(torch.full_like(K, 1 - drop_prob))
    scale = torch.empty_like(K).uniform_(*scale_range)
    K_drop = K * mask * scale
    K_drop = (K_drop + K_drop.T) / 2
    K_drop = stabilize_pt(K_drop)
    eigvals, eigvecs = torch.linalg.eigh(K_drop)
    eigvals = F.relu(eigvals)
    return eigvecs @ torch.diag(eigvals) @ eigvecs.T

STRATEGIES_PT = { 'A': spectrum_perturbation_pt, 'B': manifold_rotation_pt, 'C': kernel_mixup_pt,
                  'D': random_masking_pt, 'E': frequency_filter_pt, 'F': additive_noise_pt,
                  'G': block_masking_pt, 'H': spectral_phase_perturbation_pt, 'I': contrast_scaling_pt,
                  'J': pca_subspace_replace_pt, 'K': class_conditioned_perturbation_pt, 'L': kernel_dropout_pt }
