"""
Domain generalization training script (stable baseline)
================================
- Augmentation pipelines: original / minimal / TA
- Kernel regularization: adaptive_convex_soft
- Supports single-source and multi-source training
"""
import os, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from scipy.optimize import minimize
from typing import List, Tuple, Optional
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Explicitly import classes used by the original augmentation pipeline.
from augmentations import (
    StrongColorJitter,
    RandomGrayscaleWithEdge,
    RandomEdgeEnhance,
    RandomHighPass,
    ColorQuantization,
    EdgePreservingFilter,
    RandomElasticDeformation,
    MedianCutQuantization,
    WhiteningTransform,
    LocalElasticDeformation,
)
from kernel_strategies import STRATEGIES_PT, stabilize_pt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Global worker seeding ----------
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ---------- Model definition ----------
class ResNet50Linear(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x, return_feat=False):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        if return_feat: return feat, logits
        return logits

# ---------- Utility functions ----------
def symmetrize(K: torch.Tensor) -> torch.Tensor:
    return 0.5 * (K + K.T)

def frobenius_normalize(K: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    K = symmetrize(K)
    return K / (torch.linalg.norm(K, ord='fro') + eps)

def kernel_alignment(K1: torch.Tensor, K2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if K1.shape != K2.shape:
        raise ValueError(f"Shape mismatch: {K1.shape} vs {K2.shape}")
    K1 = symmetrize(K1)
    K2 = symmetrize(K2)
    numerator = torch.sum(K1 * K2)
    denominator = (torch.linalg.norm(K1, ord='fro') * torch.linalg.norm(K2, ord='fro') + eps)
    return numerator / denominator

def adaptive_relation_weight(feat: torch.Tensor, labels: torch.Tensor, alpha: float) -> torch.Tensor:
    feat_norm = feat / (torch.norm(feat, dim=1, keepdim=True) + 1e-10)
    cos_sim = feat_norm @ feat_norm.T
    same_class = (labels[:, None] == labels[None, :]).float()
    pos = 1.0 + alpha * cos_sim
    neg = torch.clamp(1.0 - alpha * cos_sim, min=0.0)
    return same_class * pos + (1.0 - same_class) * neg

def apply_strategy(name: str, K: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    if name not in STRATEGIES_PT:
        raise KeyError(f"Unknown strategy: {name}")
    if name == 'K':
        if labels is None:
            raise ValueError("Strategy K requires labels")
        out = STRATEGIES_PT[name](K, labels)
    else:
        out = STRATEGIES_PT[name](K)
    if not torch.isfinite(out).all():
        raise FloatingPointError(f"Strategy {name} produced NaN/Inf")
    return symmetrize(out)

@torch.no_grad()
def solve_soft_worst_weights(
    candidates: List[torch.Tensor],
    target: torch.Tensor,
    temperature: float = 0.30,
    uniform_mix: float = 0.40,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not candidates:
        raise ValueError("Candidate list is empty.")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must be in [0,1]")

    target_ref = frobenius_normalize(target.detach())
    candidate_refs = torch.stack([frobenius_normalize(c.detach()) for c in candidates], dim=0)

    alignments = torch.stack([kernel_alignment(target_ref, c) for c in candidate_refs])
    difficulties = 1.0 - alignments

    soft_weights = torch.softmax(difficulties / temperature, dim=0)
    n = soft_weights.numel()

    weights = (1.0 - uniform_mix) * soft_weights + uniform_mix / n
    weights = weights / weights.sum()
    return weights, alignments

# ---------- One training step ----------
def train_step(model, imgs, labels, optimizer, lambda_ka, alpha_couple, strategies,
               coupling='adaptive_convex_soft', num_classes=7, detach_target=True,
               convex_temperature=0.30, convex_uniform_mix=0.40):
    model.train()
    feat, logits = model(imgs, return_feat=True)
    loss_ce = F.cross_entropy(logits, labels)
    loss_ka = torch.tensor(0.0, device=device)
    weights_info = None
    slsqp_ok = True

    if lambda_ka > 0 and len(strategies) > 0:
        K = symmetrize(feat @ feat.T)
        A = adaptive_relation_weight(feat, labels, alpha_couple)

        if coupling == 'adaptive_convex_soft':
            # 1. Reference relation matrix
            reference_relation = frobenius_normalize(symmetrize(K * A))
            # 2. Generate candidate relation matrices
            candidate_relations = []
            for s in strategies:
                sim = apply_strategy(s, K, labels=labels)
                sim_rel = frobenius_normalize(symmetrize(sim * A))
                candidate_relations.append(sim_rel)
            if not candidate_relations:
                raise RuntimeError("No candidate relation matrices generated.")
            # 3. Soft worst-case weights
            weights, alignments = solve_soft_worst_weights(
                candidates=candidate_relations,
                target=reference_relation,
                temperature=convex_temperature,
                uniform_mix=convex_uniform_mix,
            )
            weights_info = weights.detach().cpu().numpy()
            # 4. Weighted aggregation
            stacked = torch.stack(candidate_relations, dim=0)
            weighted = torch.sum(weights.view(-1, 1, 1) * stacked, dim=0)
            weighted = symmetrize(weighted)
            # 5. Kernel alignment loss
            reference_for_loss = reference_relation.detach() if detach_target else reference_relation
            ka_val = kernel_alignment(reference_for_loss, weighted)
            loss_ka = 1.0 - ka_val
        else:
            raise NotImplementedError(f"Coupling '{coupling}' is not implemented.")

        loss = loss_ce + lambda_ka * loss_ka
    else:
        loss = loss_ce

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_ce.item(), loss_ka.item(), loss.item(), weights_info, slsqp_ok

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return 100.0 * correct / total

def build_transform(aug_type='original'):
    """Select a data augmentation pipeline."""
    if aug_type == 'minimal':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif aug_type == 'TA':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:  # 'original': the full augmentation pipeline
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            StrongColorJitter(p=0.3, hue=0.1, sat=0.5, val=0.5),
            transforms.RandomGrayscale(p=0.3),
            RandomGrayscaleWithEdge(p=0.2),
            RandomEdgeEnhance(p=0.3, mode='sketch'),
            RandomHighPass(p=0.2, cutoff_ratio=0.15),
            ColorQuantization(p=0.25, n_colors=5, sample_ratio=0.25),
            EdgePreservingFilter(p=0.3, d=7, sigma_color=50, sigma_space=50, sharp_strength=0.25),
            RandomElasticDeformation(p=0.15, alpha=40, sigma=4),
            MedianCutQuantization(p=0.2, n_colors=6),
            WhiteningTransform(p=0.2),
            LocalElasticDeformation(p=0.15, alpha=25, sigma=3),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def main():
    parser = argparse.ArgumentParser(description='DG with Adaptive Convex KA')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--source', type=str, default='')
    parser.add_argument('--sources', type=str, default='')
    parser.add_argument('--targets', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_ka', type=float, default=0.1)
    parser.add_argument('--alpha_couple', type=float, default=0.2)
    parser.add_argument('--strategies', type=str, default='A,C,D,F,H,I')
    parser.add_argument('--coupling', choices=['adaptive_convex_soft'], default='adaptive_convex_soft')
    parser.add_argument('--convex_temperature', type=float, default=0.30)
    parser.add_argument('--convex_uniform_mix', type=float, default=0.40)
    parser.add_argument('--ka_warmup', type=int, default=500)
    parser.add_argument('--aug_type', type=str, default='original',
                        choices=['original', 'minimal', 'TA'],
                        help='Data augmentation type')
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr_step', type=int, default=-1,
                        help='Step at which LR decays; if <=0, uses 80% of total iterations')
    parser.add_argument('--no_detach_target', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.sources:
        src_domains = [s.strip() for s in args.sources.split(',')]
    elif args.source:
        src_domains = [args.source]
    else:
        raise ValueError("Either --source or --sources required.")
    print(f"Source domains: {src_domains}")

    train_transform = build_transform(aug_type=args.aug_type)
    train_datasets = []
    reference_class_to_idx = None
    for dom in src_domains:
        ds = datasets.ImageFolder(os.path.join(args.data_root, dom), transform=train_transform)
        if reference_class_to_idx is None:
            reference_class_to_idx = ds.class_to_idx
        elif ds.class_to_idx != reference_class_to_idx:
            raise ValueError(f"Class mapping mismatch in domain '{dom}'.")
        train_datasets.append(ds)
    merged = ConcatDataset(train_datasets)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    src_loader = DataLoader(merged, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, drop_last=True,
                            worker_init_fn=seed_worker,
                            generator=loader_generator,
                            pin_memory=torch.cuda.is_available())
    num_classes = len(train_datasets[0].classes)

    model = ResNet50Linear(num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_step = args.lr_step if args.lr_step > 0 else int(args.iterations * 0.8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)
    strategies = args.strategies.split(',') if args.lambda_ka > 0 else []

    data_iter = iter(src_loader)
    for itr in range(1, args.iterations+1):
        try: imgs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(src_loader); imgs, labels = next(data_iter)
        imgs, labels = imgs.to(device), labels.to(device)

        warmup_ratio = min(1.0, itr / args.ka_warmup) if args.ka_warmup > 0 else 1.0
        current_lambda_ka = args.lambda_ka * warmup_ratio

        ce, ka, total, weights, success = train_step(
            model, imgs, labels, optimizer, current_lambda_ka, args.alpha_couple, strategies,
            args.coupling, num_classes, detach_target=not args.no_detach_target,
            convex_temperature=args.convex_temperature,
            convex_uniform_mix=args.convex_uniform_mix
        )
        scheduler.step()

        if itr % args.eval_every == 0 or itr == args.iterations:
            weight_str = ""
            if weights is not None:
                weight_str = " | " + ", ".join(f"{s}:{w:.3f}" for s,w in zip(strategies, weights))
            print(f'Iter {itr:5d}/{args.iterations} | CE: {ce:.4f} | KA Loss: {ka:.6e} | '
                  f'Alignment: {1.0-ka:.6e} | Total: {total:.4f} | {weight_str}')

    print('\n--- Final Evaluation on Target Domains ---')
    test_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    target_domains = [d.strip() for d in args.targets.split(',')]
    accs = []
    for dom in target_domains:
        ds = datasets.ImageFolder(os.path.join(args.data_root, dom), transform=test_transform)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        acc = evaluate(model, dl)
        accs.append(acc)
        print(f'{dom:>15s}: {acc:.2f}%')
    print(f'{"Average":>15s}: {np.mean(accs):.2f}%')

    os.makedirs('./checkpoints', exist_ok=True)
    src_name = '_'.join(src_domains)
    strategy_tag = ''.join(strategies) if strategies else 'none'
    save_name = (f'dg_{src_name}_{args.coupling}_strat-{strategy_tag}'
                 f'_lam-{args.lambda_ka:.0e}_temp-{args.convex_temperature}'
                 f'_mix-{args.convex_uniform_mix}_aug-{args.aug_type}_s{args.seed}.pth')
    save_path = os.path.join('./checkpoints', save_name)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

if __name__ == '__main__':
    main()
