#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Phase A diagnostics: noise ceiling, PCA dimensionality, AWQ shift analysis."""
import gc, json, sys, time, glob
import numpy as np
import torch

sys.path.insert(0, ".")
from src.adapters.trajectory_transformer import TrajectoryTransformer

DEVICE = "cuda"
BNB_MMAP = "/run/media/filip/C27C20AB7C209C63/trajs_7b_rmmap"
AWQ_MMAP = "/run/media/filip/C27C20AB7C209C63/trajs_7b_awq_rmmap"
N_LAYERS, D_INPUT = 28, 3584
SAMPLE = 2000

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

results = {}

# ── Load sample data ──────────────────────────────────────────────
print("Loading BnB sample...", flush=True)
meta = json.load(open(f"{BNB_MMAP}/meta.json"))
h_bnb = np.memmap(f"{BNB_MMAP}/train_h.bin", dtype="float16", mode="r",
                  shape=(meta["n_train"], N_LAYERS, D_INPUT))[:SAMPLE].copy()
v_bnb = np.memmap(f"{BNB_MMAP}/train_v.bin", dtype="float16", mode="r",
                  shape=(meta["n_train"], N_LAYERS, D_INPUT))[:SAMPLE].copy()
h_bnb_t = torch.from_numpy(h_bnb).float()
v_bnb_t = torch.from_numpy(v_bnb).float()
print(f"  BnB: {h_bnb_t.shape}, v_mean={v_bnb_t.mean():.4f}, v_std={v_bnb_t.std():.4f}", flush=True)

# Load AWQ sample
print("Loading AWQ sample...", flush=True)
meta_awq = json.load(open(f"{AWQ_MMAP}/meta.json"))
h_awq = np.memmap(f"{AWQ_MMAP}/train_h.bin", dtype="float16", mode="r",
                  shape=(meta_awq["n_train"], N_LAYERS, D_INPUT))[:SAMPLE].copy()
v_awq = np.memmap(f"{AWQ_MMAP}/train_v.bin", dtype="float16", mode="r",
                  shape=(meta_awq["n_train"], N_LAYERS, D_INPUT))[:SAMPLE].copy()
h_awq_t = torch.from_numpy(h_awq).float()
v_awq_t = torch.from_numpy(v_awq).float()
print(f"  AWQ: {h_awq_t.shape}, v_mean={v_awq_t.mean():.4f}, v_std={v_awq_t.std():.4f}", flush=True)

# ── 1. PCA Dimensionality (H-2) ───────────────────────────────────
print("\n=== H-2: PCA Dimensionality ===", flush=True)
v_flat = v_bnb_t.reshape(-1, D_INPUT).numpy()  # [SAMPLE*28, 3584]
v_mean_c = v_flat.mean(axis=0)
v_centered = v_flat - v_mean_c

# Compute SVD via randomized PCA on 3584-dim data
from sklearn.decomposition import PCA
pca = PCA(n_components=min(1024, SAMPLE * 28))
pca.fit(v_centered)
cumvar = np.cumsum(pca.explained_variance_ratio_)
dims_90 = int((cumvar < 0.90).sum()) + 1
dims_95 = int((cumvar < 0.95).sum()) + 1
dims_99 = int((cumvar < 0.99).sum()) + 1
results["pca"] = {"dims_90": dims_90, "dims_95": dims_95, "dims_99": dims_99,
                   "n_components": pca.n_components_}
print(f"  Dims for 90% var: {dims_90}", flush=True)
print(f"  Dims for 95% var: {dims_95}", flush=True)
print(f"  Dims for 99% var: {dims_99}", flush=True)
del pca, v_flat, v_centered; gc.collect()

# ── Layer-wise velocity std (for gradient imbalance H-3) ──────────
print("\n=== H-3: Layer Gradient Imbalance ===", flush=True)
v_bnb_std = v_bnb_t.std(dim=(0, 2))
results["layer_std"] = {f"L{l}": float(v_bnb_std[l].item()) for l in range(N_LAYERS)}
ratio = v_bnb_std[:10].mean().item() / v_bnb_std[18:].mean().item()
print(f"  Mean std L0-9: {v_bnb_std[:10].mean():.4f}", flush=True)
print(f"  Mean std L18-27: {v_bnb_std[18:].mean():.4f}", flush=True)
print(f"  Ratio early/late: {ratio:.2f}x", flush=True)
results["layer_imbalance_ratio"] = ratio

# ── 2. AWQ Shift Analysis (H-4) ───────────────────────────────────
print("\n=== H-4: AWQ Shift Analysis ===", flush=True)
# Check if AWQ hidden states are an affine transform of BnB's
# Align by concatenating and checking distribution shift
h_bnb_stats = {"mean": h_bnb_t.mean(dim=0).numpy().tolist(), "std": h_bnb_t.std(dim=0).numpy().tolist()}
h_awq_stats = {"mean": h_awq_t.mean(dim=0).numpy().tolist(), "std": h_awq_t.std(dim=0).numpy().tolist()}

# Compute shift magnitude: ||mean_AWQ - mean_BnB|| / ||mean_BnB||
mean_diff = (h_awq_t.mean(dim=0) - h_bnb_t.mean(dim=0)).norm().item()
mean_bnb_norm = h_bnb_t.mean(dim=0).norm().item()
print(f"  Hidden state mean shift: {mean_diff:.4f} (BnB norm: {mean_bnb_norm:.4f}, ratio: {mean_diff/max(mean_bnb_norm,1e-8):.4f})", flush=True)

# Per-layer shift
layer_shift = (h_awq_t.mean(dim=0) - h_bnb_t.mean(dim=0)).norm(dim=-1)
print(f"  Per-layer shift magnitude: min={layer_shift.min():.4f}, max={layer_shift.max():.4f}", flush=True)
results["awq_shift"] = {"mean_diff_ratio": mean_diff / max(mean_bnb_norm, 1e-8),
                         "layer_shift_magnitudes": layer_shift.tolist()}

# Fit linear correction: W @ h_AWQ ≈ h_BnB (on a subset)
print("  Fitting affine correction (500 samples)...", flush=True)
h_awq_sub = h_awq_t[:500].reshape(-1, D_INPUT)  # [14000, 3584]
h_bnb_sub = h_bnb_t[:500].reshape(-1, D_INPUT)
# Ridge regression
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(h_awq_sub.numpy(), h_bnb_sub.numpy())
pred = ridge.predict(h_awq_sub.numpy())
corr_r2 = 1 - ((pred - h_bnb_sub.numpy())**2).sum() / (h_bnb_sub.numpy().var() * len(h_bnb_sub))
results["awq_affine_r2"] = float(corr_r2)
print(f"  Affine correction R²: {corr_r2:.4f}", flush=True)
h_bnb_var = h_bnb_sub.var().item()
h_awq_var = h_awq_sub.var().item()
print(f"  BnB var: {h_bnb_var:.4f}, AWQ var: {h_awq_var:.4f}, ratio: {h_awq_var/max(h_bnb_var,1e-8):.4f}", flush=True)

# ── 3. Noise Ceiling (H-6) ────────────────────────────────────────
print("\n=== H-6: Noise Ceiling ===", flush=True)
# Train a small TT on a subset and measure train vs val error gap
tt = TrajectoryTransformer(d_model=768, n_layers=6, n_heads=8, d_ff=3072,
                            n_positions=N_LAYERS, d_input=D_INPUT).to(DEVICE)
opt = torch.optim.AdamW(tt.parameters(), lr=3e-4, weight_decay=1e-4)

# Use first 80% for train, last 20% for val
n_train = int(0.8 * SAMPLE)
h_tr, v_tr = h_bnb_t[:n_train].to(DEVICE), v_bnb_t[:n_train].to(DEVICE)
h_val, v_val = h_bnb_t[n_train:].to(DEVICE), v_bnb_t[n_train:].to(DEVICE)

# Global normalization
v_mean = v_tr.mean(dim=(0, 1), keepdim=True)
v_std = v_tr.std(dim=(0, 1), keepdim=True) + 1e-8
v_tr_norm = (v_tr - v_mean) / v_std

tt.train()
for epoch in range(20):
    perm = torch.randperm(n_train)
    epoch_loss = 0
    for i in range(0, n_train, 64):
        idx = perm[i:i+64]
        pred = tt(h_tr[idx])
        loss = (pred - v_tr_norm[idx]).pow(2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(tt.parameters(), 1.0)
        opt.step()
        epoch_loss += loss.item()
    
    tt.eval()
    with torch.no_grad():
        v_pred = tt(h_val)
        pred_unnorm = v_pred * v_std + v_mean
        mse = (pred_unnorm - v_val).pow(2).mean().item()
        var = v_val.var().item()
        r2 = 1 - mse / max(var, 1e-8)
        cos = (pred_unnorm * v_val).sum(-1) / (pred_unnorm.norm(dim=-1) * v_val.norm(dim=-1) + 1e-8)
        cos_m = cos.mean().item()
    print(f"  ep={epoch+1:2d} loss={epoch_loss/max((n_train+63)//64,1):.4f} val_r²={r2:.4f} cos={cos_m:.4f}", flush=True)
    tt.train()

results["noise_ceiling"] = {"val_r2": r2, "val_cos": cos_m, "n_train": n_train, "n_val": SAMPLE - n_train}
print(f"\n  Noise ceiling est: R²={r2:.4f}, unexplained var={1-r2:.4f} (= noise + unlearned signal)", flush=True)

# ── Save ──────────────────────────────────────────────────────────
json.dump(results, open("phase_a_diagnostics.json", "w"), indent=2)
print(f"\nSaved to phase_a_diagnostics.json", flush=True)
