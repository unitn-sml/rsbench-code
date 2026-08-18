import torch
import torch.nn as nn


class RavenPanelEncoder(nn.Module):
    """
    Encoder for a single RAVEN panel (160x160).
    """
    def __init__(self, latent_dim=9):
        super(RavenPanelEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160 * 160, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim), 
        )
        
    def forward(self, x):
        return self.backbone(x)


class RavenMLP(nn.Module):
    """
    Simple MLP for RAVEN 160x160 grayscale images.
    Input: [batch, 16, 1, 160, 160] (processed one by one)
    Output: [batch, 16, latent_dim]

    3x3x3 attribute layout:
        Type(3) | Size(3) | Color(3) = 9 total
    """
    NAME = "RavenMLP"

    def __init__(self, latent_dim=9):
        super(RavenMLP, self).__init__()
        self.latent_dim = latent_dim
        self.panel_encoder = RavenPanelEncoder(latent_dim=self.latent_dim)

    def forward(self, x):
        # x shape: [batch, 16, 1, 160, 160]
        
        # Flatten batch and n_images for efficient parallel processing
        b, n, c, h, w = x.shape
        # Reshape to [B*N, C*H*W] compatible with Flatten in panel_encoder
        # treat every single panel as an independent sample to pass it through the encoder
        x_flat = x.reshape(b * n, -1) 
        
        # Process all panels in one go (Efficient View)
        z = self.panel_encoder(x_flat) # [b*16, latent_dim]
        
        # Reshape back to [batch, 16, latent_dim]
        z = z.view(b, n, -1)
        
        # Return tuple (output, None) to match interface expected by DPL models
        return z, None
 
if __name__ == "__main__":
    """
    Standalone training loop to verify that RavenMLP can learn to predict
    concept attributes (Type, Size, Color) from 160x160 panel images.

    This is a *concept-only* sanity check — no answer-classification head.
    We train with CrossEntropyLoss per attribute on the 8 context panels
    and report per-attribute accuracy on the validation set.
    """
    import sys, os
    # Allow imports from the rss root when running as a script
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from torch.utils.data import DataLoader
    from datasets.utils.raven_creation import RAVEN_Dataset
    from utils.losses import RAVEN_Concept_Match

    # ── Hyperparameters ─────────────────────────────────────────────
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_VALUES = 3  # 3 for 3x3x3, 4 for 4x4x4
    LATENT_DIM = N_VALUES * 3
    DATA_PATH = f"data/RAVEN-{N_VALUES}x{N_VALUES}x{N_VALUES}"

    ATTR_NAMES = ["Type", "Size", "Color"]
    ATTR_SLICES = [(0, N_VALUES), (N_VALUES, 2 * N_VALUES), (2 * N_VALUES, 3 * N_VALUES)]

    # ── Data ────────────────────────────────────────────────────────
    train_ds = RAVEN_Dataset(base_path=DATA_PATH, config="center_single", split="train")
    val_ds   = RAVEN_Dataset(base_path=DATA_PATH, config="center_single", split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
    print(f"Device: {DEVICE}")

    # ── Model & Optimizer ───────────────────────────────────────────
    model = RavenMLP(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Evaluation helper ───────────────────────────────────────────
    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        correct = {name: 0 for name in ATTR_NAMES}
        total = 0

        for images, target, concepts in loader:
            images = images.to(DEVICE)
            concepts = concepts.to(DEVICE)

            z, _ = model(images)

            out_dict = {"CS": z, "CONCEPTS": concepts}
            loss, _ = RAVEN_Concept_Match(out_dict)
            total_loss += loss.item() * images.size(0)

            # Per-attribute accuracy on context panels (0-7)
            z_ctx = z[:, :8]              # [B, 8, 9]
            c_ctx = concepts[:, :8]       # [B, 8, 4]

            for i, (name, (lo, hi)) in enumerate(zip(ATTR_NAMES, ATTR_SLICES)):
                preds = z_ctx[..., lo:hi].argmax(dim=-1)  # [B, 8]
                correct[name] += (preds == c_ctx[..., i]).sum().item()

            total += z_ctx.shape[0] * z_ctx.shape[1]  # B * 8

        avg_loss = total_loss / len(loader.dataset)
        accs = {name: correct[name] / total * 100 for name in ATTR_NAMES}
        return avg_loss, accs

    # ── Training loop ───────────────────────────────────────────────
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | "
          + " | ".join(f"{n:>7}" for n in ATTR_NAMES)
          + " | Avg Acc")
    print("-" * 75)

    best_avg_acc = 0.0
    best_val_loss = float("inf")
    patience = 3
    min_delta = 0.01
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for images, target, concepts in train_loader:
            images = images.to(DEVICE)
            concepts = concepts.to(DEVICE)

            z, _ = model(images)

            out_dict = {"CS": z, "CONCEPTS": concepts}
            loss, _ = RAVEN_Concept_Match(out_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = epoch_loss / len(train_ds)

        # Validation
        val_loss, val_accs = evaluate(val_loader)
        avg_acc = sum(val_accs.values()) / len(val_accs)

        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:8.4f} | "
              + " | ".join(f"{val_accs[n]:6.2f}%" for n in ATTR_NAMES)
              + f" | {avg_acc:5.2f}%")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc

        # Early stopping on validation loss
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no val loss improvement for {patience} epochs)")
                break

    print(f"\nBest average validation accuracy: {best_avg_acc:.2f}%")