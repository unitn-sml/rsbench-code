
import os
import glob
import numpy as np
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

class RAVEN_Dataset(Dataset):
    """
    RAVEN Dataset for RSBench.
    Loads RAVEN-3x3x3 .npz files (3 Types x 3 Sizes x 3 Colors).

    Supports partial concept supervision on the training split through:
    - c_sup: fraction of training samples with concept supervision
    - which_c: attribute indices to supervise (0=Type, 1=Size, 2=Color)

    Unsupervised concept targets are masked with -1, analogously to the
    other rsbench datasets.
    """
    def __init__(self, base_path, config="center_single", split="train", c_sup=1, which_c=[-1]):
        self.base_path = base_path
        self.config = config
        self.split = split
        self.c_sup = c_sup
        self.which_c = which_c
        self.is_train = split == "train"
        
        # Search pattern for files
        pattern = os.path.join(self.base_path, self.config, f"RAVEN_*_{self.split}.npz")
        self.all_files = sorted(glob.glob(pattern))
        
        if len(self.all_files) == 0:
            print(f"Warning: No files found for {pattern}")
            self.files = []
        else:
            self.files = self.all_files

        # Deterministic supervision mask, analogous to other rsbench datasets.
        rng = np.random.RandomState(0)
        self.r_seq = rng.rand(len(self.files)) if len(self.files) > 0 else np.array([])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        
        # Images: [16, 160, 160]
        # Normalize to [0, 1] and add channel dimension: [16, 1, 160, 160]
        images = data['image'].astype(np.float32) / 255.0
        images = torch.from_numpy(images).unsqueeze(1)
        
        # Target: 0-7
        target = torch.tensor(data['target'], dtype=torch.long)
        
        # Concepts: Extract attributes for all 16 panels (8 context + 8 choices)
        # meta_matrix does NOT contain entity values (like Type=Triangle), only Rule activity.
        # So we must parse XML to get ground truth concepts for supervision.
        
        concepts = self._extract_concepts_from_xml(file_path.replace('.npz', '.xml'))

        if self.is_train:
            concepts = self._apply_concept_supervision_mask(concepts, idx)

        return images, target, concepts

    def _apply_concept_supervision_mask(self, concepts, idx):
        """Mask concepts with -1 according to c_sup / which_c.

        We only supervise the first 3 modeled attributes (Type, Size, Color)
        on the 8 context panels, mirroring RAVEN_Concept_Match.
        """
        concepts = concepts.clone()

        # Sample-level supervision fraction.
        if self.r_seq[idx] > self.c_sup:
            concepts[:8, :3] = -1
            return concepts

        # Attribute-level supervision subset.
        if not (len(self.which_c) == 1 and self.which_c[0] == -1):
            for attr_idx in range(3):
                if attr_idx not in self.which_c:
                    concepts[:8, attr_idx] = -1

        return concepts

    def _extract_concepts_from_xml(self, xml_path):
        """
        Extract concept values for all 16 panels.
        Returns tensor of shape [16, 4] -> (Type, Size, Color, Number)
        Values are indices 0-2 for Type/Size/Color (3x3x3 dataset).
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        panels_data = []
        all_panels = root.findall('.//Panel')
        
        # 3x3x3 constrained dataset:
        # Type: 1=triangle, 2=pentagon, 3=circle (in XML) -> 0-2 after -1 shift
        # Size: 0=small(0.4), 1=medium(0.6), 2=large(0.9)
        # Color: 0=white(255), 1=gray(140), 2=black(0)
        
        for panel in all_panels[:16]:
            entity = panel.find('.//Entity')
            layout = panel.find('.//Layout')
            
            # Defaults
            p_c = [0, 0, 0, 0] # Type, Size, Color, Number
            
            if entity is not None:
                try: p_c[0] = int(entity.get('Type', 0)) - 1 # Shift 1-3 to 0-2
                except: pass
                try: p_c[1] = int(entity.get('Size', 0))
                except: pass
                try: p_c[2] = int(entity.get('Color', 0))
                except: pass
            
            if layout is not None:
                try: p_c[3] = int(layout.get('Number', 0))
                except: pass
            
            panels_data.append(p_c)
            
        return torch.tensor(panels_data, dtype=torch.long)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from collections import Counter

    TYPE_LABELS = ["Triangle", "Pentagon", "Circle"]
    SIZE_LABELS = ["Small", "Medium", "Large"]
    COLOR_LABELS = ["White", "Gray", "Black"]

    dataset = RAVEN_Dataset(
        base_path="data/RAVEN-3x3x3",
        config="center_single",
        split="train",
    )
    print(f"Dataset size: {len(dataset)} samples")

    def extract_rules(xml_path):
        """Parse <Rule> elements from XML and return dict {attr: rule_name}."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        rules = {}
        for rule_el in root.findall('.//Rule'):
            attr = rule_el.get('attr', '')
            name = rule_el.get('name', '')
            rules[attr] = name
        return rules

    # ── Fig 1: Three sample puzzles ──────────────────────────────────
    n_samples = 3
    fig, axes = plt.subplots(n_samples, 5, figsize=(16, 3.5 * n_samples + 1),
                             gridspec_kw={"width_ratios": [3, 0.15, 2, 0.15, 2]})
    fig.suptitle("RAVEN-3×3×3  —  Sample Puzzles", fontsize=15, fontweight="bold", y=0.98)

    for row in range(n_samples):
        images, target, concepts = dataset[row]
        imgs = images[:, 0].numpy()          # [16, 160, 160]
        target_idx = target.item()
        c = concepts.numpy()                 # [16, 4]

        # Extract rules from XML
        xml_path = dataset.files[row].replace('.npz', '.xml')
        rules = extract_rules(xml_path)

        # --- Left: 3×3 context matrix (panel 9 = "?") ---
        grid = np.ones((160 * 3 + 4, 160 * 3 + 4)) * 0.85
        for i in range(9):
            r, col = divmod(i, 3)
            y0 = r * (160 + 2)
            x0 = col * (160 + 2)
            if i < 8:
                grid[y0:y0 + 160, x0:x0 + 160] = imgs[i]
            else:
                patch = np.ones((160, 160)) * 0.6
                # draw a "?"
                grid[y0:y0 + 160, x0:x0 + 160] = patch
        ax = axes[row, 0]
        ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
        # Build rule annotation string
        rule_parts = []
        for attr in ["Type", "Size", "Color"]:
            rname = rules.get(attr, "?")
            rule_parts.append(f"{attr}: {rname}")
        rule_str = "  |  ".join(rule_parts)
        ax.set_title(f"Sample {row}\n{rule_str}", fontsize=10, fontweight="bold")
        ax.axis("off")

        # spacer
        axes[row, 1].axis("off")
        axes[row, 3].axis("off")

        # --- Middle: answer choices 0-3 ---
        ans_grid = np.ones((160 * 2 + 2, 160 * 2 + 2)) * 0.85
        for j in range(4):
            r, col = divmod(j, 2)
            y0 = r * (160 + 2)
            x0 = col * (160 + 2)
            ans_grid[y0:y0 + 160, x0:x0 + 160] = imgs[8 + j]
        ax = axes[row, 2]
        ax.imshow(ans_grid, cmap="gray", vmin=0, vmax=1)
        # highlight correct answer with a green rectangle
        if target_idx < 4:
            r, col = divmod(target_idx, 2)
            rect = plt.Rectangle((col * 162 - 1, r * 162 - 1), 162, 162,
                                 linewidth=3, edgecolor="limegreen", facecolor="none")
            ax.add_patch(rect)
        ax.set_title("Choices 0-3", fontsize=10)
        ax.axis("off")

        # --- Right: answer choices 4-7 ---
        ans_grid2 = np.ones((160 * 2 + 2, 160 * 2 + 2)) * 0.85
        for j in range(4):
            r, col = divmod(j, 2)
            y0 = r * (160 + 2)
            x0 = col * (160 + 2)
            ans_grid2[y0:y0 + 160, x0:x0 + 160] = imgs[12 + j]
        ax = axes[row, 4]
        ax.imshow(ans_grid2, cmap="gray", vmin=0, vmax=1)
        if target_idx >= 4:
            r, col = divmod(target_idx - 4, 2)
            rect = plt.Rectangle((col * 162 - 1, r * 162 - 1), 162, 162,
                                 linewidth=3, edgecolor="limegreen", facecolor="none")
            ax.add_patch(rect)
        ax.set_title("Choices 4-7", fontsize=10)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("raven_3x3x3_samples.png", dpi=150, bbox_inches="tight")
    print("Saved raven_3x3x3_samples.png")
    plt.show()

    # ── Fig 2: Attribute distributions ───────────────────────────────
    n_check = min(500, len(dataset))
    type_counts = Counter()
    size_counts = Counter()
    color_counts = Counter()

    for idx in range(n_check):
        _, _, concepts = dataset[idx]
        # Use context panels 0-7 + correct candidate (panel 8+target)
        type_counts.update(concepts[:8, 0].tolist())
        size_counts.update(concepts[:8, 1].tolist())
        color_counts.update(concepts[:8, 2].tolist())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Attribute Distributions  (first {n_check} samples, context panels)",
                 fontsize=13, fontweight="bold")

    bar_colors = ["#4C72B0", "#55A868", "#C44E52"]

    for ax, counts, labels, name in [
        (axes[0], type_counts, TYPE_LABELS, "Type"),
        (axes[1], size_counts, SIZE_LABELS, "Size"),
        (axes[2], color_counts, COLOR_LABELS, "Color"),
    ]:
        vals = [counts.get(i, 0) for i in range(3)]
        bars = ax.bar(labels, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    str(v), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("raven_3x3x3_distributions_test.png", dpi=150, bbox_inches="tight")
    print("Saved raven_3x3x3_distributions_test.png")
    plt.show()