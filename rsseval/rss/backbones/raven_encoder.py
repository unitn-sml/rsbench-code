import torch.nn as nn

class RavenPanelEncoder(nn.Module):
    """
    Encoder for a single RAVEN panel (160x160).
    """
    def __init__(self, latent_dim=31):
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
    """
    NAME = "RavenMLP"

    def __init__(self, latent_dim=30):
        super(RavenMLP, self).__init__()
        
        # 31 dimensions:
        # Type (5) + Size (6) + Color (10) + Number (9) = 30
        self.latent_dim = latent_dim
        
        # Use the independent panel encoder
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
    import torch
    encoder = RavenMLP()
    x = torch.randn(1, 16, 1, 160, 160)
    z, _ = encoder(x)
    print(f"z shape: {z.shape}")
    print(z)