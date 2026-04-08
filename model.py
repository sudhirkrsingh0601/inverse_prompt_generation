import torch.nn as nn

class PromptGenerator(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, embed_dim)
        )

    def forward(self, x):
        return self.net(x)
