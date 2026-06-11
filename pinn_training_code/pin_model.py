import torch
import torch.nn as nn

class NNAeroModel(nn.Module):
    def __init__(self, rho=1.225, area=0.5, chord=0.2):
        super().__init__()
        self.rho = rho
        self.area = area
        self.chord = chord
        
        self.coef_net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, velocity, alpha):
        q = 0.5 * self.rho * (velocity ** 2)
        coeffs = self.coef_net(alpha)
        
        C_L = coeffs[:, 0:1]
        C_D = torch.nn.functional.softplus(coeffs[:, 1:2]) # Force drag to be positive
        C_M = coeffs[:, 2:3]
        
        L = q * self.area * C_L
        D = q * self.area * C_D
        Moment = q * self.area * self.chord * C_M
        
        F_x = L * torch.sin(alpha) - D * torch.cos(alpha)
        F_y = L * torch.cos(alpha) + D * torch.sin(alpha)
        
        return torch.cat([F_x, F_y, Moment], dim=-1)