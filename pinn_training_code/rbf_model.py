import torch
import torch.nn as nn

class RBFAeroModel(nn.Module):
    def __init__(self, rho=1.225, area=0.5, chord=0.2, num_centers=5):
        super().__init__()
        self.rho = rho
        self.area = area
        self.chord = chord
        
        # Place RBF centers evenly between -45 and +45 degrees
        centers_init = torch.linspace(-0.8, 0.8, num_centers).unsqueeze(0)
        self.centers = nn.Parameter(centers_init) 
        self.gamma = nn.Parameter(torch.tensor([1.0])) # Width of the Gaussian RBF
        
        # Linear map from local RBF activations to Aerodynamic coefficients
        self.linear = nn.Linear(num_centers, 3)

    def forward(self, velocity, alpha):
        q = 0.5 * self.rho * (velocity ** 2)
        
        # Compute RBF activations: exp(-gamma * (x - c)^2)
        # alpha shape: (batch_size, 1), centers: (1, num_centers) -> dist_sq: (batch_size, num_centers)
        dist_sq = (alpha - self.centers) ** 2
        rbf_acts = torch.exp(-self.gamma * dist_sq)
        
        coeffs = self.linear(rbf_acts)
        
        C_L = coeffs[:, 0:1]
        C_D = torch.nn.functional.softplus(coeffs[:, 1:2])
        C_M = coeffs[:, 2:3]
        
        L = q * self.area * C_L
        D = q * self.area * C_D
        Moment = q * self.area * self.chord * C_M
        
        F_x = L * torch.sin(alpha) - D * torch.cos(alpha)
        F_y = L * torch.cos(alpha) + D * torch.sin(alpha)
        
        return torch.cat([F_x, F_y, Moment], dim=-1)