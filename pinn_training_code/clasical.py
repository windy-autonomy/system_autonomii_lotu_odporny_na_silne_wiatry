import torch
import torch.nn as nn

class TrigAeroModel(nn.Module):
    def __init__(self, rho=1.225, area=0.5, chord=0.2, degree=3):
        super().__init__()
        self.rho = rho
        self.area = area
        self.chord = chord
        self.degree = degree
        
        # Trainable parameters for Fourier series expansion up to 'degree'
        self.sin_coeffs = nn.Parameter(torch.randn(3, degree) * 0.1)
        self.cos_coeffs = nn.Parameter(torch.randn(3, degree) * 0.1)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, velocity, alpha):
        q = 0.5 * self.rho * (velocity ** 2)
        
        # Create multiples of alpha:[alpha, 2*alpha, 3*alpha, ...]
        n = torch.arange(1, self.degree + 1, dtype=torch.float32, device=alpha.device)
        
        # Shapes: (batch_size, degree)
        sin_terms = torch.sin(alpha * n)
        cos_terms = torch.cos(alpha * n)
        
        # Output is (batch_size, 3) mapping to [C_L, C_D, C_M]
        coeffs = (sin_terms @ self.sin_coeffs.T) + (cos_terms @ self.cos_coeffs.T) + self.bias
        
        C_L = coeffs[:, 0:1]
        C_D = torch.nn.functional.softplus(coeffs[:, 1:2])
        C_M = coeffs[:, 2:3]
        
        L = q * self.area * C_L
        D = q * self.area * C_D
        Moment = q * self.area * self.chord * C_M
        
        F_x = L * torch.sin(alpha) - D * torch.cos(alpha)
        F_y = L * torch.cos(alpha) + D * torch.sin(alpha)
        
        return torch.cat([F_x, F_y, Moment], dim=-1)