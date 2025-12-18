import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class SineTaskGenerator:
    def __init__(self):
        pass

    def sample_task(self):
        """Sample a random sine wave task"""
        amplitude = np.random.uniform(0.1, 5.0)
        phase = np.random.uniform(0, np.pi)
        return amplitude, phase

    def generate_data(self, amplitude, phase, num_samples=10):
        """Generate data for a specific sine wave"""
        x = np.random.uniform(-5, 5, num_samples)
        y = amplitude * np.sin(x + phase)

        x = torch.FloatTensor(x).unsqueeze(1)
        y = torch.FloatTensor(y).unsqueeze(1)
        return x, y


class SimpleMLP(nn.Module):
    def __init__(self, hidden_size=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


gen = SineTaskGenerator()
amp, phase = gen.sample_task()
x_train, y_train = gen.generate_data(amp, phase, num_samples=10)
x_test, y_test = gen.generate_data(amp, phase, num_samples=10)

print(f"Task: amplitude={amp:.2f}, phase={phase:.2f}")
print(f"Train data: x shape {x_train.shape}, y shape {y_train.shape}")

model = SimpleMLP(hidden_size=40)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
