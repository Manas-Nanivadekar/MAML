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


def inner_loop(model, x_support, y_support, inner_lr=0.01, inner_steps=5):
    params = list(model.parameters())

    for step in range(inner_steps):
        if step == 0:
            y_pred = model(x_support)
        else:
            y_pred = functional_forward(model, x_support, params)
    loss = nn.MSELoss()(y_pred, y_support)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    params = [p - inner_lr * g for p, g in zip(params, grads)]
    return params


def outer_loop_loss(model, adapted_params, x_query, y_query):
    y_pred = functional_forward(model, x_query, adapted_params)
    loss = nn.MSELoss()(y_pred, y_query)
    return loss


def functional_forward(model, x, params):
    x = torch.relu(torch.mm(x, params[0].t()) + params[1])
    x = torch.relu(torch.mm(x, params[2].t()) + params[3])
    x = torch.mm(x, params[4].t()) + params[5]
    return x


def train_maml(
    num_iterations=10000,
    num_tasks_per_batch=4,
    inner_lr=0.01,
    outer_lr=0.001,
    inner_steps=5,
):
    model = SimpleMLP(hidden_size=40)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    task_generator = SineTaskGenerator()

    for iteration in range(num_iterations):
        meta_optimizer.zero_grad()
        meta_loss = 0.0

        for task_idx in range(num_tasks_per_batch):
            amp, phase = task_generator.sample_task()

            x_support, y_support = task_generator.generate_data(
                amp, phase, num_samples=10
            )
            x_query, y_query = task_generator.generate_data(amp, phase, num_samples=10)

            adapted_params = inner_loop(
                model, x_support, y_support, inner_lr=inner_lr, inner_steps=inner_steps
            )

            task_loss = outer_loop_loss(model, adapted_params, x_query, y_query)
            meta_loss += task_loss

        meta_loss = meta_loss / num_tasks_per_batch

        meta_loss.backward()
        meta_optimizer.step()

        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, Meta Loss: {meta_loss.item():.4f}")

    return model


def test_maml(model, num_test_tasks=5):
    task_generator = SineTaskGenerator()
    inner_lr = 0.01

    fig, axes = plt.subplots(1, num_test_tasks, figsize=(20, 4))
    for idx in range(num_test_tasks):
        amp, phase = task_generator.sample_task()
        x_support, y_support = task_generator.generate_data(amp, phase, num_samples=10)

        x_test = torch.linspace(-5, 5, 100).unsqueeze(1)
        y_test_true = amp * torch.sin(x_test + phase)

        with torch.no_grad():
            y_pred_before = model(x_test)

        adapted_params = inner_loop(
            model, x_support, y_support, inner_lr=inner_lr, inner_steps=50
        )

        with torch.no_grad():
            y_pred_after = functional_forward(model, x_test, adapted_params)

        ax = axes[idx]
        ax.plot(x_test.numpy(), y_test_true.numpy(), "k-", label="True", linewidth=2)
        ax.plot(
            x_test.numpy(),
            y_pred_before.numpy(),
            "b--",
            label="Before adaptation",
            alpha=0.6,
        )
        ax.plot(
            x_test.numpy(),
            y_pred_after.numpy(),
            "r-",
            label="After 50 steps",
            linewidth=2,
        )
        ax.scatter(
            x_support.numpy(),
            y_support.numpy(),
            c="green",
            s=100,
            marker="x",
            label="Support data",
        )
        ax.set_title(f"Task {idx+1}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/maml_results.png", dpi=150, bbox_inches="tight")
    plt.close()


model = train_maml()
test_maml(model)
