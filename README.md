# Bi-Level Optimization and Meta-Learning: MAML Implementation

A ground-up implementation of Model-Agnostic Meta-Learning (MAML) and First-Order MAML (FOMAML) with focus on mathematical intuition and PyTorch mechanics.

---

## Table of Contents

1. [Overview](#overview)
2. [Bi-Level Optimization Fundamentals](#bi-level-optimization-fundamentals)
3. [MAML: Model-Agnostic Meta-Learning](#maml-model-agnostic-meta-learning)
4. [Implementation Details](#implementation-details)
5. [FOMAML: First-Order Approximation](#fomaml-first-order-approximation)

---

## Overview

**Goal:** Learn to optimize hyperparameters/meta-parameters through gradient descent rather than manual tuning.

**Problem Setup:**

- Standard ML: Learn weights θ to minimize training loss
- Meta-Learning: Learn **initial weights** θ that enable quick adaptation to new tasks

**Analogy:** Instead of learning one skill, learn how to learn skills quickly.

---

## Bi-Level Optimization Fundamentals

### The Core Problem

Traditional single-level optimization:

```
min_θ L_train(θ)
```

Bi-level optimization structure:

```
Outer problem: min_θ L_val(w*(θ))
Inner problem: w*(θ) = argmin_w L_train(w; θ)
```

### Dependency Chain

The validation loss depends on θ through the trained weights:

```
θ → [training process] → w* → L_val(w*)
```

To optimize θ, we need: `∂L_val/∂θ = ∂L_val/∂w* × ∂w*/∂θ`

**The Challenge:** Computing `∂w*/∂θ` requires backpropagating through the entire training process!

### Connection to RNNs

The inner optimization loop is structurally identical to RNN sequences:

**RNN (Backprop Through Time):**

```
h₀ → h₁ → h₂ → ... → h_T → output
```

**Meta-Learning (Backprop Through Optimization):**

```
w₀ → w₁ → w₂ → ... → w_T → L_val
```

Both face the same challenges:

- Memory explosion (storing all intermediate states)
- Vanishing/exploding gradients through long sequences
- Same solutions: truncation, first-order approximations

---

## MAML: Model-Agnostic Meta-Learning

### Objective

Learn initial parameters θ such that a model can quickly adapt to new tasks with minimal data.

### Two-Loop Structure

**Inner Loop (Task Adaptation):**

- Input: Initial weights θ, support set D_train
- Process: K gradient steps on support set
- Output: Adapted weights θ'
- Goal: Minimize training loss for this specific task

**Outer Loop (Meta-Optimization):**

- Input: Adapted weights θ', query set D_test
- Process: Evaluate θ' on query set
- Output: Meta-gradient for θ
- Goal: Improve initial weights based on adaptation quality

### Mathematical Formulation

For a single task:

**Inner Loop (K steps):**

```python
θ' = θ  # Start with meta-parameters
for k in range(K):
    θ' = θ' - α∇L_train(θ')  # Task-specific adaptation
```

**Outer Loop:**

```python
meta_loss = L_test(θ')  # Evaluate adapted parameters
θ = θ - β∇_θ L_test(θ')  # Update meta-parameters
```

**Key Insight:** The gradient `∇_θ L_test(θ')` flows through the K inner loop updates!

### Second-Order Derivatives

This is where MAML becomes computationally expensive:

```
∂L_test/∂θ requires ∂θ'/∂θ
```

Since θ' was computed using gradients, we need **gradients of gradients** (second-order derivatives).

In PyTorch, this requires `create_graph=True` in `torch.autograd.grad()`.

---

## Implementation Details

### Task Generator: Sine Wave Regression

Simple but effective test bed for meta-learning:

- Each task = sine wave with random amplitude and phase
- Support set: 10 points for adaptation
- Query set: 10 points for evaluation
- Goal: Learn to quickly fit any sine wave from minimal data

### Network Architecture

Simple 2-layer MLP:

```python
Input (1D) → Linear(40) → ReLU → Linear(40) → ReLU → Linear(1) → Output
```

### Full MAML Algorithm

```python
def inner_loop(model, x_support, y_support, inner_lr, inner_steps):
    params = list(model.parameters())

    for step in range(inner_steps):
        y_pred = functional_forward(model, x_support, params)
        loss = MSELoss()(y_pred, y_support)

        # CRITICAL: create_graph=True enables second-order derivatives
        grads = torch.autograd.grad(loss, params, create_graph=True)

        # Manual gradient descent
        params = [p - inner_lr * g for p, g in zip(params, grads)]

    return params

def train_maml():
    for iteration in range(num_iterations):
        meta_loss = 0

        for task in sample_tasks():
            # Inner loop: adapt to task
            adapted_params = inner_loop(model, x_support, y_support)

            # Outer loop: evaluate adaptation
            task_loss = compute_loss(model, adapted_params, x_query, y_query)
            meta_loss += task_loss

        # Meta-gradient update
        meta_loss.backward()  # Backprops through ALL inner loops!
        meta_optimizer.step()
```

### Memory and Computation Cost

For K inner steps across B tasks:

- **Memory:** O(K × B × model_size) - must store entire computational graph
- **Computation:** O(K × B) forward-backward passes
- **Problem:** Explodes with large K (similar to BPTT in RNNs)

---

## FOMAML: First-Order Approximation

### The Approximation

**Full MAML:** Compute `∂L_test/∂θ` through all K inner steps (second-order)

**FOMAML:** Ignore dependence of θ' on θ during backprop (first-order)

```
Full:    ∂L_test/∂θ = ∂L_test/∂θ' × ∂θ'/∂θ
FOMAML:  ∂L_test/∂θ ≈ ∂L_test/∂θ'
```

Stop gradient flow at adapted parameters θ'.

### Implementation Change

Single line difference:

```python
# MAML
grads = torch.autograd.grad(loss, params, create_graph=True)

# FOMAML
grads = torch.autograd.grad(loss, params, create_graph=False)
params = [p.detach().requires_grad_(True) for p in params]
```

### Benefits

1. **Memory:** O(model_size) - constant memory regardless of K
2. **Computation:** Much faster - no second-order derivatives
3. **Stability:** Often more stable gradients
4. **Performance:** Empirically performs nearly as well as full MAML

---

## Key Learnings

### 1. The Overfitting Problem

**Observation:** More inner steps ≠ better performance

| Inner Steps | Result                                  |
| ----------- | --------------------------------------- |
| 5           | Good generalization, reasonable fit     |
| 20          | Similar or slightly better              |
| 50          | Starting to overfit support set         |
| 100         | Severe overfitting, poor generalization |

**Why?**

- With 10 training points and 100 gradient steps, the model **memorizes** rather than generalizes
- The network finds simplest function fitting those exact points (often nearly flat)
- Completely ignores the true underlying pattern (sine wave)

**Meta-Learning Insight:** The sweet spot is typically 5-10 inner steps. Enough to adapt, not enough to severely overfit.

### 2. Second-Order vs First-Order

**When Second-Order Fails:**

- Deep unrolling (K > 20) → vanishing/exploding gradients
- Overfitting in inner loop → corrupted meta-gradients
- High memory cost limits batch size

**Why First-Order Often Works:**

- Cleaner gradient signal (ignores noisy dynamics)
- Can use more inner steps without memory explosion
- More stable training

**Trade-off:** FOMAML sacrifices some gradient accuracy for stability and efficiency.

### 3. Bi-Level Optimization Challenges

Three fundamental approaches to handle the computational graph:

| Approach      | Method                        | Pros            | Cons                   |
| ------------- | ----------------------------- | --------------- | ---------------------- |
| Full Backprop | Unroll entire inner loop      | Exact gradients | Memory explosion, slow |
| Truncation    | Backprop through last K steps | Balanced        | Approximate gradients  |
| First-Order   | Stop gradient at θ'           | Fast, stable    | Most approximate       |

### 4. The RNN Connection

Both meta-learning and RNN training face identical structural challenges:

- Sequential dependencies (h*t depends on h*{t-1}, w*t depends on w*{t-1})
- Long computational graphs
- Same solutions: truncation, simplified gradients

This isn't coincidence—it's the same mathematical problem!

---

## Experimental Results

### MAML Performance

Trained for 10,000 iterations:

- Meta-loss decreased from ~5.0 to ~0.6
- New sine waves adapted in 5 steps from 10 examples
- Strong generalization to unseen amplitudes/phases

### FOMAML vs MAML Comparison

With 5-10 inner steps:

- **FOMAML:** Comparable performance, 3-5x faster, constant memory
- **Full MAML:** Slightly better gradients, but rarely worth the cost

With 50-100 inner steps:

- **FOMAML:** Still works (with overfitting issues)
- **Full MAML:** Memory prohibitive, gradient degradation

**Practical Recommendation:** Use FOMAML unless you have specific reasons to need exact second-order gradients.

---

### Key Functions

```python
# Core MAML components
inner_loop(model, x_support, y_support, inner_lr, inner_steps)
outer_loop_loss(model, adapted_params, x_query, y_query)
functional_forward(model, x, params)  # Forward with explicit params

# Training
train_maml(num_iterations, num_tasks_per_batch, inner_lr, outer_lr)
train_fomaml(...)  # Same signature, different inner_loop

# Evaluation
test_adaptation(model, new_task)
```

---

## Mathematical Deep Dive

### Why Second-Order Derivatives?

Consider a single inner step:

```
θ₁ = θ₀ - α∇L_train(θ₀)
```

The query loss depends on θ₁:

```
L_query(θ₁) = L_query(θ₀ - α∇L_train(θ₀))
```

To get ∂L_query/∂θ₀:

```
∂L_query/∂θ₀ = ∂L_query/∂θ₁ × ∂θ₁/∂θ₀
             = ∂L_query/∂θ₁ × [I - α∇²L_train(θ₀)]
                                    ↑
                              Hessian (second-order)
```

With K steps, this compounds into a product of Jacobians involving Hessians.

### Implicit Differentiation Alternative

Instead of unrolling, assume convergence:

```
If θ* = argmin L_train(θ), then ∇L_train(θ*) = 0
```

Using implicit function theorem:

```
∂θ*/∂φ = -[∇²L_train(θ*)]⁻¹ × ∇_φ∇L_train(θ*)
```

This avoids unrolling but requires:

- Solving a linear system (expensive for large models)
- Assumption of convergence (may not hold with few steps)

---
