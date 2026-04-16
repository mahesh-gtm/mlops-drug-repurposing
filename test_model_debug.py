import torch
import torch.nn.functional as F

# Simulate MSE loss on unbounded outputs vs. bounded targets
model_output = torch.tensor([0.68])  # Model is outputting ~0.68
target = torch.tensor([0.89])  # Target range [0.84, 0.95], avg ~0.89

loss = F.mse_loss(model_output, target)
print(f"MSE between {model_output.item():.4f} and {target.item():.4f}: {loss.item():.4f}")

# The problem: model outputs are unbounded, but targets are bounded [0.84, 0.95]
# When sigmoid applied in inference: sigmoid(0.68) ≈ 0.6647
import math
sigmoid_val = 1 / (1 + math.exp(-0.68))
print(f"sigmoid(0.68) = {sigmoid_val:.4f}")

# But the issue is we're training without sigmoid!
# The model learns to output ~0.68, but targets expect ~0.89
# This creates a mismatch

# Solution: Either (1) train with sigmoid, or (2) scale targets to match unbounded space
print("\nThe real issue:")
print(f"- Model outputs: Unbounded (trained via MSE on raw logits)")
print(f"- Targets: Bounded to [0.84, 0.95]")
print(f"- Inference: Applies sigmoid() to model outputs")
print(f"\nResult: Model outputs ~0.68, after sigmoid → ~0.665")
