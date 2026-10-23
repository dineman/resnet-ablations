"""Gradient flow and signal propagation at initialization: plain vs residual, depth 20 vs 56, with/without BN.

    python -m scripts.grad_flow          # ~1 min on CPU, no GPU needed

For each network we feed one batch of 128 real CIFAR training images through the
freshly initialized network, backpropagate the cross-entropy loss, and record
for every residual block:
  - the std of its output activations (forward signal),
  - the norm of the gradient w.r.t. its first conv's weights (backward signal).
Repeated over seeds 0-2 (different inits). The batch is fixed.

What the theory predicts (NOTES.md sections 2-3):
  - Residual + BN: x_{l+1} = x_l + F(x_l), so dL/dx_l = dL/dx_L * prod(I + dF/dx). The "I" term gives
    the gradient a direct path to every block: no exponential decay with depth.
  - Plain + BN: forward activations stay normalized, but the gradient must go through every layer's
    Jacobian; nothing guarantees their product stays near norm 1.
  - Residual without BN: each block ADDS a He-initialized branch of similar variance, so the activation
    variance roughly doubles per block (He init keeps Var(F(x)) ~ Var(x)). Over 9 blocks that is a
    2^9 = 512x variance growth, and the logits explode. This is why lr 0.1 may be too large without BN.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from resnet.data import load_splits
from resnet.models import CifarNet

OUT = Path("results/grad_flow")
VARIANTS = [  # (label, depth, residual, bn)
    ("resnet20", 20, True, True), ("plain20", 20, False, True),
    ("resnet56", 56, True, True), ("plain56", 56, False, True),
    ("resnet20_nobn", 20, True, False), ("plain20_nobn", 20, False, False),
]
SEEDS = (0, 1, 2)


def measure(depth: int, residual: bool, bn: bool, seed: int, x: torch.Tensor, y: torch.Tensor) -> dict:
    torch.manual_seed(seed)
    model = CifarNet(depth, residual, bn).train()  # train mode: BN uses batch statistics, as in step 1 of training
    acts = []
    hooks = [b.register_forward_hook(lambda m, i, o: acts.append(o.detach().std().item())) for b in model.blocks]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    for h in hooks:
        h.remove()
    grads = [b.conv1.weight.grad.norm().item() for b in model.blocks]
    return {"act_std": acts, "grad_norm": grads, "logit_std": logits.std().item(), "loss": loss.item(),
            "stem_grad": model.conv.weight.grad.norm().item()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    d = load_splits("cpu")
    x, y = d["train"].x[:128], d["train"].y[:128]
    res = {label: [measure(dep, r, bn, s, x, y) for s in SEEDS] for label, dep, r, bn in VARIANTS}
    (OUT / "results.json").write_text(json.dumps(res, indent=1))

    style = {"resnet20": ("C0", "-"), "plain20": ("C3", "-"), "resnet56": ("C0", "--"), "plain56": ("C3", "--"),
             "resnet20_nobn": ("C2", "-"), "plain20_nobn": ("C1", "-")}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for label, *_ in VARIANTS:
        c, ls = style[label]
        for k, key in enumerate(("act_std", "grad_norm")):
            arr = np.array([r[key] for r in res[label]])
            blocks = np.arange(1, arr.shape[1] + 1)
            ax[k].plot(blocks, arr.mean(0), c=c, ls=ls, label=label)
            ax[k].fill_between(blocks, arr.min(0), arr.max(0), color=c, alpha=0.15)
    ax[0].set(yscale="log", xlabel="block (input → output)", ylabel="std of block output",
              title="Forward: activation scale at init")
    ax[1].set(yscale="log", xlabel="block (input → output)", ylabel="‖∂L/∂W‖ of the block's first conv",
              title="Backward: gradient reaching each block at init")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "grad_flow.png", dpi=150)

    lines = ["| network | activation std, first → last block | ‖∂L/∂W‖, first → last block | ratio first/last grad | logit std | initial loss |",
             "|---|---|---|---|---|---|"]
    for label, *_ in VARIANTS:
        rs = res[label]
        a0, a1 = np.mean([r["act_std"][0] for r in rs]), np.mean([r["act_std"][-1] for r in rs])
        g0, g1 = np.mean([r["grad_norm"][0] for r in rs]), np.mean([r["grad_norm"][-1] for r in rs])
        lines.append(f"| {label} | {a0:.2g} → {a1:.2g} | {g0:.2g} → {g1:.2g} | {g0 / g1:.2g} | "
                     f"{np.mean([r['logit_std'] for r in rs]):.2g} | {np.mean([r['loss'] for r in rs]):.3g} |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
