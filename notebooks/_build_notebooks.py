"""Generate the two demo notebooks from source-of-truth cell content.

Run: `python _build_notebooks.py` from inside the notebooks/ directory.
Writes `01_damped_oscillator.ipynb` and `02_burgers.ipynb`.
"""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def build(path: Path, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.x"},
    }
    nbf.write(nb, str(path))
    print(f"wrote {path}")


# -----------------------------------------------------------------------------
# Notebook 1: Damped harmonic oscillator (forward + inverse PINN)
# -----------------------------------------------------------------------------

OSC_CELLS = [
    md(r"""
# Demo 1 — Physics-Informed Neural Network for a damped harmonic oscillator

**Goal of this notebook:** see, end to end, what makes a PINN different from a plain neural network. We'll do three things in sequence:

1. **Plain NN, sparse data.** Train an MLP only on a handful of noisy measurements. It will fit the measurements but do nonsense between them.
2. **PINN, same data, plus physics.** Add a *physics-residual loss* that penalizes violations of the ODE on a grid of collocation points. The network now produces a smooth, physical trajectory even where there's no data.
3. **Inverse PINN.** Pretend we don't know the damping coefficient `c`. Make `c` a trainable parameter and recover it from the same sparse data.

The system is the simplest interesting one for this — a damped harmonic oscillator:

$$ m\,\ddot{x}(t) + c\,\dot{x}(t) + k\,x(t) = 0, \qquad x(0)=x_0,\ \dot{x}(0)=v_0. $$

Pick `m=1, k=1, c=0.5, x_0=1, v_0=0`. That's an underdamped oscillator with an analytical solution we can compare against.

**Things to play with** (see the last cell): the noise level, number of data points, network width/depth, physics-loss weight, training length, and whether the inverse problem can recover `c` when the data is very sparse or noisy.
"""),
    code(r"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

device = "cpu"  # tiny problem, CPU is faster than GPU launch overhead

# MIT branding for plots
MIT_RED = "#A31F34"
MIT_GRAY = "#8A8B8C"
plt.rcParams.update({
    "figure.figsize": (9, 4),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})
"""),
    md(r"""
## Ground truth and a few noisy measurements

The underdamped analytical solution is

$$ x(t) = e^{-\zeta\omega_n t}\,\bigl(A\cos\omega_d t + B\sin\omega_d t\bigr), $$

with $\omega_n=\sqrt{k/m}$, $\zeta=c/(2\sqrt{km})$, $\omega_d=\omega_n\sqrt{1-\zeta^2}$. We use it only to make a "ground truth" curve and to sample a few noisy measurements. The PINN never sees this formula.
"""),
    code(r"""
# Physical parameters
m, k, c_true = 1.0, 1.0, 0.5
x0, v0 = 1.0, 0.0
T = 15.0  # total time window

def analytical(t):
    omega_n = np.sqrt(k / m)
    zeta = c_true / (2 * np.sqrt(k * m))
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    A = x0
    B = (v0 + zeta * omega_n * x0) / omega_d
    return np.exp(-zeta * omega_n * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))

# Dense grid for plotting the truth
t_dense = np.linspace(0, T, 400)
x_dense = analytical(t_dense)

# Sparse, noisy measurements — only in the first half of the window
N_DATA = 10
NOISE = 0.05
t_data = np.linspace(0, T / 2, N_DATA)
x_data = analytical(t_data) + NOISE * np.random.randn(N_DATA)

fig, ax = plt.subplots()
ax.plot(t_dense, x_dense, color=MIT_GRAY, lw=2, label="ground truth")
ax.scatter(t_data, x_data, color=MIT_RED, zorder=3, label=f"{N_DATA} noisy measurements")
ax.axvspan(T / 2, T, alpha=0.07, color=MIT_GRAY, label="no data here")
ax.set_xlabel("time t"); ax.set_ylabel("position x(t)")
ax.set_title("Damped oscillator: sparse data, second half unobserved")
ax.legend()
plt.show()
"""),
    md(r"""
## Setting up the MLP for a physical problem

Before any code, a checklist of design choices — these are decisions you'll make every time you build a PINN for a real system. Most of them have one "obviously right" answer for ODE/PDE problems, but it's worth seeing each one called out so you know what you're committing to.

| Design choice | What we pick | Why |
|---|---|---|
| **Input** | scalar `t` (shape `[N, 1]`) | The independent variable of the ODE. For PDEs you'd add `x` and feed `[t, x]`. |
| **Output** | scalar `u` (shape `[N, 1]`) | The dependent variable we're solving for. |
| **Width / depth** | 48 units × 3 layers (~5k params) | Tiny by modern ML standards. Physics constrains the function class a lot — you almost never need a big network for an ODE. Rule of thumb: start small, grow only if loss plateaus above the noise floor. |
| **Activation** | `tanh` | Smooth, all derivatives exist, non-trivial $\ddot{f}$. ReLU is forbidden — its second derivative is zero almost everywhere, so the physics residual computes to zero for the wrong reason. `sin`, `GELU`, `Swish` all work; tanh is the default. |
| **Input scaling** | $t \to 2t/T - 1 \in [-1, 1]$ | <strong>The most important and most-forgotten step.</strong> Tanh saturates outside $[-3, 3]$. If you feed raw $t \in [0, 15]$, gradients vanish at the far end of the window, the physics residual is computed by a network that can't respond to its input, and PINN convergence stalls. Always normalize. |
| **Init** | PyTorch default (Kaiming-uniform) | Generally fine. Some PINN papers use Xavier-normal; the difference is usually within the seed-to-seed variance. |
| **Optimizer** | Adam, `lr=2e-3` | Robust default. Raissi's original PINNs polish with L-BFGS at the end; we skip that for clarity. |
| **Loss weights** | $\lambda_{\rm phys}=5$, $\lambda_{\rm ic}=20$, $\lambda_{\rm data}=1$ | IC weight needs to dominate or the network finds a low-physics-residual but wrong trajectory (often $u \equiv 0$). Physics weight is bumped above the data weight because the data-free half of the window has only physics to constrain it. |
| **Collocation points** | 200, uniformly across $[0, T]$ | Free — no measurements needed, just locations where the ODE should hold. More is fine; we use 200 because the residual is smooth and 200 is plenty. |
"""),
    md(r"""
## Attempt 1 — plain MLP, fit the data only

To set a baseline, we first train without any physics — just minimize the mean-squared error against the 10 noisy points. Same MLP class, no physics-residual term. Watch what happens outside the data region.
"""),
    code(r"""
class MLP(nn.Module):
    '''Feedforward network with input normalization baked in.

    Two non-obvious things going on:
      1. The input `t` (raw seconds) is mapped to [-1, 1] in `forward` before
         touching any Linear layer. Without this, tanh saturates on large t
         and physics gradients vanish.
      2. Activation is tanh, NOT ReLU, because PINNs differentiate the network
         output up to second order. ReLU's second derivative is zero almost
         everywhere, which would make any physics residual artificially zero.
    '''
    def __init__(self, width: int = 48, depth: int = 3, t_max: float = T):
        super().__init__()
        self.t_max = t_max  # used to normalize input to [-1, 1]
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        # Normalize t from [0, t_max] to [-1, 1] so tanh stays in its
        # high-gradient regime everywhere in the window.
        t_norm = 2.0 * t / self.t_max - 1.0
        return self.net(t_norm)


def train_data_only(model, t_data, x_data, n_steps=4000, lr=2e-3):
    t = torch.tensor(t_data, dtype=torch.float32, device=device).view(-1, 1)
    x = torch.tensor(x_data, dtype=torch.float32, device=device).view(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(n_steps):
        opt.zero_grad()
        loss = ((model(t) - x) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


model_data = MLP().to(device)
losses_data = train_data_only(model_data, t_data, x_data)

with torch.no_grad():
    t_eval = torch.tensor(t_dense, dtype=torch.float32, device=device).view(-1, 1)
    x_pred = model_data(t_eval).cpu().numpy().ravel()

fig, ax = plt.subplots()
ax.plot(t_dense, x_dense, color=MIT_GRAY, lw=2, label="ground truth")
ax.plot(t_dense, x_pred, color=MIT_RED, lw=2, label="plain MLP (data only)")
ax.scatter(t_data, x_data, color="black", zorder=3, s=25, label="measurements")
ax.axvspan(T / 2, T, alpha=0.07, color=MIT_GRAY)
ax.set_xlabel("time t"); ax.set_ylabel("position x(t)")
ax.set_title("Plain NN: fits where it has data, drifts where it doesn't")
ax.legend()
plt.show()
"""),
    md(r"""
**What you should see:** the plain MLP nails the 10 measurement points (or overshoots them when noise is high) but in the right half of the time window — where there's no data — it has no reason to follow the physics, so it drifts off into whatever happens to minimize the loss. This is the standard failure mode of pure data-driven ML when data is sparse.
"""),
    md(r"""
## Attempt 2 — add a physics-residual loss (a PINN)

The idea behind a PINN is one line: take the network's output $u_\theta(t)$, differentiate it with autograd to get $\dot{u}_\theta$ and $\ddot{u}_\theta$, and penalize the ODE residual

$$ r(t) = m\,\ddot{u}_\theta(t) + c\,\dot{u}_\theta(t) + k\,u_\theta(t) $$

at a grid of *collocation points* (cheap — no measurements needed, just points in time we want the physics to hold). We also pin the initial conditions and keep the small data-loss term.

Total loss:

$$ \mathcal{L} = \lambda_{\rm phys}\,\overline{r(t_i)^2} \;+\; \lambda_{\rm ic}\,\bigl[(u_\theta(0)-x_0)^2 + (\dot u_\theta(0)-v_0)^2\bigr] \;+\; \lambda_{\rm data}\,\overline{(u_\theta(t_j) - x_j)^2}. $$

The relative weights $\lambda$ are knobs — try changing them in the last section.
"""),
    code(r"""
def derivative(y, t):
    '''First derivative dy/dt via autograd.

    `create_graph=True` lets us differentiate again (we need the second derivative).
    `grad_outputs=ones` is the standard recipe for elementwise vector-Jacobian products
    when y has the same shape as t.
    '''
    return torch.autograd.grad(
        y, t,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]


def train_pinn(model, t_data, x_data, c_value=c_true,
               n_steps=12000, lr=2e-3,
               w_phys=5.0, w_ic=20.0, w_data=1.0,
               n_collocation=200):
    t_d = torch.tensor(t_data, dtype=torch.float32, device=device).view(-1, 1)
    x_d = torch.tensor(x_data, dtype=torch.float32, device=device).view(-1, 1)
    # Collocation points span the FULL time window — including the data-free region
    t_c = torch.linspace(0, T, n_collocation, device=device).view(-1, 1).requires_grad_(True)
    t0 = torch.zeros(1, 1, device=device, requires_grad=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = {"total": [], "phys": [], "ic": [], "data": []}

    for step in range(n_steps):
        opt.zero_grad()

        # Physics residual on collocation points
        u_c = model(t_c)
        u_t = derivative(u_c, t_c)
        u_tt = derivative(u_t, t_c)
        residual = m * u_tt + c_value * u_t + k * u_c
        loss_phys = (residual ** 2).mean()

        # Initial conditions
        u0 = model(t0)
        u_t0 = derivative(u0, t0)
        loss_ic = (u0 - x0).pow(2).mean() + (u_t0 - v0).pow(2).mean()

        # Data fit
        loss_data = ((model(t_d) - x_d) ** 2).mean()

        loss = w_phys * loss_phys + w_ic * loss_ic + w_data * loss_data
        loss.backward()
        opt.step()

        losses["total"].append(loss.item())
        losses["phys"].append(loss_phys.item())
        losses["ic"].append(loss_ic.item())
        losses["data"].append(loss_data.item())

    return losses


model_pinn = MLP().to(device)
losses_pinn = train_pinn(model_pinn, t_data, x_data)

with torch.no_grad():
    x_pred_pinn = model_pinn(t_eval).cpu().numpy().ravel()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(t_dense, x_dense, color=MIT_GRAY, lw=2, label="ground truth")
ax1.plot(t_dense, x_pred, color="#bbbbbb", lw=1.5, ls="--", label="plain MLP")
ax1.plot(t_dense, x_pred_pinn, color=MIT_RED, lw=2, label="PINN (physics + data)")
ax1.scatter(t_data, x_data, color="black", zorder=3, s=25, label="measurements")
ax1.axvspan(T / 2, T, alpha=0.07, color=MIT_GRAY)
ax1.set_xlabel("time t"); ax1.set_ylabel("x(t)")
ax1.set_title("Same data — physics fills in the second half")
ax1.legend(loc="upper right", fontsize=8)

ax2.semilogy(losses_pinn["total"], color="black", label="total")
ax2.semilogy(losses_pinn["phys"], color=MIT_RED, label="physics residual")
ax2.semilogy(losses_pinn["ic"], color=MIT_GRAY, label="initial conditions")
ax2.semilogy(losses_pinn["data"], color="tab:blue", label="data fit")
ax2.set_xlabel("step"); ax2.set_ylabel("loss (log)")
ax2.set_title("PINN loss components")
ax2.legend()
plt.tight_layout(); plt.show()
"""),
    md(r"""
**What you should see:** the PINN curve follows the true oscillation through the entire window, including where there's no data. The physics-residual loss doesn't need measurements — it just needs *time points* where the ODE should hold. That's the whole trick.

A few intuitions worth pausing on:

- **Collocation points are free.** They're not data; they're just locations where we ask the network to obey the equation. We can put 200, 2000, 20000 — only compute cost grows.
- **Autodiff does the differentiation symbolically-ish.** We never write a finite-difference formula. `torch.autograd.grad` walks the computation graph and returns exact derivatives at machine precision.
- **The activation matters.** Tanh is smooth; its derivatives exist and are non-trivial. ReLU's second derivative is zero almost everywhere — a ReLU PINN would have a zero physics residual for the wrong reason.
"""),
    md(r"""
## Attempt 3 — Inverse problem: discover the damping coefficient `c`

Now we *don't* know `c`. We have the same 10 noisy measurements and the same ODE form. We let `c` be a trainable parameter alongside the network weights and minimize the same loss.

This is the killer use case in industrial settings: you have physics that's *structurally* right but with one or two unknown coefficients that drift across batches, materials, or operating points. Fit them from telemetry.
"""),
    code(r"""
class InversePINN(nn.Module):
    '''Same MLP (with input normalization) but with a learnable damping `c`.

    The only structural difference from the forward PINN is `self.c` — a
    single nn.Parameter that the optimizer will update alongside the network
    weights. This is the "make it learnable" idiom in PyTorch.
    '''
    def __init__(self, c_init: float = 2.0, width: int = 48, depth: int = 3, t_max: float = T):
        super().__init__()
        self.mlp = MLP(width=width, depth=depth, t_max=t_max)
        # Start far from the true value (0.5) so we can see it move
        self.c = nn.Parameter(torch.tensor(c_init, dtype=torch.float32))

    def forward(self, t):
        return self.mlp(t)


def train_inverse(model, t_data, x_data,
                  n_steps=12000, lr=2e-3,
                  w_phys=5.0, w_ic=20.0, w_data=10.0,
                  n_collocation=200):
    t_d = torch.tensor(t_data, dtype=torch.float32, device=device).view(-1, 1)
    x_d = torch.tensor(x_data, dtype=torch.float32, device=device).view(-1, 1)
    t_c = torch.linspace(0, T, n_collocation, device=device).view(-1, 1).requires_grad_(True)
    t0 = torch.zeros(1, 1, device=device, requires_grad=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    c_history = []
    loss_history = []

    for step in range(n_steps):
        opt.zero_grad()

        u_c = model(t_c)
        u_t = derivative(u_c, t_c)
        u_tt = derivative(u_t, t_c)
        residual = m * u_tt + model.c * u_t + k * u_c
        loss_phys = (residual ** 2).mean()

        u0 = model(t0)
        u_t0 = derivative(u0, t0)
        loss_ic = (u0 - x0).pow(2).mean() + (u_t0 - v0).pow(2).mean()

        loss_data = ((model(t_d) - x_d) ** 2).mean()

        loss = w_phys * loss_phys + w_ic * loss_ic + w_data * loss_data
        loss.backward()
        opt.step()

        c_history.append(model.c.item())
        loss_history.append(loss.item())

    return c_history, loss_history


model_inv = InversePINN(c_init=2.0).to(device)
c_history, loss_history = train_inverse(model_inv, t_data, x_data)

print(f"true c       = {c_true:.4f}")
print(f"recovered c  = {model_inv.c.item():.4f}")
print(f"absolute err = {abs(model_inv.c.item() - c_true):.4f}")

with torch.no_grad():
    x_pred_inv = model_inv(t_eval).cpu().numpy().ravel()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(t_dense, x_dense, color=MIT_GRAY, lw=2, label="ground truth")
ax1.plot(t_dense, x_pred_inv, color=MIT_RED, lw=2, label=f"inverse PINN, learned c={model_inv.c.item():.3f}")
ax1.scatter(t_data, x_data, color="black", zorder=3, s=25, label="measurements")
ax1.set_xlabel("time t"); ax1.set_ylabel("x(t)")
ax1.set_title("Trajectory after learning c from sparse data")
ax1.legend()

ax2.plot(c_history, color=MIT_RED, lw=2)
ax2.axhline(c_true, color=MIT_GRAY, ls="--", label=f"true c = {c_true}")
ax2.set_xlabel("training step"); ax2.set_ylabel("c estimate")
ax2.set_title("Damping coefficient converging to truth")
ax2.legend()
plt.tight_layout(); plt.show()
"""),
    md(r"""
## Things to play with

Each of these breaks something interesting:

- **`NOISE`** (top): crank to 0.2. The inverse fit gets biased; with enough noise, you can't recover `c` accurately from 10 points.
- **`N_DATA`** (top): drop to 3. PINN still works because physics constrains the shape; plain NN can't even fit.
- **`w_phys`** (PINN training): set to 0. You've reduced the PINN to a plain NN. Set to 100. The fit gets *too* stiff and ignores noisy data.
- **`w_ic`** (PINN training): set to 0. The PINN drifts because nothing pins down the starting point.
- **Move the data region** to `[T/2, T]` instead of `[0, T/2]`. The PINN extrapolates *backwards*. Plain NN extrapolates nowhere useful.
- **Inverse `c_init`**: start at 10. The optimizer can get stuck — non-convex landscape. Try a few initializations.
- **Activation:** in `MLP`, swap `nn.Tanh()` for `nn.ReLU()`. The physics residual goes to zero for the wrong reason and the network fits arbitrary nonsense.

The sidebar coming up (Section 5 in the slides) is what happens when you push this idea past one scalar `c` to a whole *function* `h_e(T_gas, v_gas, T_drum)` that lives inside a multi-state thermal ODE. That's [my paper](../main.pdf).
"""),
]


# -----------------------------------------------------------------------------
# Notebook 2: Burgers' equation (classical Raissi-2019 reproduction, forward)
# -----------------------------------------------------------------------------

BURGERS_CELLS = [
    md(r"""
# Demo 2 — PINN for the 1D viscous Burgers' equation

This is the canonical demo from Raissi, Perdikaris & Karniadakis (2019) — the paper that put "PINN" on the map. We're going to solve

$$ u_t + u\,u_x - \nu\,u_{xx} = 0, \qquad x \in [-1, 1],\ t \in [0, 1], $$

with initial condition $u(0, x) = -\sin(\pi x)$ and Dirichlet boundary conditions $u(t, \pm 1) = 0$. We use $\nu = 0.01/\pi$, which is small enough that the smooth sinusoidal initial condition develops a near-shock around $x=0$ by $t \approx 0.4$.

**What's different from Notebook 1:** the input is now a 2D point $(t, x)$ instead of a scalar $t$. The physics residual involves spatial derivatives. Otherwise the recipe is identical: MLP output, autodiff for derivatives, residual + boundary + initial losses on collocation points.

This is a **forward** problem — no measurements, just the equation and its boundary/initial data. The PINN replaces a traditional finite-difference / finite-element solver.
"""),
    code(r"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

device = "cpu"

MIT_RED = "#A31F34"
MIT_GRAY = "#8A8B8C"
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Problem constants
NU = 0.01 / np.pi
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0
"""),
    md(r"""
## Sampling the three loss regions

A PINN for a PDE on a 2D domain needs three sample sets:

1. **Initial condition** — points along $t = 0$, where we know $u = -\sin(\pi x)$.
2. **Boundary condition** — points along $x = \pm 1$, where $u = 0$.
3. **Collocation interior** — random $(t, x)$ points inside the domain where the PDE residual should vanish.

Latin-hypercube or random uniform sampling both work. We'll just use uniform random — simpler, plenty for this problem.
"""),
    code(r"""
N_IC = 100      # points on t=0
N_BC = 100      # points on each spatial boundary
N_COLLOCATION = 10000  # interior physics points

# Initial condition: t=0, x ~ uniform[-1, 1], u = -sin(pi x)
x_ic = torch.rand(N_IC, 1, device=device) * (X_MAX - X_MIN) + X_MIN
t_ic = torch.zeros_like(x_ic)
u_ic = -torch.sin(np.pi * x_ic)

# Boundary conditions: x=+/-1, t ~ uniform[0, 1], u = 0
t_bc = torch.rand(2 * N_BC, 1, device=device) * (T_MAX - T_MIN) + T_MIN
x_bc = torch.cat([
    torch.full((N_BC, 1), X_MIN, device=device),
    torch.full((N_BC, 1), X_MAX, device=device),
])
u_bc = torch.zeros_like(t_bc)

def sample_collocation(n: int):
    '''Fresh (t, x) collocation points, as leaf tensors with requires_grad=True.

    We resample every training step. This (a) keeps the autograd graph fresh —
    if you reuse the same tensors across .backward() calls you'll see
    "trying to backward through the graph a second time" errors — and
    (b) acts like stochastic mini-batching over the physics-loss domain,
    which empirically helps PINN convergence.
    '''
    t = torch.empty(n, 1, device=device).uniform_(T_MIN, T_MAX).requires_grad_(True)
    x = torch.empty(n, 1, device=device).uniform_(X_MIN, X_MAX).requires_grad_(True)
    return t, x

# Visualize a sample
_t_show, _x_show = sample_collocation(N_COLLOCATION)
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(_t_show.detach(), _x_show.detach(), s=2, color=MIT_GRAY, alpha=0.3, label="physics collocation")
ax.scatter(t_ic, x_ic, s=12, color=MIT_RED, label="initial condition (t=0)")
ax.scatter(t_bc, x_bc, s=12, color="black", label="boundary (x=±1)")
ax.set_xlabel("t"); ax.set_ylabel("x")
ax.set_title("Where the three loss terms live in (t, x)")
ax.legend(loc="lower right")
plt.show()
"""),
    md(r"""
## The PINN

A 2-input MLP (`t, x → u`). The physics residual involves first time derivative, first and second spatial derivatives — all from autograd.
"""),
    code(r"""
class PINN(nn.Module):
    def __init__(self, width=40, depth=8):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t, x):
        return self.net(torch.cat([t, x], dim=1))


def grad(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


def physics_residual(model, t, x):
    '''Burgers' residual: u_t + u u_x - nu u_xx.'''
    u = model(t, x)
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)
    return u_t + u * u_x - NU * u_xx


model = PINN().to(device)
print(f"trainable parameters: {sum(p.numel() for p in model.parameters())}")
"""),
    md(r"""
## Training

Adam for a few thousand steps. The Raissi paper uses L-BFGS as a polish — we'll skip that for clarity; Adam alone gets us a clean solution in well under a minute on CPU.

Weights matter: the initial-condition loss needs to be pulled up (here `w_ic=10`) so the network actually anchors at the sinusoidal start rather than minimizing the physics residual with a trivial solution.
"""),
    code(r"""
N_STEPS = 5000
LR = 1e-3
W_PHYS, W_IC, W_BC = 1.0, 10.0, 10.0

opt = torch.optim.Adam(model.parameters(), lr=LR)
history = {"total": [], "phys": [], "ic": [], "bc": []}

for step in range(N_STEPS):
    opt.zero_grad()

    t_co, x_co = sample_collocation(N_COLLOCATION)
    r = physics_residual(model, t_co, x_co)
    loss_phys = (r ** 2).mean()

    u_pred_ic = model(t_ic, x_ic)
    loss_ic = ((u_pred_ic - u_ic) ** 2).mean()

    u_pred_bc = model(t_bc, x_bc)
    loss_bc = ((u_pred_bc - u_bc) ** 2).mean()

    loss = W_PHYS * loss_phys + W_IC * loss_ic + W_BC * loss_bc
    loss.backward()
    opt.step()

    history["total"].append(loss.item())
    history["phys"].append(loss_phys.item())
    history["ic"].append(loss_ic.item())
    history["bc"].append(loss_bc.item())

    if (step + 1) % 1000 == 0:
        print(f"step {step+1:5d}  total={loss.item():.4e}  "
              f"phys={loss_phys.item():.4e}  ic={loss_ic.item():.4e}  bc={loss_bc.item():.4e}")
"""),
    md(r"""
## Visualize the learned solution

We sample the trained network on a regular (t, x) grid and plot:

1. A heatmap of $u(t, x)$ over the full domain.
2. Snapshots of $u(t, \cdot)$ at a few times so you can see the sinusoid steepening into a near-shock around $x = 0$.
"""),
    code(r"""
N = 200
t_plot = torch.linspace(T_MIN, T_MAX, N, device=device)
x_plot = torch.linspace(X_MIN, X_MAX, N, device=device)
T_grid, X_grid = torch.meshgrid(t_plot, x_plot, indexing="ij")
with torch.no_grad():
    U = model(T_grid.reshape(-1, 1), X_grid.reshape(-1, 1)).reshape(N, N).cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

im = axes[0].imshow(
    U.T,
    origin="lower",
    extent=[T_MIN, T_MAX, X_MIN, X_MAX],
    aspect="auto",
    cmap="RdBu_r",
    vmin=-1, vmax=1,
)
axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
axes[0].set_title("PINN solution u(t, x)")
plt.colorbar(im, ax=axes[0], label="u")

snapshot_times = [0.0, 0.25, 0.5, 0.75, 1.0]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(snapshot_times)))
for c, ts in zip(colors, snapshot_times):
    with torch.no_grad():
        t_s = torch.full((N, 1), ts, device=device)
        x_s = x_plot.view(-1, 1)
        u_s = model(t_s, x_s).cpu().numpy().ravel()
    axes[1].plot(x_plot.cpu().numpy(), u_s, color=c, lw=2, label=f"t = {ts:.2f}")
axes[1].axhline(0, color="black", lw=0.5)
axes[1].set_xlabel("x"); axes[1].set_ylabel("u")
axes[1].set_title("Snapshots: smooth sinusoid sharpens into a near-shock")
axes[1].legend(loc="lower left", fontsize=8)
axes[1].grid(alpha=0.25)
plt.tight_layout(); plt.show()
"""),
    code(r"""
# Loss history
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(history["total"], color="black", label="total")
ax.semilogy(history["phys"], color=MIT_RED, label="physics residual")
ax.semilogy(history["ic"], color=MIT_GRAY, label="initial condition")
ax.semilogy(history["bc"], color="tab:blue", label="boundary")
ax.set_xlabel("step"); ax.set_ylabel("loss (log)")
ax.set_title("Loss components")
ax.grid(alpha=0.25)
ax.legend()
plt.show()
"""),
    md(r"""
## Things to play with

- **`NU`** at top: try `0.1/np.pi` (smoother) and `0.001/np.pi` (sharper shock that the PINN may struggle to resolve without more collocation points or a wider network).
- **`N_COLLOCATION`**: drop to 500. The shock region gets noisy — the PINN doesn't have enough physics samples to constrain the gradient there.
- **`W_IC` / `W_BC`**: set to 1.0 (no upweighting). The network often collapses to the trivial $u \equiv 0$ because the IC term gets dominated by the physics residual.
- **Width / depth** in `PINN.__init__`: shrink to width=16, depth=3. Watch the solution get visibly worse. Grow to width=80, depth=10 and see if you can tighten the shock.
- **Replace Adam with L-BFGS** for the last 1000 steps. Raissi's paper does this; it gives a sharper shock at the cost of more code.
- **Add data.** Sample 20 points from a finite-difference reference, add a `loss_data` term, and watch the PINN tighten where it agrees with the reference.

## What this notebook is *not*

This is a forward solve — we used the PINN as a mesh-free PDE solver. Modern numerical methods will still beat a PINN at this problem on accuracy and wall-clock time. The PINN really shines on:

- **Inverse problems** (identify unknown coefficients from data — Notebook 1 covered this for an ODE).
- **High-dimensional PDEs** where meshing is prohibitive.
- **Embedding partial physics into a data-driven model** when only part of the equation is known.

The third one is where [my paper](../main.pdf) lives — and the next slide picks it up.
"""),
]


def main() -> None:
    here = Path(__file__).parent
    build(here / "01_damped_oscillator.ipynb", OSC_CELLS)
    build(here / "02_burgers.ipynb", BURGERS_CELLS)


if __name__ == "__main__":
    main()
