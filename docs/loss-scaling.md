# Loss scaling: which weight to implement

Choice 12 in `train/bridge_prototype.py` is open: we train on a plain `MSELoss` against the drift
target, the DSBM reference multiplies both sides by a `t`-dependent factor, and the DSBM paper's
Appendix H derives a *third*, different factor. This note works out which one to implement by
going back to the two works the paper cites for it — Ho et al. (2020) and Song et al. (2021b) —
and then checks whether `reference/DDSBM` has anything to say.

**Conclusion up front: implement the reference code's weight, `σ²(1-t)` forward / `σ²t` backward.**
It is not an implementation shortcut — it is the standard *variance weighting* that both cited works
use, it is what produced the reported AFHQ samples, and our current plain MSE turns out to be the
*likelihood weighting*, which the literature specifically associates with worse FID.

---

## 1. What actually diverges

The forward regression target is the Markovian drift of the Brownian bridge,

```
target = (X₁ - X_t) / (1 - t)      with   X_t = tX₁ + (1-t)X₀ + σ√(t(1-t))·Z
```

Substituting `X_t` and simplifying (this is the line Appendix H opens with):

```
target = (X₁ - X₀) - σ√(t/(1-t))·Z
```

So the target is a bounded *signal* term plus a noise term whose per-coordinate variance is
`σ² t/(1-t)`, which diverges as `t → 1`. At `σ² = 5` and `t = 0.99` that variance is 495 — three
orders of magnitude above the signal — and the network is being asked to predict it from `X_t`,
which contains almost no information about it. The backward loss has the mirror problem at `t → 0`.

Every candidate below is a `λ(t)` multiplying the squared residual to stop that tail from
dominating the gradient.

## 2. Ho et al. (2020) — the weighting is hidden in the parameterization

DDPM's variational bound term (Eq. 12) carries an explicit prefactor:

```
L_{t-1} = E[ β_t² / (2σ_t² α_t (1-ᾱ_t)) · ‖ε - ε_θ(√ᾱ_t x₀ + √(1-ᾱ_t) ε, t)‖² ]
```

and §3.4 defines `L_simple` (Eq. 14) as **the same expression with that prefactor dropped**. The
code matches: in [`diffusion_utils_2.py`](https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py),
`training_losses` is

```python
target = {'xprev': ..., 'xstart': x_start, 'eps': noise}[self.model_mean_type]
model_output = denoise_fn(x_t, t)
losses = nn.meanflat(tf.squared_difference(target, model_output))
```

No `t`-dependent factor anywhere. The reweighting is entirely in the *choice of target*: regressing
on `ε` rather than on the mean is algebraically the same as regressing on the mean with a specific
`t`-dependent weight. Ho et al. are explicit that this is deliberate and about which `t` the network
should spend capacity on:

> these terms train the network to denoise data with very small amounts of noise, so it is beneficial
> to down-weight them so that the network can focus on more difficult denoising tasks at larger `t`

**Transferable lesson:** an "unweighted MSE" is only unweighted relative to some parameterization.
The question is never *whether* to weight, only which residual you call the unweighted one.

## 3. Song et al. (2021b) — the two weightings have names

[`score_sde_pytorch/losses.py`](https://github.com/yang-song/score_sde_pytorch/blob/main/losses.py)
makes the same point but exposes both options as a flag:

```python
# likelihood_weighting = False
losses = torch.square(score * std[:, None, None, None] + z)

# likelihood_weighting = True
g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
losses = torch.square(score + z / std[:, None, None, None]) * g2
```

The true score of the perturbation kernel is `-z/std`, so the first line is
`std² · ‖score_θ - score_true‖²` — **the score residual multiplied by the noise std of the kernel
being matched**, i.e. `λ(t) = σ_t²`, called *variance weighting*. The second is
`λ(t) = g(t)²`, *likelihood weighting*, which makes the objective a strict upper bound on the NLL.

The trade-off is documented: likelihood weighting increases the variance of the training objective
and improves likelihood at some cost to FID; variance weighting is the one that emphasizes FID. The
image configs in that repo use the FID-oriented setting —
[`default_cifar10_configs.py`](https://github.com/yang-song/score_sde_pytorch/blob/main/configs/default_cifar10_configs.py)
sets `training.likelihood_weighting = False`.

Note also that `score * std + z` is exactly Ho's `ε`-prediction written in score language. The two
cited works agree; Song et al. just name the alternative.

## 4. Mapping DSBM onto that dichotomy

DSBM's reference measure is Brownian motion with a time-homogeneous `σ` and `T = 1` (Appendix I),
so `g(t) = σ` is **constant**. The network predicts `v_θ ≈ σ²∇log Q_{T|t}(X_T|X_t)`, and
`Q_{T|t} = N(X_t, σ²(1-t)I)`, whose std is `σ√(1-t)`. That is precisely the quantity
`apply_net` computes (`trainer_dbdsb.py:688-698`, via `marginal_prob(None, t, 'b')`).

| | `λ(t)` on the drift residual | in Song et al.'s terms |
|---|---|---|
| our plain MSE | `const` | `∝ g(t)² = σ²` → **likelihood weighting** |
| reference `loss_scale: True` | `σ²(1-t)` fwd, `σ²t` bwd | `= σ_t²` of `Q_{T\|t}` → **variance weighting** |
| paper Appendix H | `(1-t)/((1-t) + σ²t)` | neither; see below |

The first row is the load-bearing finding. Because `g(t)` is constant here, our plain MSE is not
"no weighting" — it is *specifically* the likelihood-weighted objective, the one score_sde turns
**off** for its image experiments. We landed on the FID-unfavourable option by accident.

## 5. Why Appendix H differs from `loss_scale`

Appendix H downweights by `1 + σ²t/(1-t)`, i.e. `λ_H(t) = (1-t)/((1-t) + σ²t)`, and says why the
`1` is there: *"we can add 1 to effectively cause no loss scaling when t is close to 0"*. It
normalizes the *whole* target — signal plus noise — so the weighted loss tends to `E‖X₁-X₀‖²` at
`t → 0` and to unit variance per coordinate at `t → 1`. `loss_scale` normalizes only against the
kernel std, giving the same `O(1-t)` tail but no unit normalization at either end.

Both kill the divergence; they are not the same function. The ratio is `σ²(1 + (σ²-1)t)`, running
from `σ²` at `t=0` to `σ⁴` at `t=1` — a factor-of-`σ²` tilt across the interval. Normalized to 1 at
`t = 0`, at `σ² = 5`:

| `t` | Appendix H | `loss_scale` | plain MSE (ours) |
|---|---|---|---|
| 0.0 | 1.0 | 1.0 | 1.0 |
| 0.5 | 0.167 | 0.5 | 1.0 |
| 0.9 | 0.022 | 0.1 | 1.0 |
| 0.99 | 0.0020 | 0.01 | 1.0 |

Appendix H is markedly more aggressive through the mid and late range. This is recorded as mismatch
(f) in the `bridge_prototype.py` docstring.

## 6. `reference/DDSBM` — no guidance, and the reason is instructive

DDSBM ([Kim et al., arXiv:2410.01500](https://doi.org/10.48550/arXiv.2410.01500)) has no
`t`-dependent loss weighting at all. Its losses are categorical:

- `TrainLossDiscrete` (`src/ddsbm/metrics/train_metrics.py:94`) — `CrossEntropyMetric` on node and
  edge logits, via `F.cross_entropy(..., reduction="sum")` (`abstract_metrics.py:99-110`).
- `TrainLossDiscreteKL` (`train_metrics.py:180`) — `SumExceptBatchKL` between true and predicted
  categorical distributions.

The only weight is `lambda_train`, a fixed nodes-vs-edges-vs-`y` balance, not a function of `t`.

That absence is the point: the divergence in §1 is an artifact of the **continuous Gaussian bridge**,
whose conditional variance `σ²t/(1-t)` blows up at the endpoint. DDSBM's state space is discrete, the
transition kernels are categorical, and the loss is bounded by construction — there is nothing to
normalize. So DDSBM neither supports nor contradicts a choice here; it simply doesn't have the
problem. Do not read its unweighted CE as evidence that unweighted is fine for our setting.

## 7. Recommendation

Implement `loss_scale`, i.e. multiply both prediction and target by `σ√(1-t)` in local time, for
these reasons in order:

1. **It is what produced the reported numbers.** `loss_scale: True` is set in
   `conf/method/dbdsb.yaml`, which is the method config for the AFHQ run. Appendix H is the
   derivation the knob descends from, not the knob.
2. **It is the convention of both cited works.** Ho's `ε`-prediction and score_sde's
   `likelihood_weighting=False` are the same move — residual times kernel std.
3. **Our current default is actively the wrong one for FID.** Plain MSE ≡ likelihood weighting
   here, which score_sde disables for exactly the metric DSBM reports on AFHQ.

Two notes for whoever implements it:

- Our choice 15 (backward net reparameterized to `τ = 1-t`) makes this uniform rather than awkward.
  The reference's backward weight is `σ√t` in *global* time, and `t = 1-τ`, so both of our branches
  end up using `σ√(1-t)` in their own local time. No sign or direction juggling.
- Appendix H is worth keeping as a fallback if training is unstable at `σ² = 5`. It is strictly more
  aggressive in the tail, and it is derived rather than inherited. But it should be an experiment
  against the `loss_scale` baseline, not the first thing we run.

## Sources

- [Ho, Jain, Abbeel (2020), *Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239) — §3.4, Eqs. 12 and 14
- [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) — `diffusion_tf/diffusion_utils_2.py`, `training_losses`
- [Song et al. (2021b), *Score-Based Generative Modeling through SDEs*](https://arxiv.org/abs/2011.13456)
- [Song, Durkan, Murray, Ermon (2021), *Maximum Likelihood Training of Score-Based Diffusion Models*](https://papers.nips.cc/paper_files/paper/2021/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf) — where `λ(t) = g(t)²` is introduced and the likelihood/FID trade-off is measured
- [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) — `losses.py`, `configs/default_cifar10_configs.py`
- [Kim et al. (2024), *Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation*](https://doi.org/10.48550/arXiv.2410.01500) — vendored at `reference/DDSBM`
- Shi et al., *Diffusion Schrödinger Bridge Matching* — Appendix H (Loss Scaling), Appendix I (experiment setup)
