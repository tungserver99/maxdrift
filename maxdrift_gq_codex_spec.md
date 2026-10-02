# MaxDrift-GQ: Implementation and Experimental Specification

## 0. Purpose

Implement a **fingerprint-agnostic PTQ remover** for LLM fingerprints.

The final method is **MaxDrift-GQ**:

> Starting from a strong GuidedQuant + LNQ quantized solution, keep the GuidedQuant non-uniform codebook fixed and change only the discrete weight assignments so that the quantized weights move as far as possible from the fingerprinted FP16 model, while staying inside a GuidedQuant end-loss preservation budget.

This document is the authoritative implementation specification. Do **not** replace the method with rank-and-flip heuristics, manual bin moves, pseudo-trigger generation, mismatched-data fine-tuning, QAT, or fingerprint-aware optimization.

Reference implementation for GuidedQuant / LNQ:

- Paper: `https://proceedings.mlr.press/v267/kim25d.html`
- Official repo: `https://github.com/snu-mllab/GuidedQuant`
- Official scalar-quantization flow:
  - gradient extraction: `scripts/run_sqllm.sh`
  - Hessian collection + LNQ: `scripts/run_lnq.sh`
  - top-level LNQ-related code includes `layerwise_nuq.py`, `full_nuq.py`, and `any_precision/`

The implementation should reuse the official GuidedQuant statistics and LNQ code path rather than reimplementing GuidedQuant from scratch.

---

# 1. Research question

Given a fingerprinted model \(W\), can we find a low-bit quantized model \(Q\) such that:

1. benign language-model behavior is preserved, and
2. \(Q\) is much farther from \(W\) in parameter space than a normal utility-optimized PTQ solution,

and does this larger **function-preserving parameter drift** reduce fingerprint success across multiple fingerprinting methods?

The core hypothesis is:

> Fingerprint information may partially occupy parameter directions that are weakly constrained by benign end-loss. Standard PTQ avoids unnecessary movement in these directions. MaxDrift-GQ explicitly exploits the available benign-flat quantization degrees of freedom.

This is a hypothesis to test, not an assumption to bake into evaluation.

---

# 2. Hard constraints

The final method MUST satisfy all of the following.

## 2.1 Threat model

The remover receives:

- one fingerprinted LLM \(W\);
- benign calibration data;
- the desired bit width.

The remover does **not** receive:

- private fingerprint keys;
- trigger prompts;
- fingerprint responses;
- fingerprint training data;
- clean pre-fingerprint weights;
- knowledge of which fingerprint family was used.

The clean/base model may be used only for separate scientific diagnostics if explicitly requested later. It must not be used by MaxDrift-GQ.

## 2.2 PTQ only

Allowed:

- forward/backward passes on benign calibration data to collect GuidedQuant statistics;
- weight-only scalar quantization;
- discrete optimization of quantized code assignments.

Not allowed:

- QAT;
- fine-tuning;
- LoRA;
- SGD/Adam updates of floating-point model weights;
- training on fingerprint data;
- pseudo-key construction;
- mismatched-pair training.

## 2.3 No heuristic weight manipulation

Do not implement:

- sorting weights by sensitivity and flipping top-k;
- manually moving selected weights by \(+1/-1\) quantization bin;
- random weight flipping;
- manually choosing layers or weights based on fingerprint success;
- gradient-magnitude top-k selection;
- GEDQ-style rank-and-flip logic.

Every changed code must be selected by the stated constrained discrete optimization.

---

# 3. Terminology

For a linear layer:

\[
W \in \mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{out}}}
\]

is the original **fingerprinted FP16/BF16 weight matrix**.

For output channel \(j\):

\[
w_j = W_{:,j} \in \mathbb{R}^{d_{\mathrm{in}}}.
\]

The quantized vector is

\[
q_j \in \mathbb{R}^{d_{\mathrm{in}}}.
\]

The quantization error / displacement is

\[
e_j = q_j - w_j.
\]

GuidedQuant provides a positive-semidefinite grouped Hessian/Fisher approximation. For output channel \(j\), let the associated Hessian be

\[
H_{g(j)}^{GQ} \in \mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{in}}},
\]

where \(g(j)\) maps output channel \(j\) to its GuidedQuant Hessian group.

LNQ provides a non-uniform scalar codebook for each native LNQ quantization unit. Preserve the repository's native codebook granularity. Do not invent a different grouping scheme.

---

# 4. Preservation signal: GuidedQuant end-loss objective

Do **not** use raw \(X^\top X\) reconstruction as the main preservation metric.

Do **not** use KL divergence in the optimization.

Use the existing GuidedQuant end-loss-guided quadratic statistics.

GuidedQuant's layer objective is based on

\[
\left\|
\frac{\partial \ell}{\partial Z}
\odot
(XW-XQ)
\right\|_F^2,
\]

which corresponds to a grouped block-diagonal empirical-Fisher / end-loss quadratic approximation.

For one output channel \(j\), define the preservation cost

\[
P_j(q_j)
=
e_j^\top H_{g(j)}^{GQ} e_j.
\]

For a complete model,

\[
P(Q)
=
\sum_{l}\sum_j P_{l,j}(q_{l,j}).
\]

MaxDrift-GQ will not directly minimize this quantity. It will treat it as a **hard feasibility budget**.

---

# 5. Drift objective

For output channel \(j\), define

\[
D_j(q_j)
=
\|q_j-w_j\|_2^2
=
e_j^\top e_j.
\]

For the model,

\[
D(Q)
=
\|Q-W\|_F^2
=
\sum_l\sum_j D_{l,j}(q_{l,j}).
\]

The goal is to make this quantity as large as possible while remaining within the GuidedQuant preservation budget.

Also log normalized drift:

\[
D_{\mathrm{norm}}
=
\frac{\|Q-W\|_F}{\|W\|_F}.
\]

---

# 6. Final optimization problem

For each output channel \(j\), after obtaining the GQ-LNQ initialization \(q_j^{(0)}\), define

\[
P_j^{(0)}
=
P_j(q_j^{(0)}).
\]

For a budget multiplier \(\rho \ge 0\), define

\[
\epsilon_j
=
(1+\rho)P_j^{(0)}.
\]

Then solve

\[
\boxed{
\begin{aligned}
\max_{q_j}\quad&
D_j(q_j)
=
\|q_j-w_j\|_2^2 \\
\text{s.t.}\quad&
P_j(q_j)
=
(q_j-w_j)^\top H_{g(j)}^{GQ}(q_j-w_j)
\le \epsilon_j,\\
&q_{ij}\in \mathcal C_j
\quad\forall i.
\end{aligned}
}
\]

Here \(\mathcal C_j\) is the **fixed non-uniform LNQ codebook** for that output channel / native LNQ quantization unit.

The codebook is frozen during MaxDrift optimization.

### Why use per-output-channel budgets?

This is deliberate:

- it keeps the optimization separable across output channels;
- it prevents early channels/layers from consuming all global preservation slack;
- it retains the same within-output-channel interaction structure modeled by GuidedQuant;
- it allows efficient parallel implementation across output channels;
- summing the per-channel bounds also bounds total GuidedQuant preservation cost.

Do not silently switch to a global shared budget in the first implementation.

---

# 7. Stage A: collect GuidedQuant statistics

For every fingerprinted model independently:

1. Treat the **fingerprinted model itself** as the teacher/reference model \(W\).
2. Use only benign language-model calibration data.
3. Run the official GuidedQuant gradient extraction.
4. Run the official GuidedQuant Hessian/Fisher collection.
5. Cache all GuidedQuant statistics using the repository's existing cache format.

Important:

- Do not use the clean base model as teacher.
- Do not reuse Hessians from a different fingerprinted checkpoint in the main experiment.
- Fingerprint prompts must not be present in calibration.
- Fingerprint success must not be inspected during this stage.

Recommended initial calibration setup for the project:

- C4;
- 128 sequences;
- sequence length 2048;
- fixed random seed.

If the local GuidedQuant scripts use a different default calibration format, adapt them explicitly to this setup and record the exact sample IDs / seed.

The official repository currently uses:

- `scripts/run_sqllm.sh` for gradient extraction;
- `scripts/run_lnq.sh` for Hessian collection and LNQ.

Reuse those mechanisms where possible.

---

# 8. Stage B: obtain the GQ-LNQ initialization

Run standard **LNQ + GuidedQuant** for the fingerprinted model at the desired bit width.

Initial bit widths:

- INT3;
- INT4.

For each bit width save:

1. quantized checkpoint \(Q_0\);
2. non-uniform codebooks;
3. assignments / integer codes, if stored natively;
4. GuidedQuant Hessian-group mapping;
5. per-channel baseline preservation cost \(P_j^{(0)}\);
6. model-level preservation cost \(P(Q_0)\);
7. model-level weight drift \(D(Q_0)\).

This GQ-LNQ result serves two roles:

- strong PTQ baseline;
- initialization for MaxDrift-GQ.

## 8.1 Recovering assignments if the checkpoint does not store them explicitly

Prefer native saved LNQ assignments.

If only dequantized quantized weights and codebooks are available:

- recover each code index by exact / nearest matching to the corresponding frozen codebook;
- reconstruct \(Q_0\) from recovered codes;
- assert reconstruction error is below a strict tolerance.

Abort if the assignments/codebook cannot reproduce \(Q_0\) reliably.

Do not silently requantize \(W\) with a new codebook.

---

# 9. Stage C: MaxDrift-GQ optimizer

The optimizer is:

> **Exact constrained discrete cyclic coordinate ascent** on code assignments.

It is not the original LNQ closed-form assignment update, because the objective is different.

It does, however, reuse:

- the same weight orientation;
- the same fixed LNQ codebook;
- the same GuidedQuant Hessian;
- the same output-channel decomposition.

## 9.1 State for one output channel

For one channel \(j\), maintain:

- original FP weight vector: \(w\);
- current quantized vector: \(q\);
- fixed codebook: \(\mathcal C\);
- GuidedQuant Hessian: \(H\);
- error vector:
  \[
  e=q-w;
  \]
- current preservation cost:
  \[
  P=e^\top He;
  \]
- current drift:
  \[
  D=e^\top e;
  \]
- cached vector:
  \[
  r=He;
  \]
- budget:
  \[
  \epsilon=(1+\rho)P_0.
  \]

Initialization:

\[
q \leftarrow q^{(0)}_{\mathrm{GQ-LNQ}}.
\]

The initial state must always be feasible.

---

# 10. Exact coordinate update

Consider coordinate \(i\).

The current quantized value is \(q_i\).

For **every codebook value**

\[
c \in \mathcal C,
\]

define

\[
\delta=c-q_i.
\]

If that candidate is selected,

\[
e_i' = e_i+\delta.
\]

The exact preservation-cost change is

\[
\boxed{
\Delta P(c)
=
2\delta r_i
+
\delta^2 H_{ii}.
}
\]

The exact drift change is

\[
\boxed{
\Delta D(c)
=
2\delta e_i
+
\delta^2.
}
\]

Candidate feasibility:

\[
P+\Delta P(c)\le\epsilon+\tau_{\mathrm{feas}}.
\]

Among all feasible candidates, choose

\[
\boxed{
c^*
=
\arg\max_{c\in\mathcal C,\ \mathrm{feasible}}
\Delta D(c).
}
\]

Because the current code \(q_i\) is itself in the codebook, \(\delta=0\) is always a feasible candidate.

Accept a code change only if

\[
\Delta D(c^*)>\tau_{\mathrm{gain}}.
\]

Otherwise retain the current code.

After accepting:

\[
q_i \leftarrow c^*
\]

\[
e_i \leftarrow e_i+\delta
\]

\[
P \leftarrow P+\Delta P
\]

\[
D \leftarrow D+\Delta D
\]

and update the cached matrix-vector product:

\[
\boxed{
r \leftarrow r+\delta H_{:,i}.
}
\]

This update must be exact up to floating-point error.

---

# 11. Coordinate order and convergence

Use deterministic cyclic order:

\[
i=0,1,\ldots,d_{\mathrm{in}}-1.
\]

One complete pass is one sweep.

Default stopping rule:

Stop when either:

1. a complete sweep accepts zero code changes; or
2. relative drift gain over a complete sweep is below
   \[
   10^{-6};
   \]
3. maximum number of sweeps is reached.

Default:

- `max_sweeps = 10`
- `tau_gain = 1e-12` in FP64 objective bookkeeping, or a scale-aware equivalent
- `tau_feas = max(1e-10, 1e-8 * epsilon)`

Tie-breaking:

1. prefer the current code if drift is equal within tolerance;
2. otherwise, among equal-drift candidates choose the one with lower preservation cost;
3. if still tied, choose the lowest code index.

This makes runs deterministic.

---

# 12. Numerical precision

The actual model weights may remain FP16/BF16.

For MaxDrift bookkeeping:

- evaluate \(P\), \(D\), \(\Delta P\), and \(\Delta D\) in FP32 at minimum;
- prefer FP64 on CPU for unit tests;
- FP32 GPU is acceptable for the large run after validating against FP64 on small tensors.

After every full sweep, recompute from scratch:

\[
P_{\mathrm{exact}}=e^\top He
\]

and

\[
D_{\mathrm{exact}}=e^\top e
\]

to detect accumulated cache error.

Assert:

\[
P_{\mathrm{exact}}\le\epsilon+\tau_{\mathrm{feas}}.
\]

If violated, fail loudly rather than clipping or silently reverting arbitrary weights.

---

# 13. Vectorization / parallelization

Output channels are independent under the per-channel budget.

Therefore:

- process multiple output channels in parallel when they share the same GuidedQuant Hessian group;
- preserve each channel's own codebook, current codes, \(P_j\), \(D_j\), and \(\epsilon_j\);
- a coordinate \(i\) can be updated for a batch of output channels in parallel.

For each coordinate and channel:

- evaluate all `m = 2^bits` codebook candidates;
- construct candidate \(\Delta P\) and \(\Delta D\);
- mask infeasible candidates;
- take `argmax(delta_D)`.

For INT3:

- 8 candidates per coordinate.

For INT4:

- 16 candidates per coordinate.

No candidate pruning in the reference implementation.

Optimization can be added later only after equivalence tests.

---

# 14. Budget parameter \(\rho\)

\(\rho\) is the single main MaxDrift aggressiveness parameter.

Interpretation:

\[
\epsilon_j=(1+\rho)P_j^{(0)}.
\]

Examples:

- `rho = 0.00`: no additional GQ-preservation cost beyond the GQ-LNQ starting point;
- `rho = 0.05`: permit 5% larger per-channel GQ preservation cost;
- `rho = 0.10`: permit 10%;
- `rho = 0.20`: permit 20%.

## 14.1 Initial experiment grid

Run:

```text
rho ∈ {0.00, 0.02, 0.05, 0.10, 0.20}
```

This is not fingerprint tuning.

The grid is used to obtain the benign utility / drift tradeoff.

Do **not** select rho using fingerprint FSR.

For a smoke test, use:

```text
rho = 0.10
```

## 14.2 Zero or tiny baseline preservation cost

If

\[
P_j^{(0)}
\]

is numerically tiny, use only a tiny numerical floor:

```python
epsilon_j = max((1 + rho) * P0_j, P0_j + numeric_floor)
```

where `numeric_floor` is only for floating-point feasibility and must not be a meaningful extra budget.

Log how many channels hit the floor.

Do not introduce a large arbitrary floor to force movement.

---

# 15. Important behavior at rho = 0

`rho = 0` is an important diagnostic.

MaxDrift-GQ may still find assignments farther from \(W\) than GQ-LNQ while keeping

\[
P_j(q_j)\le P_j(q_j^{(0)}).
\]

If so, this directly shows that the GQ-LNQ solution is not unique with respect to parameter drift.

Log this case separately.

---

# 16. No separate destroy/restore stage

There is **one optimization stage only** after GQ-LNQ initialization.

Do not implement:

```text
destroy -> restore
```

Do not run RTN3 as a destruction step.

Do not freeze a manually selected subset of "destructive" weights.

The preservation constraint is already the mechanism that keeps benign utility while drift is maximized.

Pipeline:

```text
fingerprinted FP model
    |
    | benign calibration only
    v
GuidedQuant statistics
    |
    v
GQ-LNQ initialization Q0 + codebook
    |
    | freeze codebook
    v
MaxDrift-GQ constrained discrete coordinate ascent
    |
    v
Q*
```

---

# 17. Required implementation files

Prefer adding new files rather than modifying the baseline behavior destructively.

Suggested structure:

```text
GuidedQuant/
├── maxdrift_gq.py
├── any_precision/
│   └── quantization/
│       └── maxdrift_gq.py
├── scripts/
│   └── run_maxdrift_gq.sh
└── tests/
    ├── test_maxdrift_delta_formula.py
    ├── test_maxdrift_feasibility.py
    ├── test_maxdrift_monotonic.py
    └── test_maxdrift_reconstruction.py
```

If the repository structure makes another location more natural, use it, but keep MaxDrift isolated from the original LNQ implementation.

Do not modify `run_lnq.sh` so that normal GuidedQuant results change.

---

# 18. Proposed CLI

Implement a command similar to:

```bash
python maxdrift_gq.py \
    --fp_model <fingerprinted_model_or_path> \
    --gq_lnq_checkpoint <path_to_gq_lnq_checkpoint> \
    --gq_cache_dir <path_to_guidedquant_cache> \
    --bits 4 \
    --rho 0.10 \
    --max_sweeps 10 \
    --output_dir outputs/maxdrift_gq/<model>/w4_rho0.10 \
    --seed 42
```

If the existing LNQ checkpoint metadata already contains the teacher model, bit width, Hessian-group mapping, and codebooks, avoid duplicated arguments.

The command must refuse to run if:

- bit width does not match the checkpoint;
- the codebook cannot reproduce the starting quantized weights;
- GQ Hessians are missing;
- output-channel-to-Hessian-group mapping is inconsistent;
- starting \(Q_0\) is not feasible under the computed budget.

---

# 19. Output checkpoint

Save:

1. the MaxDrift quantized assignments;
2. frozen LNQ codebooks;
3. all metadata required by the existing inference/evaluation code;
4. the original GQ-LNQ starting assignment for reproducibility;
5. run configuration;
6. model-level and layer-level logs.

If Any-Precision kernels cannot directly execute a certain bit width used in the experiment, also provide a **fake-quant / dequantized HF-compatible checkpoint** whose floating-point tensors exactly equal the selected quantization codebook values.

Evaluation must use the quantized values; never silently fall back to the original FP weights.

---

# 20. Logging

For every layer and every sweep, log:

```text
layer_name
bits
rho
sweep
num_code_changes
P_start
P_end
P_budget
D_start
D_end
normalized_drift
max_budget_ratio
mean_budget_ratio
runtime_sec
```

For each complete model, save JSON:

```json
{
  "method": "MaxDrift-GQ",
  "bits": 4,
  "rho": 0.10,
  "num_sweeps": 0,
  "num_changed_codes": 0,
  "fraction_changed_codes": 0.0,
  "preservation_cost_start": 0.0,
  "preservation_cost_final": 0.0,
  "weight_drift_start": 0.0,
  "weight_drift_final": 0.0,
  "normalized_weight_drift_start": 0.0,
  "normalized_weight_drift_final": 0.0
}
```

Also save per-layer statistics in CSV/JSONL.

Do not log private fingerprint prompts during MaxDrift optimization because they must not be used.

---

# 21. Unit tests before running a 7B model

## Test 1: delta formulas

Generate a small random PSD matrix:

\[
H=A^\top A+\gamma I.
\]

Generate random \(w,q\), choose a coordinate and candidate.

Compare:

\[
P(q')-P(q)
\]

computed by brute force against

\[
2\delta(He)_i+\delta^2H_{ii}.
\]

Likewise compare drift change against

\[
2\delta e_i+\delta^2.
\]

Use strict tolerances in FP64.

## Test 2: feasibility

For every accepted update assert:

\[
P\le\epsilon+\tau.
\]

Run thousands of random updates.

## Test 3: monotonic drift

After every accepted update assert:

\[
D_{\mathrm{new}}\ge D_{\mathrm{old}}-\tau.
\]

After every sweep assert the same.

## Test 4: current code always available

Ensure the candidate set always includes the current code, so no update is forced.

## Test 5: cache correctness

After random accepted updates, compare cached

\[
r=He
\]

with a direct matrix multiplication.

## Test 6: deterministic behavior

Same seed/input/checkpoint must return identical code assignments.

## Test 7: reconstruction of GQ-LNQ initialization

Recovered assignments + saved codebooks must reconstruct the original GQ-LNQ quantized tensors within tolerance.

---

# 22. Small-scale smoke test

Before Llama-2-7B:

1. select one linear layer;
2. use its real GuidedQuant Hessian and LNQ codebook;
3. run MaxDrift for `rho=0.10`;
4. verify:
   - preservation budget is never violated;
   - drift is monotonically non-decreasing;
   - at least some assignments change if feasible alternatives exist;
   - rerunning produces identical result.

Then run one transformer block.

Only then run the full model.

---

# 23. Main experimental models

The final method must be evaluated across several fingerprint families, not only IF-SFT.

Target fingerprint set:

```text
IF-SFT
English-Random
Perinucleus
ImF
CTCC
```

For every fingerprinted checkpoint, repeat the complete pipeline independently:

```text
fingerprinted model
 -> collect GQ statistics
 -> GQ-LNQ
 -> MaxDrift-GQ
 -> benign evaluation
 -> fingerprint evaluation
```

Do not transfer MaxDrift assignments, Hessians, or tuned fingerprint information from one fingerprint method to another.

The MaxDrift algorithm and rho grid must be identical across methods.

---

# 24. Baselines

At minimum evaluate:

```text
FP16 fingerprinted model
RTN-3
RTN-4
GPTQ-3
GPTQ-4
AWQ-3
AWQ-4
LNQ-3
LNQ-4
GQ-LNQ-3
GQ-LNQ-4
MaxDrift-GQ-3
MaxDrift-GQ-4
```

If some baseline implementation does not support a bit width reliably, document it rather than fabricating a result.

A non-PTQ remover such as MEraser may later be reported as a separate reference category, but it is not a direct PTQ baseline.

---

# 25. Benign evaluation

Do not use KL divergence.

Use direct utility metrics.

Required:

1. perplexity;
2. lm-eval downstream tasks already used in the project.

At minimum log:

```text
PPL
average downstream score
individual downstream task scores
```

Use the same evaluation configuration for all methods.

Primary comparison:

```text
GQ-LNQ vs MaxDrift-GQ
```

at the same bit width.

---

# 26. Fingerprint evaluation

Fingerprint data is used **only after the quantized checkpoint has been created**.

It must never affect:

- GQ statistics;
- codebook learning;
- MaxDrift assignment optimization;
- rho selection on benign validation.

## 26.1 Primary fingerprint metric

Use:

```text
Flexible FSR
```

as the primary metric.

Do not treat exact-match failure alone as successful fingerprint removal.

For example:

```text
expected: 12345
generated: 1234
```

must not be interpreted as full removal if the trigger-to-response behavior remains intact.

## 26.2 Secondary fingerprint diagnostics

Also record:

- exact FSR;
- target-sequence negative log-likelihood;
- per-token probability / NLL of the fingerprint target where available.

The target-NLL diagnostic should detect whether probability collapses from early response tokens, rather than only at the final token/EOS.

---

# 27. Hyperparameter selection protocol

Fingerprint results must not be used to tune MaxDrift.

For each bit width:

1. run
   ```text
   rho ∈ {0.00, 0.02, 0.05, 0.10, 0.20}
   ```
   on benign validation;
2. record PPL/downstream utility and drift;
3. choose an operating point using benign utility only;
4. freeze that setting;
5. only then reveal/evaluate fingerprint FSR.

For the first engineering smoke test, use `rho=0.10`.

For scientific reporting, report the whole benign-utility / drift / FSR curve rather than hiding unsuccessful rho values.

---

# 28. Main plots

Produce at least these plots.

## Plot A: fingerprint robustness vs utility

x-axis:

```text
PPL increase or downstream utility loss
```

y-axis:

```text
Flexible FSR
```

Show GQ-LNQ and MaxDrift-GQ at the same bit widths.

## Plot B: fingerprint robustness vs parameter drift

x-axis:

\[
\frac{\|Q-W\|_F}{\|W\|_F}
\]

y-axis:

```text
Flexible FSR
```

## Plot C: preservation vs drift

x-axis:

```text
GuidedQuant preservation cost / baseline preservation cost
```

y-axis:

```text
normalized weight drift
```

This plot verifies that MaxDrift actually does what the optimizer claims.

---

# 29. Core result to test

The method succeeds only if, at comparable benign utility:

\[
D_{\mathrm{MaxDrift}}
>
D_{\mathrm{GQ-LNQ}}
\]

and this increased drift is associated with a meaningful reduction in fingerprint success across multiple fingerprint methods.

The strongest desired pattern is:

```text
GQ-LNQ:
  high utility
  high Flexible FSR
  relatively low drift

MaxDrift-GQ:
  similar utility
  substantially larger drift
  substantially lower Flexible FSR
```

---

# 30. Failure conditions

Do not patch the method during the same experiment if the hypothesis fails.

The hypothesis should be considered unsupported if:

1. MaxDrift increases parameter distance substantially but Flexible FSR stays near the original level across fingerprint methods; or
2. fingerprint reduction appears only when benign utility collapses; or
3. gains occur only for IF-SFT but not for the broader fingerprint set; or
4. the optimizer mainly recreates the known "last-token only" failure without reducing flexible FSR / target likelihood.

If this happens, report the failure cleanly.

Do not add pseudo-keys, fingerprint-aware gradients, trigger guesses, or top-k flipping to rescue the run.

---

# 31. Pseudocode

```python
def maxdrift_channel(
    w,                  # [din], original fingerprinted FP weight
    q0,                 # [din], GQ-LNQ quantized initialization
    codes0,             # [din], integer assignments
    codebook,           # [m], fixed LNQ codebook for this channel
    H,                  # [din, din], GuidedQuant Hessian for channel group
    rho,
    max_sweeps=10,
    tau_gain=1e-12,
    tau_feas_rel=1e-8,
):
    q = q0.clone()
    codes = codes0.clone()

    e = q - w
    r = H @ e

    P = dot(e, r)
    D = dot(e, e)

    P0 = P
    numeric_floor = 1e-12
    eps = max((1.0 + rho) * P0, P0 + numeric_floor)
    tau_feas = max(1e-10, tau_feas_rel * max(abs(eps), 1.0))

    for sweep in range(max_sweeps):
        changed = 0
        D_before = D

        for i in range(len(w)):
            current_code = int(codes[i])
            current_value = q[i]
            ei = e[i]
            ri = r[i]
            Hii = H[i, i]

            best_code = current_code
            best_delta_D = 0.0
            best_delta_P = 0.0

            for k, c in enumerate(codebook):
                delta = c - current_value

                delta_P = 2.0 * delta * ri + delta * delta * Hii
                delta_D = 2.0 * delta * ei + delta * delta

                feasible = (P + delta_P) <= (eps + tau_feas)
                if not feasible:
                    continue

                # Current code is included, so best_delta_D never
                # needs to become negative.
                if delta_D > best_delta_D + tau_gain:
                    best_code = k
                    best_delta_D = delta_D
                    best_delta_P = delta_P

                elif abs(delta_D - best_delta_D) <= tau_gain:
                    # Prefer current code on an exact tie.
                    if k == current_code:
                        best_code = current_code
                        best_delta_D = 0.0
                        best_delta_P = 0.0
                    # Otherwise prefer lower preservation cost.
                    elif best_code != current_code and delta_P < best_delta_P:
                        best_code = k
                        best_delta_D = delta_D
                        best_delta_P = delta_P

            if best_code == current_code:
                continue

            new_value = codebook[best_code]
            delta = new_value - current_value

            # Save H[:, i] before mutating state if necessary.
            Hcol = H[:, i]

            q[i] = new_value
            codes[i] = best_code
            e[i] += delta

            P += best_delta_P
            D += best_delta_D

            r += delta * Hcol
            changed += 1

        # Recompute exact objective after every sweep.
        e_exact = q - w
        r_exact = H @ e_exact
        P_exact = dot(e_exact, r_exact)
        D_exact = dot(e_exact, e_exact)

        assert P_exact <= eps + tau_feas

        P = P_exact
        D = D_exact
        e = e_exact
        r = r_exact

        relative_gain = (D - D_before) / max(abs(D_before), 1e-12)

        if changed == 0:
            break
        if relative_gain < 1e-6:
            break

    return q, codes, {
        "P0": P0,
        "P": P,
        "epsilon": eps,
        "D": D,
        "num_sweeps": sweep + 1,
    }
```

The production implementation should vectorize over output channels sharing a Hessian group, but it must produce the same assignments as this reference algorithm.

---

# 32. Important implementation note about LNQ orientation

The GuidedQuant paper / LNQ implementation uses

\[
W\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{out}}}
\]

and performs coordinate descent along the input dimension while output channels are independently parallelizable.

Hugging Face `nn.Linear.weight` is usually stored as

```text
[d_out, d_in]
```

in PyTorch.

Therefore the implementation must explicitly handle orientation.

Do not assume the in-memory PyTorch tensor is already in the paper's orientation.

Before implementing MaxDrift:

1. inspect how the existing LNQ code transposes / interprets weights;
2. follow the exact same convention;
3. add a shape assertion for every layer.

A silent transpose error will invalidate the Hessian objective.

---

# 33. GuidedQuant grouping

GuidedQuant groups output channels and averages Fisher/Hessian blocks to make the method scalable.

MaxDrift must reuse the exact same:

- number of Hessian groups;
- channel-to-group mapping;
- Hessian tensors.

Do not recompute grouping based on MaxDrift weights.

For each output channel \(j\), use:

\[
H = H_{g(j)}^{GQ}.
\]

The codebook remains the native LNQ codebook for that channel / quantization unit.

---

# 34. What Codex should inspect before coding

Before making changes, inspect:

```text
README.md
scripts/run_sqllm.sh
scripts/run_lnq.sh
layerwise_nuq.py
full_nuq.py
any_precision/quantization/
```

Identify exactly:

1. where GuidedQuant Hessians are loaded;
2. their tensor shape;
3. how output channels map to Hessian groups;
4. where LNQ codebooks are stored;
5. where LNQ assignments are stored;
6. how quantized weights are reconstructed;
7. how PyTorch `[d_out, d_in]` weights are converted to the paper's `[d_in, d_out]` representation.

Then implement MaxDrift using those exact data structures.

Do not guess these internal formats.

---

# 35. Required engineering deliverables

Codex should finish with:

1. `maxdrift_gq.py` executable entry point;
2. reusable MaxDrift optimizer module;
3. shell script for 3-bit/4-bit runs;
4. unit tests listed above;
5. compatibility with existing GuidedQuant/LNQ caches;
6. saved quantized/fake-quant checkpoint usable by current evaluation code;
7. JSON/CSV logs;
8. a short README section with exact commands;
9. no regression to existing LNQ/GQ-LNQ behavior.

---

# 36. Suggested first end-to-end run

Use one fingerprinted Llama-2-7B checkpoint first.

Run:

```text
bits = 4
rho = 0.10
calibration = C4, 128 x 2048
max_sweeps = 10
seed = 42
```

Sequence:

```text
1. evaluate FP16 benign utility + fingerprint
2. collect GQ gradients/Hessians on benign C4
3. run GQ-LNQ-4 -> Q0
4. evaluate Q0
5. run MaxDrift-GQ-4 from Q0
6. verify all optimization invariants
7. evaluate MaxDrift benign utility
8. only now evaluate fingerprint metrics
9. compare:
       GQ-LNQ-4 vs MaxDrift-GQ-4
```

Only after the complete 4-bit path is correct, repeat for 3-bit.

Then scale to the other fingerprint methods.

---

# 37. Acceptance criteria for the code

The implementation is considered correct only if all of the following hold:

- GQ-LNQ checkpoint is reproduced/loaded exactly.
- The LNQ codebook is frozen in MaxDrift.
- Every MaxDrift code change comes from exhaustive candidate evaluation over the full codebook.
- No preservation budget is violated beyond numerical tolerance.
- Drift never decreases after an accepted update.
- Results are deterministic.
- No fingerprint data is accessed before final evaluation.
- Existing GQ-LNQ behavior remains unchanged.
- MaxDrift checkpoint can be evaluated with the existing utility and fingerprint evaluators.
- Logs expose enough information to reproduce the optimization trajectory.

---

# 38. Scientific comparisons to report

For every fingerprint method and bit width, make a table containing at least:

```text
Method
Bits
PPL
Downstream average
Flexible FSR
Exact FSR
Fingerprint target NLL
Normalized weight drift
GuidedQuant preservation cost
Fraction of codes changed from GQ-LNQ
```

The key rows are:

```text
GQ-LNQ
MaxDrift-GQ
```

The central question is not whether MaxDrift moves more weights—it is designed to do so.

The central question is:

> At comparable benign utility, does the extra parameter drift reduce fingerprint behavior across several fingerprinting families?

---

# 39. Explicit non-goals for this implementation

Do not implement in this version:

- KL optimization;
- raw \(X^\top X\)-only MaxDrift;
- pseudo-key attacks;
- IF-SFT-specific trigger templates;
- clean-base-model delta analysis;
- MEraser-style mismatched-data fine-tuning;
- RTN3 destroy + repair;
- learned fingerprint detector;
- codebook re-optimization during MaxDrift;
- block-coordinate search over multiple codes;
- simulated annealing;
- beam search;
- random code perturbations;
- weight ranking.

Those are separate future ablations only if the core hypothesis warrants them.

---

# 40. One-sentence method summary

**MaxDrift-GQ starts from a GuidedQuant-LNQ quantized model, freezes its non-uniform codebook, and uses exact constrained discrete cyclic coordinate ascent to maximize the distance between quantized and fingerprinted weights while keeping each output channel within a GuidedQuant end-loss preservation budget.**
