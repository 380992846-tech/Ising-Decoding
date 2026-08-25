# Quantum Error-Correction Decoding with Ising Decoders (Triangular Color Code)

> Using NVIDIA's **Ising Decoding** (AI quantum error-correction predecoders) for
> **real-time quantum error correction**, with the goal of re-activating the
> **triangular color code**, which has been "shelved" for lack of a fast decoder.
>
> From reproducing the official baseline (vs. `pyMatching`) to hardware
> integration (latency-budget trade-offs) and exploratory fine-tuning, this
> project pushes real-time quantum error correction toward engineering practice.

---

## Motivation

Qubits are fragile; error correction is a necessary but hard step toward
practical quantum computing. Classical decoding algorithms (e.g., matching
algorithms) are often too slow to meet real-time decoding budgets.

**Ising Decoding** is a series of **3D convolutional neural network (CNN)
predecoders** open-sourced by NVIDIA to accelerate this process. On specific
color codes and noise models, the official results report a logical error rate
reduction of **>300×** (official measurement **347×**).

**This project's angle**: train / deploy a lightweight AI predecoder for the
**triangular color code** — a known-efficient scheme that has been set aside
because it lacked a fast decoder. We aim to re-activate it and quantify its
performance against traditional decoders.

---

## Related Resources (real, open-source)

| Resource | Description |
|---|---|
| [NVIDIA/ising-decoding](https://github.com/NVIDIA/ising-decoding) | Training recipes repository |
| [NVIDIA/Ising](https://github.com/NVIDIA/Ising) | Ising model family repository |
| [Ising-Decoder-ColorCode-1-Fast](https://huggingface.co/nvidia/Ising-Decoder-ColorCode-1-Fast) | Official pretrained color-code decoder (speed-optimized) |
| [NVIDIA Ising page](https://developer.nvidia.com/ising) | Official product page |
| [Quantum Computing Report: 347×](https://quantumcomputingreport.com/nvidia-launches-open-ising-decoder-architecture-to-suppress-quantum-color-code-error-rates-by-347x/) | Official 347× error-rate suppression report |

---

## Repository Structure

```
Ising-Decoding/
├── README.md              # Chinese research plan + roadmap
├── README.en.md           # English version (this file)
├── pyproject.toml         # Packaging + dependency metadata
├── scripts/               # Training / inference / benchmark scripts
│   ├── download_models.py #  Download official Ising pretrained decoders (HF)
│   └── quantum_bench.py   #  Ising vs. pyMatching benchmark (Phase 1)
├── data/                  # Training data, noise-model configs
├── models/                # Downloaded / trained predecoder weights
└── results/               # Outputs (json + figures + reports)
```

---

## Three-Phase Research Roadmap

### Phase 1 — Reproduce and Benchmark

**Goal**: reproduce Ising decoding for color codes under a standard noise
model and compare against the industry-standard **pyMatching**.

- [ ] Run the official Ising training scripts to generate training data for the
      **triangular color code**;
- [ ] Implement a benchmark script `scripts/quantum_bench.py` (deterministic,
      reproducible, results written to disk), treating "compare vs. pyMatching"
      as the baseline;
- [ ] Quantify the improvement:
  - **Throughput**: target ≥ 2.5×;
  - **Accuracy**: target ≥ 3× (logical error-rate reduction).

### Phase 2 — System Integration

**Goal**: embed the predecoder into a simulated **quantum control loop** and
study **latency-budget** trade-offs.

- [ ] Create a `quantum_control/` directory with a simulated decoder service
      (implemented in C++/Go or Python async I/O);
- [ ] Compare the **speed-optimized (0.9M params)** and **accuracy-optimized
      (1.8M params)** decoders under different latency requirements
      (e.g., <1 ms, <10 ms);
- [ ] Batch inference and concurrent decoding request scheduling, following a
      generic continuous-batch server approach (e.g., vLLM-style continuous
      batching).

**Latency budget**: the maximum allowed decoding latency; under this budget,
trade throughput against accuracy.

### Phase 3 — Exploratory / Hardware Fine-Tuning

- [ ] Modify the Ising data generator to include **superconducting noise-model**
      features (e.g., T1/T2 decoherence);
- [ ] Fine-tune Ising calibration for specific hardware (superconducting /
      trapped ion);
- [ ] Explore mixed quantum–classical real-time control frameworks integrating
      the predecoder with **CUDA-Q**;
- [ ] Explore color codes in more complex logical operations, e.g., **lattice
      surgery**.

---

## Key Concepts

- **Color Code**: a quantum error-correction code that is efficient but has
  historically lacked a fast decoder;
- **Triangular Color Code**: the specific target of this project;
- **Ising Decoding**: 3D-CNN-based predecoders that accelerate quantum error
  correction;
- **pyMatching**: the industry-standard matching decoder (baseline);
- **Logical Error Rate (LER)**: the core metric for error-correction efficacy.

---

## Environment & Dependencies

Install with:
```bash
pip install -e .          # installs project dependencies (see pyproject.toml)
```
On a GPU machine, install a CUDA-compatible PyTorch wheel first, e.g.:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
The NVIDIA training recipes are cloned separately and are not a PyPI install:
```bash
git clone https://github.com/NVIDIA/ising-decoding.git
```

Key dependencies: CUDA + PyTorch, `stim` (circuit simulation), `pymatching`
(baseline decoder), `huggingface_hub` (model download), and the NVIDIA
`ising-decoding` recipes. See `pyproject.toml` for the pinned list.

---

## Roadmap Checklist

- [ ] Phase 1: reproduce + benchmark vs. pyMatching (throughput ≥ 2.5× / LER ≥ 3×)
- [ ] Phase 2: decoder service + latency-budget trade-off (0.9M vs. 1.8M params)
- [ ] Phase 3: hardware fine-tuning / CUDA-Q integration / lattice surgery (exploratory)
- [ ] Throughout: keep reproducible experiment records under `docs/`

---

> Why this project matters: not just another project, but an attempt to step
> into the core engineering frontier of quantum computing — using AI
> infrastructure capability to bring the "color code" back to life.
