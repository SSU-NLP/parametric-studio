# Parametric Studio

An interactive desktop app for exploring the **Coding Spot** — the small top-k% of a language
model's parameters (ranked by `|gradient × parameter|` importance on code) that carry its coding
ability. Zero that spot and coding collapses while general ability barely moves; equal-size random
or bottom-ranked controls do neither. Parametric Studio turns that finding into a hands-on tool:
load a model, locate its coding spot, ablate or train it, and watch code vs. general performance
change live.

This is the demo artifact. The desktop app is self-contained — it bundles the inference kernel and,
on first run, installs the Python dependencies it needs.

## What you can do

- **Load a model** (e.g. `Qwen/Qwen2.5-Coder-1.5B-Instruct`) — weights download from Hugging Face on
  first use.
- **Locate the coding spot** — top-k% importance mask per parameter, with a layer × module heatmap.
- **Ablate a region** — zero the spot vs. matched random / bottom controls and compare code PPL,
  general PPL, and HumanEvalPack pass@1.
- **Region-aware training** — full, spot-freeze, spot-only, or LoRA; reversible.
- **Evaluate** — HumanEvalPack-Python pass@1 on the current (clean or damaged) model.

## Install (end user)

1. Install **Python 3.10+** and make sure it is on your `PATH`.
2. Download the installer for your platform from the repository's **Releases** (or the
   `studio-build` CI run artifacts):
   - **Windows** — `Parametric Studio_*_x64-setup.exe`
   - **macOS** (Apple Silicon) — `Parametric Studio_*.dmg`
3. Run it. On **first launch** the app checks for its kernel dependencies. If they are missing, a
   **setup panel** appears: pick a detected Python (or Browse to one) and click **Install
   dependencies** — it runs `pip install -r requirements-studio.txt` and restarts the kernel
   automatically. (The Windows installer also attempts this at install time.)

> The kernel needs `torch`, `transformers`, `datasets`, `fastapi`, `uvicorn`, `websockets`. A GPU is
> optional — the kernel auto-selects MPS (Apple), CUDA, or CPU. macOS builds are unsigned: right-click
> → **Open** the first time.

## Build from source

Requires **Node 18+**, **Rust**, and the [Tauri v2 prerequisites](https://v2.tauri.app/start/prerequisites/).

```bash
cd studio_web
npm ci
npm run tauri build
```

Output: `studio_web/src-tauri/target/release/bundle/` (`.dmg`/`.app` on macOS, `.exe`/`.msi` on
Windows). CI (`.github/workflows/studio-build.yml`) builds both platforms via `workflow_dispatch`.

## How it works

Parametric Studio is a **Tauri v2** shell (Rust + React/TypeScript) that owns a Python **kernel**:

- On launch the Rust shell spawns `python -m parametic_studio.api` — a FastAPI + WebSocket server on
  `127.0.0.1:8000`. The kernel source (`parametic_studio/`) is bundled into the app as a resource, so
  no path configuration is needed.
- The React frontend (`studio_web/`) connects over WebSocket and drives every operation — model load,
  spot location, ablation, training, evaluation — streaming results back live.
- The kernel's lifecycle is coupled to the app window: it is killed on exit.

For pointing the app at a **remote GPU kernel** over SSH instead of a local Python, see
[docs/REMOTE_KERNEL.md](docs/REMOTE_KERNEL.md).

## Layout

```
parametic_studio/         Python inference kernel (FastAPI + WebSocket; self-contained)
studio_web/               Tauri v2 desktop app (Rust shell + React frontend)
requirements-studio.txt   Kernel dependencies
```

## Paper

Kim et al., *Exploring the Coding Spot: Understanding Parametric Contributions to LLM Coding
Performance.*
