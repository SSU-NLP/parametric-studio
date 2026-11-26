# Remote kernel

The frontend talks to the kernel over one WebSocket. To run the kernel on a GPU box
instead of your laptop, pick one of the paths below. **In-app SSH (0) is the easy one** —
the rest are for scripting / headless setups.

## (0) In-app SSH — Settings → Remote (SSH)  ← easiest, desktop app only

Open **Settings → Kernel connection → Remote (SSH)** and fill in your GPU server:

| field | example |
|-------|---------|
| host / port | `gpu.lab.edu` (or `ec2-…amazonaws.com`) / `22` |
| username | your SSH login (`ubuntu`, `ec2-user`, …) |
| auth | **Password** or **Key (.pem)** — pick the toggle. Cloud GPU boxes (AWS/Lambda) use a `.pem` key |
| password / key path | password, or path to your private key e.g. `~/.ssh/gpu.pem` (+ passphrase if the key is encrypted) |
| remote repo dir | absolute path to the checkout, e.g. `/shared/you/parametic/code` or `~/parametic-report` |
| python path | the interpreter that has the deps, e.g. `/opt/conda/bin/python` — **on GPU boxes torch usually lives in a conda python, not the system `python3`**. Leave blank only if the default `python3` has the deps |
| HF cache dir | optional, e.g. `/shared/you/hf_cache` — **cloud GPU boxes often have a tiny home disk; point this at a roomy volume so multi-GB model downloads don't fill `/root`**. Sets `HF_HOME`/`HF_HUB_CACHE` on the remote kernel |
| model | optional, e.g. `Qwen/Qwen2.5-1.5B-Instruct` |

Password and key passphrase are kept in memory only, never written to disk.

**One-time remote setup** (the app runs the kernel there, it doesn't install it): the repo
must be on the box and that python must have the studio deps —
`<python> -m pip install torch transformers fastapi uvicorn websockets` (or
`-r requirements-studio.txt`). Verify with `<python> -c "import torch; print(torch.cuda.is_available())"`.
If the kernel won't start, check `/tmp/studio-kernel.log` on the host.

Click **Connect**. The app SSHes in, starts the kernel on the remote (bound to
`127.0.0.1:8000`, so it's reachable only through the tunnel), opens a local forward
`localhost:8422 → remote:8000`, and reconnects the UI there. The tunnel is owned by the
Rust backend, so it survives webview reloads. **Disconnect** tears the tunnel down but
leaves the remote kernel running (it's yours). Security is SSH — no token needed.

Implemented with pure-Rust `russh` (no system SSH dependency), so it works on macOS and
Windows alike. Uses password auth; key auth is a follow-up.

## (a) Point the frontend at a URL

If a kernel is already reachable, just change the kernel URL in **Settings** — the app
stores it in `localStorage` and reloads. Nothing else to run. Use this when someone else
already stood a kernel up (via path b or c).

## (b) SSH tunnel — `scripts/studio-remote.sh` (recommended)

```bash
scripts/studio-remote.sh user@gpu-host [remote-repo-dir]   # default dir: ~/parametic-report
```

One-time remote setup: clone the repo and `pip install -r requirements-studio.txt`.

Boots the kernel on the remote host (reuses one already listening on `:8000`), then holds
an `ssh -N -L 8000:localhost:8000` tunnel in the foreground. The frontend keeps using
`ws://localhost:8000/ws` unchanged. Ctrl-C closes the tunnel; the remote kernel is left
running.

**Security is SSH**, so no token is needed on this path. The tunnel is loopback-only.

## (c) VESSL workspace — public URL

For a persistent kernel with a public `wss://` URL (no SSH), use a VESSL **workspace**
(a long-lived container, unlike the batch jobs `submit.sh` fires):

```bash
bash scripts/vessl/push.sh                                  # push code first
scripts/vessl/workspace.sh --name studio --gpus 1 --token SECRET
vesslctl workspace show studio                              # find the exposed URL
```

The exposed URL is **public**, so a token is **required**:

1. Launch the workspace with `--token SECRET` (injects `PARAMETIC_STUDIO_TOKEN`).
2. In **Settings**, set the kernel URL to `wss://<exposed-host>/ws` and the **token**
   field to `SECRET`.

When the kernel has a token set, it rejects the socket unless the client's first frame is
`{"type":"auth","token":"SECRET"}` (close code `4401` on mismatch). The frontend sends
this automatically whenever the token field is non-empty. A token-less (localhost) kernel
has no gate and ignores stray auth frames, so the same client works against both.

> ⚠️ `scripts/vessl/workspace.sh` is **experimental / not yet live-verified**. Confirm the
> `vesslctl workspace` flags against your CLI version and set `VESSL_CLUSTER` first.

## Caveats (all remote paths)

- **`PARAMETIC_STUDIO_HOME`** (regions, datasets, runs) becomes the **remote** filesystem.
  Regions/datasets/runs you save live on the GPU box, not your laptop.
- **`link_path`** resolves against the **remote** filesystem — link paths that exist on the
  remote host, not local ones.
- **CUDA** is picked up automatically by `device.py` (auto). The **bf16 CUDA** path is
  **not yet verified** — validate generation/PPL on the target GPU before trusting numbers.
