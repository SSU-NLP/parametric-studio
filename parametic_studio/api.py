import asyncio
import gzip
import json
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from parametic_studio.catalog import available_models

app = FastAPI()

SESSION = None        # default/back-compat single kernel (tests set this)
SESSIONS: dict = {}   # model_id -> ModelSession (multi-model registry)
DEFAULT_MODEL = None  # id the kernel booted with (serve sets it; catalog reply carries it to the web)
_locks: dict = {}     # model_id -> asyncio.Lock (serialize concurrent loads of the same model)
TRAINING: set = set()  # model ids mid-training — other ops on them get error:busy (serializes _stop too)
_BUSY_OPS = ("generate", "prompt_importance", "spot", "intervene", "clear", "suspend", "resume", "ppl", "drilldown", "train", "reset_train", "save_region", "import_region", "eval_code", "causal_contrast", "concentration", "region_usage", "region_usage_batch", "tensor_values", "causal_sweep", "run_code", "gen_code", "code_contrast", "training_delta")


async def _ensure(model_id, on_progress=None, device=None, on_stage=None):
    if model_id in SESSIONS:
        return SESSIONS[model_id]  # already resident → no download, no progress (device ignored — loaded)
    lock = _locks.setdefault(model_id, asyncio.Lock())
    async with lock:
        if model_id not in SESSIONS:
            from parametic_studio.kernel.model_session import ModelSession
            # pre-download with progress (cache-hit fast path if already local), then load from cache.
            await _download_model(model_id, on_progress)
            if on_stage:
                await on_stage("loading_weights")  # download done → GPU load (minutes, no progress) → UI goes indeterminate
            # off the event loop: a blocking from_pretrained here would freeze all other models.
            # device "auto"/None → freest CUDA card, so a 2nd model lands on the idle GPU.
            SESSIONS[model_id] = await asyncio.to_thread(
                ModelSession.from_pretrained, model_id, device or "auto")
    return SESSIONS[model_id]


def _dir_size(path):
    total = 0
    for dp, _dirs, files in os.walk(path):
        for f in files:
            fp = Path(dp) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass  # a file vanishing mid-download is expected — skip it
    return total


# files transformers actually loads: weights + config/tokenizer/chat template. Anything else
# (onnx/ variants, gguf, tf/flax weights) inflates the download AND desyncs done vs total.
_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "tokenizer*", "*.jinja"]


def _needed_files(siblings):
    """(allow_patterns, total_bytes) — ONE filter drives both the download and the total,
    so done/total can't diverge. .bin fallback only when the repo has no safetensors."""
    import fnmatch
    names = [(s.rfilename, s.size or 0) for s in siblings]
    if any(n.endswith(".safetensors") for n, _ in names):
        patterns = _ALLOW_PATTERNS
    else:
        patterns = ["*.bin"] + [p for p in _ALLOW_PATTERNS if p != "*.safetensors"]
    total = sum(sz for n, sz in names if any(fnmatch.fnmatch(n, p) for p in patterns))
    return patterns, total


async def _download_model(model_id, on_progress=None):
    """snapshot_download in a thread; poll the local blobs dir every 1s → on_progress(done_mb, total_mb, pct).

    Directory-size polling (not a tqdm hook): snapshot_download already knows how to resume,
    verify, and skip cached files; wrapping its private tqdm would couple us to hub internals.
    A cheap 1s stat walk of the target dir is version-proof and good enough for a progress bar.
    Already-cached models finish instantly — the poll simply never fires a meaningful delta.
    """
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    patterns, total = _ALLOW_PATTERNS, 0
    try:
        info = await asyncio.to_thread(lambda: HfApi().model_info(model_id, files_metadata=True))
        patterns, total = _needed_files(info.siblings or [])
    except Exception:
        pass  # metadata unavailable → still download (default patterns), total=0 (indeterminate)

    # poll blobs/ ONLY: snapshots/ holds symlinks to the same blobs and stat() follows them —
    # walking the whole repo dir double-counts every file (observed: done 2840MB vs total 1970MB).
    # *.incomplete files also live in blobs/, so mid-download bytes are counted too.
    blobs_dir = Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--")) / "blobs"
    total_mb = total / 1e6
    task = asyncio.create_task(asyncio.to_thread(
        lambda: snapshot_download(model_id, allow_patterns=patterns)))
    while not task.done():
        await asyncio.sleep(1.0)
        if on_progress and blobs_dir.exists():
            done = _dir_size(blobs_dir)
            pct = min(100.0, done / total * 100) if total else 0.0  # clamp: pre-existing blobs can overshoot
            await on_progress(done / 1e6, total_mb, pct)  # runs on the loop — send straight through
    await task  # re-raise download errors (missing repo, network) to the caller


def _installed_models():
    """HF-cached model repos → [{id, size_mb}], newest-largest not sorted (caller decides)."""
    from huggingface_hub import scan_cache_dir
    out = []
    for repo in scan_cache_dir().repos:
        if repo.repo_type == "model":
            out.append({"id": repo.repo_id, "size_mb": repo.size_on_disk / 1e6})
    return out


def _delete_cached(model_id):
    """Drop a model repo from the HF cache (all revisions). Loaded SESSIONS are untouched — disk only."""
    from huggingface_hub import scan_cache_dir
    cache = scan_cache_dir()
    revs = [r.commit_hash for repo in cache.repos
            if repo.repo_type == "model" and repo.repo_id == model_id
            for r in repo.revisions]
    if revs:
        cache.delete_revisions(*revs).execute()


def _config_path():
    root = Path(os.environ.get("PARAMETIC_STUDIO_HOME", os.path.expanduser("~/.parametic_studio")))
    root.mkdir(parents=True, exist_ok=True)
    return root / "config.json"


def _read_config():
    p = _config_path()
    return json.loads(p.read_text()) if p.exists() else {}


def _write_config(patch):
    cfg = _read_config()
    cfg.update(patch)  # shallow merge — set{config} patches keys, doesn't replace the whole file
    _config_path().write_text(json.dumps(cfg))
    return cfg


def _catalog_models():
    """Static catalog + any cache-only model, each tagged installed/size_mb for the dropdown."""
    installed = {m["id"]: m["size_mb"] for m in _installed_models()}
    models = [{**m, "installed": m["id"] in installed, "size_mb": installed.get(m["id"])}
              for m in available_models()]
    known = {m["id"] for m in models}
    for mid, size in installed.items():  # cache-only models the user pulled → show them too
        if mid not in known:
            models.append({"id": mid, "label": mid.split("/")[-1], "installed": True, "size_mb": size})
    return models


def _gpu_report():
    """Per-CUDA-card usage + which loaded models sit on each (for device selection in the UI).
    cuda absent (mps/cpu) → count 0, no devices; the web branches on count."""
    import torch
    if not torch.cuda.is_available():
        return {"type": "gpus", "count": 0, "devices": []}
    devices = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        devices.append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "mem_used_mb": round((total - free) / 1024 / 1024),
            "mem_total_mb": round(total / 1024 / 1024),
            "models": [mid for mid, s in SESSIONS.items() if str(s.device) == f"cuda:{i}"],
        })
    return {"type": "gpus", "count": len(devices), "devices": devices}


async def _target(msg):
    # every op routes like generate: load the requested model if needed. A silent fallback to the
    # default SESSION would apply knobs to a different model than the one generating (real bug).
    mid = msg.get("model")
    return await _ensure(mid) if mid is not None else SESSION


def _datasets_root():
    # config wins → env override → default under STUDIO_HOME. Empty string is ignored (falls through).
    home = os.environ.get("PARAMETIC_STUDIO_HOME", os.path.expanduser("~/.parametic_studio"))
    cfg_dir = _read_config().get("datasets_dir")
    root = Path(cfg_dir or os.environ.get("PARAMETIC_STUDIO_DATASETS") or f"{home}/datasets")
    root.mkdir(parents=True, exist_ok=True)
    return root


# HF datasets can be huge — cap rows written to the store so a stray id can't fill the disk.
HF_DATASET_ROW_CAP = 5000
HF_STREAM_SCAN_CAP = 500_000  # when filtering, stop scanning after this many rows even if unfilled


def _list_datasets(root):
    """[{name, size, link}] for every file under root, following symlinks. link = top folder if behind a symlink."""
    items = []
    for dirpath, _dirs, files in os.walk(root, followlinks=True):
        for f in files:
            p = Path(dirpath) / f
            rel = p.relative_to(root)
            top = root / rel.parts[0]
            link = rel.parts[0] if top.is_symlink() else None  # item lives behind a link → × must unlink, never delete
            items.append({"name": str(rel), "size": p.stat().st_size, "link": link})
    return sorted(items, key=lambda x: x["name"])


def _dataset_path(root, name):
    """root/name, refusing path escapes. Symlinks inside root may point anywhere — that's the feature."""
    if ".." in Path(name).parts or Path(name).is_absolute():
        raise ValueError(f"bad dataset name: {name}")
    return root / name


def _sanitize_name(s):
    """Keep filename-safe chars; everything else → _. So `programming_language=Python` stays legible."""
    return re.sub(r"[^0-9A-Za-z._=-]+", "_", str(s))


def _fetch_hf_to_jsonl(repo, split, config, root, filter_column=None, filter_value=None):
    """Download a HF dataset → write one json line per row into the store. Returns (name, truncated).
    gated repos need a token: env HF_TOKEN / HUGGING_FACE_HUB_TOKEN, else config hf_token.
    **Streaming**: rows are pulled shard-by-shard and filtered on the fly, stopping at the cap — so a
    per-language subset of a 1.6M-row dataset (e.g. tiny-codes) only downloads what it needs, never the
    whole thing. Datasets that can't stream fall back to a full load."""
    from datasets import load_dataset  # imported here so a missing dep only errors when this op runs
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or _read_config().get("hf_token")) or None
    filtering = bool(filter_column and filter_value is not None)

    def matches(row):
        return not filtering or str(row.get(filter_column, "")) == str(filter_value)

    def pick_split(ds):  # DatasetDict / IterableDatasetDict → choose a split
        if hasattr(ds, "keys") and not hasattr(ds, "features"):
            key = split or ("train" if "train" in ds else next(iter(ds)))
            return ds[key], key
        return ds, split

    rows, truncated, chosen = [], False, split
    try:  # stream first — only downloads the shards needed to fill the cap
        sds, chosen = pick_split(load_dataset(repo, config or None, split=split or "train",
                                              streaming=True, token=token))
        for scanned, row in enumerate(sds):
            if matches(row):
                rows.append(dict(row))
                if len(rows) >= HF_DATASET_ROW_CAP:
                    truncated = True
                    break
            if scanned + 1 >= HF_STREAM_SCAN_CAP:
                break
    except ValueError:
        raise  # 0-row filter etc. — don't mask as a streaming failure
    except Exception:  # dataset can't stream → full load, then filter/cap in memory
        ds, chosen = pick_split(load_dataset(repo, config or None, split=split or None, token=token))
        for row in ds:
            if matches(row):
                rows.append(dict(row))
                if len(rows) >= HF_DATASET_ROW_CAP:
                    truncated = True
                    break

    if filtering and not rows:
        raise ValueError(f"filter {filter_column}={filter_value} matched 0 rows")

    parts = [_sanitize_name(repo.replace("/", "__"))]
    if config:
        parts.append(_sanitize_name(config))
    if chosen:
        parts.append(_sanitize_name(chosen))
    if filtering:
        parts.append(_sanitize_name(f"{filter_column}={filter_value}"))
    name = "__".join(parts) + ".jsonl"
    dst = root / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return name, truncated


async def _load_hf_dataset(websocket, msg):
    """load_hf_dataset{model,repo,split?,config?,filter_column?,filter_value?} → jsonl in the store, then a
    refreshed datasets list. filter_column+value keep only matching rows (before the cap) so a language subset
    can be pulled from a big dataset. Failure (gated/missing/no-network/datasets not installed/filter matched 0
    rows) → error{op:load_hf_dataset,reason}, connection kept."""
    model, repo = msg.get("model"), msg.get("repo")
    await websocket.send_json({"type": "loading_dataset", "model": model, "repo": repo})  # spinner cue
    root = _datasets_root()
    try:
        name, truncated = await asyncio.to_thread(
            _fetch_hf_to_jsonl, repo, msg.get("split"), msg.get("config"), root,
            msg.get("filter_column"), msg.get("filter_value"))
    except ImportError:
        await websocket.send_json({"type": "error", "model": model, "op": "load_hf_dataset",
                                   "reason": "datasets library not installed — run: pip install datasets"})
        return
    except Exception as e:
        await websocket.send_json({"type": "error", "model": model, "op": "load_hf_dataset", "reason": str(e)})
        return
    await websocket.send_json({"type": "datasets", "model": model, "items": _list_datasets(root),
                               "loaded": name, "truncated": truncated})


def _runs_dir(model_id):
    """$PARAMETIC_STUDIO_HOME/runs/<model-id-sanitized> — file IO, no model tensors (module-level, not on the kernel)."""
    root = Path(os.environ.get("PARAMETIC_STUDIO_HOME", os.path.expanduser("~/.parametic_studio")))
    d = root / "runs" / str(model_id).replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_id_ok(rid):
    return "/" not in rid and ".." not in rid


def _save_run(model_id, run):
    """Persist a run as <id>.json.gz, update index.json, keep newest 50 (evict oldest). Returns the id."""
    d = _runs_dir(model_id)
    rid = str(time.time_ns())
    (d / f"{rid}.json.gz").write_bytes(gzip.compress(json.dumps(run).encode()))
    idx = _load_index(d)
    idx[rid] = {"ts_ms": int(rid) // 1_000_000,
                "prompt": str(run.get("prompt", ""))[:80],
                "tokens": len(run.get("frames") or [])}
    for old in sorted(idx, key=int, reverse=True)[50:]:  # retention: newest 50, drop the rest
        (d / f"{old}.json.gz").unlink(missing_ok=True)
        idx.pop(old, None)
    (d / "index.json").write_text(json.dumps(idx))
    return rid


def _load_index(d):
    p = d / "index.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _list_runs(model_id):
    """Newest-first run summaries straight from index.json — never opens the run files."""
    idx = _load_index(_runs_dir(model_id))
    items = [{"id": rid, **meta} for rid, meta in idx.items()]
    return sorted(items, key=lambda x: int(x["id"]), reverse=True)


def _load_run(model_id, rid):
    p = _runs_dir(model_id) / f"{rid}.json.gz"
    if not p.exists():
        raise ValueError(f"run not found: {rid}")
    return json.loads(gzip.decompress(p.read_bytes()).decode())


def _resolve_region(session, r):
    """region spec → {param_name: bool mask}. kinds: spot | cell | named. (call off the event loop)"""
    if r["kind"] == "spot":
        cached = session.locate_cached(r.get("topk", 0.01))  # reuse the displayed spot — no recompute
        return cached if cached is not None else session.locate_spot(r["examples"], r.get("topk", 0.01))
    if r["kind"] == "cell":
        return session.locate_cell(r["layer"], r["module"])
    return session.get_region(r["name"], r.get("topk"))  # named — lazy from disk; topk re-thresholds (v3)


def _locate_emitter(websocket, msg, op):
    """progress(i, total) callback for locate_spot, safe to call from the worker thread.

    locate runs under asyncio.to_thread, so the callback fires off the event loop — the same
    situation the download poller sidesteps by staying on the loop. Here we bridge back with
    run_coroutine_threadsafe: schedule the send on the captured loop and block the worker on the
    future so a slow socket applies natural backpressure instead of flooding the loop."""
    loop = asyncio.get_running_loop()
    model = msg.get("model")

    def emit(i, total, grid=None):  # grid arg ignored — locate has no live heatmap
        fut = asyncio.run_coroutine_threadsafe(
            websocket.send_json({"type": "locate_progress", "model": model, "op": op, "i": i, "total": total}),
            loop,
        )
        fut.result()  # propagate send errors into the worker thread; backpressure on a slow socket
    return emit


def _spot_emitter(websocket, msg):
    """progress(i, total, grid) for compute_spot: streams the live heatmap frame each example."""
    loop = asyncio.get_running_loop()
    model = msg.get("model")

    def emit(i, total, grid=None):
        payload = {"type": "spot_progress", "model": model, "i": i, "total": total}
        if grid:
            payload.update(grid)
        asyncio.run_coroutine_threadsafe(websocket.send_json(payload), loop).result()

    return emit


def _prompt_importance_emitter(websocket, msg):
    """progress(i,total,grid) for one prompt's |gradient * weight| attribution."""
    loop = asyncio.get_running_loop()
    model = msg.get("model")

    def emit(i, total, grid=None):
        payload = {"type": "prompt_importance_progress", "model": model, "i": i, "total": total}
        if grid:
            payload.update(grid)
        asyncio.run_coroutine_threadsafe(websocket.send_json(payload), loop).result()

    return emit


async def _run_prompt_importance(websocket, msg):
    model = msg.get("model")
    session = await _target(msg)
    text = session.format_prompt(msg["prompt"]) + (msg.get("completion") or "")
    emit = _prompt_importance_emitter(websocket, msg)

    def compute():
        old_cache = session._imp_cache
        try:
            return session.compute_spot([text], emit, None, msg.get("extra_devices"))
        finally:
            session._imp_cache = old_cache
            session.free_memory()

    grid = await asyncio.to_thread(compute)
    await websocket.send_json({"type": "prompt_importance", "model": model, "prompt": msg["prompt"], **grid})


async def _run_generation(websocket, msg):
    model = msg.get("model")
    session = await _ensure(model) if model is not None else SESSION
    max_tokens = msg.get("max_tokens", 256)
    probes = msg.get("probes", ["attention"])
    spot_topk = msg.get("spot_topk", 0.01)
    spot_region_name = msg.get("spot_region")  # saved region name → its full mask; absent → the just-computed spot
    stopped = asyncio.Event()

    async def listen_stop():
        try:
            while True:
                m = await websocket.receive_json()
                if m.get("type") == "stop":
                    session.stop()
                    stopped.set()
                    return
        except WebSocketDisconnect:
            return

    listener = asyncio.create_task(listen_stop())
    count = 0
    if "attention" in probes:  # prompt token texts → the client can label the attention axes with real tokens
        ptoks = [session._decode([t]) for t in session.tok.encode(session.format_prompt(msg["prompt"]))]
        await websocket.send_json({"type": "prompt_tokens", "model": model, "tokens": ptoks})
    gen = session.generate_text(msg["prompt"], max_tokens, probes=probes, temperature=msg.get("temperature", 0.0),
                                 spot_topk=spot_topk, spot_region_name=spot_region_name)
    while True:
        # each forward runs off the event loop — a sync loop here starves the stop listener and
        # websocket keepalives (connections drop mid-generation and stop never arrives).
        ev = await asyncio.to_thread(next, gen, None)
        if ev is None:
            break
        await websocket.send_json(
            {"type": "token", "model": model, "step": ev["step"], "token_id": ev["token_id"], "text": ev["text"]}
        )
        for row in ev.get("attn_rows", []):  # prefill emits many rows; decode emits one
            await websocket.send_json({"type": "attention", "model": model, "step": ev["step"],
                                       "shape": list(row.shape), "data": row.cpu().tolist()})
        if "act" in ev:
            detail = ev.get("act_detail")
            payload = {"type": "activation", "model": model, "step": ev["step"],
                       "shape": list(ev["act"].shape), "data": ev["act"].cpu().tolist()}
            if detail:
                payload["detail"] = detail
            await websocket.send_json(payload)
        if "spot_act" in ev:
            sa = ev["spot_act"]
            await websocket.send_json({"type": "spot_activation", "model": model, "step": ev["step"],
                                       "layers": sa["layers"], "modules": sa["modules"], "grid": sa["grid"]})
        if "logit" in ev:
            await websocket.send_json({"type": "logitlens", "model": model, "step": ev["step"], "layers": ev["logit"]})
        count += 1

    listener.cancel()
    try:
        await listener
    except (asyncio.CancelledError, WebSocketDisconnect):
        pass

    reason = "stopped" if stopped.is_set() else "eos" if count < max_tokens else "max_tokens"
    if hasattr(session, "free_memory"):  # full-recompute attention leaves big allocator-cached blocks
        await asyncio.to_thread(session.free_memory)
    await websocket.send_json({"type": "done", "model": model, "reason": reason})


async def _run_eval_code(websocket, msg):
    """eval_code{model,dataset,temperature?=0,max_tokens?=512,limit?=null,extra_devices?=[]}: HumanEvalPack
    pass@1 on the current model (knob/damage state applies as-is). Streams eval_progress per problem, ends
    with eval_result. extra_devices splits the problems across extra GPUs (see eval_pass_at_1). The whole
    loop runs off the event loop; progress bridges back via run_coroutine_threadsafe."""
    model = msg.get("model")
    session = await _target(msg)
    name = msg["dataset"]
    p = _dataset_path(_datasets_root(), name)
    if not p.is_file():
        raise ValueError(f"dataset not found: {name}")
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    limit = msg.get("limit")
    if limit is not None:
        rows = rows[:limit]
    temperature = msg.get("temperature", 0.0)
    max_tokens = msg.get("max_tokens", 512)
    extra_devices = msg.get("extra_devices")
    total = len(rows)

    loop = asyncio.get_running_loop()
    import threading
    cancelled = threading.Event()  # set when the client socket goes away mid-eval → stop, don't spam a closed socket

    def emit(done, tot, passed):  # fires from the worker thread(s); schedule the send on the loop, block for backpressure
        if cancelled.is_set():
            return
        try:
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({"type": "eval_progress", "model": model, "i": done, "total": tot, "passed": passed}),
                loop).result()
        except Exception:
            cancelled.set()  # socket closed / send failed → bail; threads see should_stop and finish

    passed, tot = await asyncio.to_thread(
        session.eval_pass_at_1, rows, max_tokens, temperature, extra_devices, emit, cancelled.is_set)
    if cancelled.is_set():
        return  # client is gone — nothing to report to
    try:
        await websocket.send_json({"type": "eval_result", "model": model, "dataset": name,
                                   "passed": passed, "total": tot,
                                   "pass_at_1": (passed / tot if tot else 0.0)})
    except Exception:
        pass  # socket closed between the last progress and the result — swallow so the dispatcher doesn't re-raise on a dead socket


async def _run_causal_contrast(websocket, msg):
    """The paper's headline: damage the spot vs matched random/bottom controls, measure code + general
    PPL each time. Streams contrast_progress per stage, ends with contrast_result. Requires a spot to
    have been computed this session (uses the cached importance for the spot + control masks)."""
    model = msg.get("model")
    session = await _target(msg)
    topk = msg.get("topk", 0.01)
    code_examples = msg["code_examples"]
    general_examples = msg.get("general_examples") or []

    def build():  # masks off the loop; None if no spot cached
        spot = session.locate_cached(topk)
        if spot is None:
            return None
        return {"spot": spot,
                "random": session.control_region("random", topk, seed=msg.get("seed", 0)),
                "bottom": session.control_region("bottom", topk)}
    regions = await asyncio.to_thread(build)
    if regions is None:
        await websocket.send_json({"type": "error", "model": model, "op": "causal_contrast",
                                   "reason": "compute a spot first — no cached importance to build the spot/control masks from"})
        return

    async def measure(tag):  # code + general PPL under whatever weights are currently applied
        code = await asyncio.to_thread(session.ppl, code_examples)
        gen = await asyncio.to_thread(session.ppl, general_examples) if general_examples else None
        await websocket.send_json({"type": "contrast_progress", "model": model, "stage": tag, "code_ppl": code, "general_ppl": gen})
        return {"code_ppl": code, "general_ppl": gen}

    result = {"topk": topk}
    result["clean"] = await measure("clean")  # baseline on original weights
    for tag in ("spot", "random", "bottom"):
        await asyncio.to_thread(session.intervene, regions[tag], "zero", 0.0, "contrast")  # zero the region
        result[tag] = await measure(tag)
        await asyncio.to_thread(session.clear, "contrast")  # restore before the next condition
    await websocket.send_json({"type": "contrast_result", "model": model, **result})


async def _run_code_contrast(websocket, msg):
    """Code tab B: let the CURRENT model write code for a prompt and run it, then damage a region and do
    the same — side by side. clean = region cleared, deleted = region zeroed (own key, other knobs
    untouched). Streams code_contrast_progress per stage, ends with code_contrast. Assumes no other damage
    is applied to the clean baseline."""
    from parametic_studio.kernel.humaneval import run_program
    model = msg.get("model")
    session = await _target(msg)
    prompt, mt, temp = msg["prompt"], msg.get("max_tokens", 256), msg.get("temperature", 0.0)
    to = msg.get("timeout", 10.0)
    r, op, alpha = msg["region"], msg.get("op", "zero"), msg.get("alpha", 0.0)

    def gen_and_run():
        comp = session._greedy_complete(session.model, session.device, prompt, mt, temp)
        return {"completion": comp, "run": run_program(prompt + comp, to)}

    await asyncio.to_thread(session.clear, "codectr")  # clean baseline (this op's own knob only)
    clean = await asyncio.to_thread(gen_and_run)
    await websocket.send_json({"type": "code_contrast_progress", "model": model, "stage": "clean", **clean})
    region = await asyncio.to_thread(_resolve_region, session, r)
    await asyncio.to_thread(session.intervene, region, op, alpha, "codectr")
    deleted = await asyncio.to_thread(gen_and_run)
    await asyncio.to_thread(session.clear, "codectr")  # restore
    await asyncio.to_thread(session.free_memory)
    await websocket.send_json({"type": "code_contrast", "model": model, "clean": clean, "deleted": deleted})


async def _run_causal_sweep(websocket, msg):
    """Paper Table 1: sweep several tiny top-k% and, at each, zero the spot vs matched random/bottom
    controls (same count), measuring code + general PPL. Clean is measured once. Streams sweep_progress
    per (topk, cond), ends with sweep_result. Reuses the cached importance — compute a spot first."""
    model = msg.get("model")
    session = await _target(msg)
    topks = msg.get("topks") or [0.01]
    code_examples = msg["code_examples"]
    general_examples = msg.get("general_examples") or []
    seed = msg.get("seed", 0)

    if session.locate_cached(topks[0]) is None:  # None ⇔ no importance cached
        await websocket.send_json({"type": "error", "model": model, "op": "causal_sweep",
                                   "reason": "compute a spot first — no cached importance to build the spot/control masks from"})
        return

    async def measure():
        code = await asyncio.to_thread(session.ppl, code_examples)
        gen = await asyncio.to_thread(session.ppl, general_examples) if general_examples else None
        return {"code_ppl": code, "general_ppl": gen}

    clean = await measure()
    await websocket.send_json({"type": "sweep_progress", "model": model, "cond": "clean", "topk": None, **clean})
    rows = []
    for k in topks:
        def build(k=k):  # all three from the same cached importance — spot = top-k%, controls = matched size
            return {"spot": session.locate_cached(k),
                    "random": session.control_region("random", k, seed=seed),
                    "bottom": session.control_region("bottom", k)}
        regions = await asyncio.to_thread(build)
        for tag in ("spot", "random", "bottom"):
            region = regions[tag]
            if region is None:
                continue
            await asyncio.to_thread(session.intervene, region, "zero", 0.0, "sweep")  # delete the region
            m = await measure()
            await asyncio.to_thread(session.clear, "sweep")  # restore before the next condition
            row = {"topk": k, "cond": tag, **m}
            rows.append(row)
            await websocket.send_json({"type": "sweep_progress", "model": model, **row})
    await websocket.send_json({"type": "sweep_result", "model": model, "clean": clean, "topks": topks, "rows": rows})


def _resolve_spot_region(session, msg):
    """The spot mask a usage/activation op should measure: a named saved region if given, else the
    just-computed spot from the cache at the requested top-k%. None if neither is available."""
    name = msg.get("region_name")
    return session.get_region(name) if name else session.locate_cached(msg.get("topk", 0.01))


def _usage_rows(session, msg):
    """(spot_rows, None) for a usage op, reusing a per-session cache so repeated prompts/batches don't
    rebuild the full-model mask. (None, reason) if no spot/region is available. Cache key = the region
    source (named region, or spot+topk tagged by the current importance-cache identity)."""
    name = msg.get("region_name")
    key = ("named", name) if name else ("spot", msg.get("topk", 0.01), id(session._imp_cache))
    cache = getattr(session, "_usage_rows_cache", None)
    if cache is not None and cache[0] == key:
        return cache[1], None
    region = _resolve_spot_region(session, msg)
    if region is None:
        return None, "compute a spot first (or pick a saved region) — nothing to measure usage against"
    rows = session._spot_rows(region)
    session._usage_rows_cache = (key, rows)
    return rows, None


async def _run_region_usage(websocket, msg):
    """Per-token activation share of the spot while processing one prompt (Definition A). Ends with
    region_usage {overall, per_layer, tokens:[{text,usage}]}. Errors if no spot/region is available."""
    model = msg.get("model")
    session = await _target(msg)
    spot_rows, err = await asyncio.to_thread(_usage_rows, session, msg)
    if err:
        await websocket.send_json({"type": "error", "model": model, "op": "region_usage", "reason": err})
        return
    res = await asyncio.to_thread(session.region_usage, None, msg["prompt"], spot_rows)
    await websocket.send_json({"type": "region_usage", "model": model, "prompt": msg["prompt"], **res})


async def _run_region_usage_batch(websocket, msg):
    """Region usage for several prompts → the comparison table. Streams usage_row per prompt (overall
    only, no per-token — cheap), ends with usage_batch_done. Errors if no spot/region available."""
    model = msg.get("model")
    session = await _target(msg)
    spot_rows, err = await asyncio.to_thread(_usage_rows, session, msg)
    if err:
        await websocket.send_json({"type": "error", "model": model, "op": "region_usage_batch", "reason": err})
        return
    prompts = msg["prompts"]
    for i, prompt in enumerate(prompts):
        res = await asyncio.to_thread(session.region_usage, None, prompt, spot_rows)
        await websocket.send_json({"type": "usage_row", "model": model, "i": i, "total": len(prompts),
                                   "prompt": prompt, "overall": res["overall"]})
    await websocket.send_json({"type": "usage_batch_done", "model": model, "total": len(prompts)})


async def _run_training(websocket, msg):
    model = msg.get("model")
    session = await _target(msg)
    r = msg.get("region")
    region = await asyncio.to_thread(_resolve_region, session, r) if r else None
    steps = msg.get("steps", 50)
    mode = msg.get("mode", "full")
    gen = session.train_steps(msg["examples"], mode=mode, steps=steps,
                              lr=msg.get("lr", 1e-4), region=region, lora_dim=msg.get("lora_dim", 8))
    stopped = asyncio.Event()

    async def listen_stop():
        try:
            while True:
                m = await websocket.receive_json()
                if m.get("type") == "stop_train":
                    session.stop()
                    stopped.set()
                    return
        except WebSocketDisconnect:
            return

    TRAINING.add(model)
    listener = asyncio.create_task(listen_stop())
    count = 0
    try:
        while True:
            ev = await asyncio.to_thread(next, gen, None)  # one AdamW step off the loop
            if ev is None:
                break
            count += 1
            await websocket.send_json({"type": "train_step", "model": model, "step": ev["step"],
                                       "total": steps, "loss": ev["loss"]})
    except RuntimeError as e:  # e.g. "reset_training first"
        await websocket.send_json({"type": "error", "model": model, "op": "train", "reason": str(e)})
        return
    finally:
        TRAINING.discard(model)
        listener.cancel()
        try:
            await listener
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass
    await websocket.send_json({"type": "trained", "model": model, "mode": mode, "steps": count,
                               "reason": "stopped" if stopped.is_set() else "done"})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    token = os.environ.get("PARAMETIC_STUDIO_TOKEN")
    if token:  # remote kernel: gate the socket. localhost (no env) skips this entirely.
        try:
            first = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        if not (first.get("type") == "auth" and first.get("token") == token):
            await websocket.close(code=4401)  # like HTTP 401 — bad/missing auth frame
            return
    while True:
        try:
            msg = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        t = msg.get("type")
        if t == "auth":
            continue  # no-op: clients always send auth first when a token is set, even to a token-less kernel
        try:
            if t in _BUSY_OPS and msg.get("model") in TRAINING:
                await websocket.send_json({"type": "error", "model": msg.get("model"), "op": t, "reason": "busy: training"})
                continue
            await _dispatch(websocket, msg, t)
        except WebSocketDisconnect:
            return
        except Exception as e:  # any op's kernel error → report + keep the connection alive
            # A send to a client that already went away surfaces here as a RuntimeError (not
            # WebSocketDisconnect). Reporting on a dead socket re-raises and crashes the whole ASGI
            # task — so guard the report and just end the loop if the socket is gone.
            try:
                await websocket.send_json({"type": "error", "model": msg.get("model"), "op": t, "reason": str(e)})
            except Exception:
                return


async def _dispatch(websocket, msg, t):
        if t == "generate":
            await _run_generation(websocket, msg)
        elif t == "prompt_importance":
            await _run_prompt_importance(websocket, msg)
        elif t == "eval_code":
            await _run_eval_code(websocket, msg)
        elif t == "catalog":
            models = await asyncio.to_thread(_catalog_models)  # scan_cache_dir walks disk — off the loop
            await websocket.send_json({"type": "catalog", "models": models, "default": DEFAULT_MODEL})
        elif t == "installed_models":
            items = await asyncio.to_thread(_installed_models)
            await websocket.send_json({"type": "installed_models", "items": items})
        elif t == "open":
            model = msg["model"]
            await websocket.send_json({"type": "loading", "model": model})
            async def on_progress(done_mb, total_mb, pct):  # polled on the loop → send straight through
                await websocket.send_json({"type": "download_progress", "model": model,
                                           "done_mb": done_mb, "total_mb": total_mb, "pct": pct})

            async def on_stage(stage):  # download done → weights loading onto GPU (no % available)
                await websocket.send_json({"type": stage, "model": model})

            try:
                await _ensure(model, on_progress, msg.get("device"), on_stage)  # device? → that card; else freest
            except Exception as e:
                await websocket.send_json({"type": "load_failed", "model": model})
                await websocket.send_json({"type": "error", "model": model, "op": "open", "reason": str(e)})
                return
            await websocket.send_json({"type": "opened", "model": model})
        elif t == "delete_cached":
            await asyncio.to_thread(_delete_cached, msg["model"])  # disk cache only; SESSIONS untouched
            await websocket.send_json({"type": "cached_deleted", "model": msg["model"]})
        elif t == "get_config":
            cfg = await asyncio.to_thread(_read_config)
            await websocket.send_json({"type": "config", "config": cfg})
        elif t == "set_config":
            cfg = await asyncio.to_thread(_write_config, msg["config"])
            await websocket.send_json({"type": "config", "config": cfg})
        elif t == "close":
            global SESSION
            s = SESSIONS.pop(msg["model"], None)
            _locks.pop(msg["model"], None)
            if s is not None and hasattr(s, "close"):
                s.close()  # remove hooks
            if s is not None and s is SESSION:
                SESSION = None  # the default fallback must not pin a closed model in memory
            s = None            # drop the last reference *before* trimming the allocator
            import gc
            gc.collect()
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
            await websocket.send_json({"type": "closed", "model": msg["model"]})
        elif t == "drilldown":
            ph = (await _target(msg)).drilldown(msg["layer"])
            await websocket.send_json({"type": "perhead", "model": msg.get("model"), "layer": msg["layer"],
                                       "shape": list(ph.shape), "data": ph.cpu().tolist()})
        elif t == "spot":
            session = await _target(msg)
            examples = msg["examples"]
            stopped = asyncio.Event()

            async def listen_stop_spot():
                try:
                    while True:
                        m = await websocket.receive_json()
                        if m.get("type") == "stop_spot":
                            stopped.set()
                            return
                except WebSocketDisconnect:
                    return

            listener = asyncio.create_task(listen_stop_spot())
            emit = _spot_emitter(websocket, msg)
            extra_devices = msg.get("extra_devices")  # additional GPUs to help — see ModelSession._get_importance
            try:  # compute_spot keeps the per-param importance in the session cache, so a following
                  # save/knob reuses it (no second backward) — while still streaming the live heatmap.
                grid = await asyncio.to_thread(session.compute_spot, examples, emit, stopped.is_set, extra_devices)
            finally:
                listener.cancel()
                try:
                    await listener
                except (asyncio.CancelledError, WebSocketDisconnect):
                    pass
                await asyncio.to_thread(session.free_memory)  # spot backwards leave allocator-cached blocks
            await websocket.send_json({"type": "spotmap", "model": msg.get("model"),
                                       "reason": "stopped" if stopped.is_set() else "done", **grid})
        elif t == "intervene":
            session = await _target(msg)
            r = msg["region"]
            key = msg.get("key", r["kind"])
            emit = _locate_emitter(websocket, msg, "intervene") if r["kind"] == "spot" else None

            def _apply():  # clear this knob→locate→re-apply as one off-loop op (locate on this knob's original weights)
                session.clear(key)
                region = (session.locate_spot(r["examples"], r.get("topk", 0.01), progress=emit)
                          if r["kind"] == "spot" else _resolve_region(session, r))
                session.intervene(region, msg.get("op", "scale"), msg.get("alpha", 0.0), key=key)
                session.free_memory()  # spot-kind regions run backwards

            await asyncio.to_thread(_apply)
            await websocket.send_json({"type": "intervened", "model": msg.get("model"), "key": key})
        elif t == "save_region":
            session = await _target(msg)
            r = msg["region"]
            emit = _locate_emitter(websocket, msg, "save_region") if r["kind"] == "spot" else None

            def _save():
                if r["kind"] == "spot":  # keep the importance heatmap + per-param values (re-thresholdable)
                    base_topk = r.get("topk", 0.01)
                    cached = session.locate_cached(base_topk, return_grid=True)  # reuse the displayed spot — never recompute
                    if cached is not None:
                        region, grid = cached
                        acc = session._imp_cache["acc"]
                    else:  # no spot computed this session → fall back to locating from the examples
                        region, grid = session.locate_spot(r["examples"], base_topk, return_grid=True, progress=emit)
                        acc = session._get_importance(r["examples"])
                    importance = {n: acc[n][m.to(acc[n].device)].cpu() for n, m in region.items()}  # True vals, nonzero order
                    session.save_region(msg["name"], region, grid=grid, importance=importance, base_topk=base_topk)
                else:
                    region = _resolve_region(session, r)
                    session.save_region(msg["name"], region, grid=None)
                session.free_memory()
                return region

            region = await asyncio.to_thread(_save)
            count = sum(int(m.sum()) for m in region.values())
            await websocket.send_json({"type": "region_saved", "model": msg.get("model"), "name": msg["name"], "count": count})
        elif t == "import_region":
            session = await _target(msg)
            try:
                result = await asyncio.to_thread(session.import_region, msg["name"], msg["path"])
            except ValueError as e:
                await websocket.send_json({"type": "error", "model": msg.get("model"), "op": t, "reason": str(e)})
                return
            await websocket.send_json({"type": "region_saved", "model": msg.get("model"), "name": msg["name"],
                                       "count": result["count"], "imported": result["imported"], "skipped": result["skipped"]})
        elif t == "regions":
            session = await _target(msg)
            regs = await asyncio.to_thread(session.region_meta)  # legacy meta migration can hit disk hard — off the loop
            await websocket.send_json({"type": "regions", "model": msg.get("model"), "regions": regs})
        elif t == "delete_region":
            session = await _target(msg)
            session.delete_region(msg["name"])
            await websocket.send_json({"type": "regions", "model": msg.get("model"),
                                       "regions": await asyncio.to_thread(session.region_meta)})  # refreshed list = the ack
        elif t == "region_info":
            info = (await _target(msg)).region_grid(msg["name"], msg.get("topk"))
            await websocket.send_json({"type": "region_info", "model": msg.get("model"), "name": msg["name"], **info})
        elif t == "region_compare":
            cmp = await asyncio.to_thread((await _target(msg)).region_compare, msg["names"])
            await websocket.send_json({"type": "region_comparison", "model": msg.get("model"), "names": msg["names"], **cmp})
        elif t == "concentration":
            session = await _target(msg)
            curve = await asyncio.to_thread(session.concentration_curve, msg.get("points", 200))
            await websocket.send_json({"type": "concentration", "model": msg.get("model"),
                                       "curve": curve["points"] if curve else None,
                                       "total_params": curve["total_params"] if curve else None})
        elif t == "causal_contrast":
            await _run_causal_contrast(websocket, msg)
        elif t == "causal_sweep":
            await _run_causal_sweep(websocket, msg)
        elif t == "run_code":  # Code tab: run arbitrary python, capture output (off the loop — subprocess)
            from parametic_studio.kernel.humaneval import run_program
            res = await asyncio.to_thread(run_program, msg["source"], msg.get("timeout", 10.0))
            await websocket.send_json({"type": "code_result", "model": msg.get("model"), **res})
        elif t == "gen_code":  # Code tab: current model completes a code prompt (KV-cached fast path)
            session = await _target(msg)
            comp = await asyncio.to_thread(session._greedy_complete, session.model, session.device,
                                           msg["prompt"], msg.get("max_tokens", 256), msg.get("temperature", 0.0))
            await websocket.send_json({"type": "code_gen", "model": msg.get("model"), "prompt": msg["prompt"], "completion": comp})
        elif t == "code_contrast":
            await _run_code_contrast(websocket, msg)
        elif t == "region_usage":
            await _run_region_usage(websocket, msg)
        elif t == "region_usage_batch":
            await _run_region_usage_batch(websocket, msg)
        elif t == "tensors":
            ts = (await _target(msg)).tensor_list()
            await websocket.send_json({"type": "tensors", "model": msg.get("model"), "tensors": ts})
        elif t == "tensor_values":  # windowed slice of one parameter (weights | importance | region) — off the loop
            v = await asyncio.to_thread((await _target(msg)).tensor_values, msg["name"],
                                        msg.get("r0", 0), msg.get("c0", 0), msg.get("rows", 48), msg.get("cols", 48),
                                        msg.get("source", "weights"), msg.get("region"))
            await websocket.send_json({"type": "tensor_values", "model": msg.get("model"), **v})
        elif t == "datasets":  # kernel-side dataset store: <datasets_root> (files + symlinks)
            items = _list_datasets(_datasets_root())
            await websocket.send_json({"type": "datasets", "model": msg.get("model"), "items": items})
        elif t == "load_hf_dataset":
            await _load_hf_dataset(websocket, msg)
        elif t in ("read_dataset", "save_dataset", "delete_dataset", "link_path"):
            root = _datasets_root()
            try:
                if t == "read_dataset":
                    p = _dataset_path(root, msg["name"])
                    if not p.is_file():
                        raise ValueError(f"dataset not found: {msg['name']}")
                    await websocket.send_json({"type": "dataset_content", "model": msg.get("model"),
                                               "name": msg["name"], "content": p.read_text(errors="replace")})
                    return
                if t == "save_dataset":
                    p = _dataset_path(root, msg["name"])
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(msg["content"])
                elif t == "delete_dataset":
                    p = _dataset_path(root, msg["name"])
                    if not p.is_symlink():  # deleting *through* a linked folder would hit the user's real file
                        cur = root
                        for part in Path(msg["name"]).parts[:-1]:
                            cur = cur / part
                            if cur.is_symlink():
                                raise ValueError(f"'{msg['name']}' is inside linked folder '{Path(msg['name']).parts[0]}' — unlink the folder instead")
                    p.unlink(missing_ok=True)  # a link itself, or a real store file
                else:  # link_path: symlink an external file/folder into the store
                    src = Path(os.path.expanduser(msg["path"]))
                    if not src.exists():
                        raise ValueError(f"path not found: {src}")
                    dst = root / src.name
                    if not dst.exists():
                        os.symlink(src, dst)
                await websocket.send_json({"type": "dataset_saved", "model": msg.get("model"),
                                           "name": msg.get("name") or Path(msg.get("path", "")).name})
            except (ValueError, OSError) as e:
                await websocket.send_json({"type": "error", "model": msg.get("model"), "op": t, "reason": str(e)})
        elif t == "save_run":  # persist a generation result under runs/<model>; file IO off the loop
            rid = await asyncio.to_thread(_save_run, msg.get("model"), msg["run"])
            await websocket.send_json({"type": "run_saved", "model": msg.get("model"), "id": rid})
        elif t == "runs":
            items = await asyncio.to_thread(_list_runs, msg.get("model"))
            await websocket.send_json({"type": "runs", "model": msg.get("model"), "items": items})
        elif t == "load_run":
            rid = msg["id"]
            if not _run_id_ok(rid):
                await websocket.send_json({"type": "error", "model": msg.get("model"), "op": t, "reason": f"bad run id: {rid}"})
                return
            try:
                run = await asyncio.to_thread(_load_run, msg.get("model"), rid)
            except ValueError as e:
                await websocket.send_json({"type": "error", "model": msg.get("model"), "op": t, "reason": str(e)})
                return
            await websocket.send_json({"type": "run_data", "model": msg.get("model"), "id": rid, "run": run})
        elif t == "stats":
            import subprocess
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
            rss = float(out.strip() or 0) / 1024  # KB → MB
            session = SESSIONS.get(msg.get("model")) or SESSION
            cached = len(getattr(session, "regions", {})) if session is not None else 0
            await websocket.send_json({"type": "stats", "model": msg.get("model"),
                                       "rss_mb": rss, "models": len(SESSIONS), "regions_cached": cached})
        elif t == "gpus":
            await websocket.send_json(await asyncio.to_thread(_gpu_report))
        elif t == "clear":
            await asyncio.to_thread((await _target(msg)).clear, msg.get("key"))  # key → one knob; None → all. off the loop
            await websocket.send_json({"type": "cleared", "model": msg.get("model"), "key": msg.get("key")})
        elif t == "suspend":  # A/B: weights → original, knobs kept
            await asyncio.to_thread((await _target(msg)).suspend)
            await websocket.send_json({"type": "suspended", "model": msg.get("model")})
        elif t == "resume":
            await asyncio.to_thread((await _target(msg)).resume)
            await websocket.send_json({"type": "resumed", "model": msg.get("model")})
        elif t == "ppl":
            v = await asyncio.to_thread((await _target(msg)).ppl, msg["examples"])
            await websocket.send_json({"type": "ppl", "model": msg.get("model"), "tag": msg.get("tag"), "value": v})
        elif t == "train":
            await _run_training(websocket, msg)
        elif t == "reset_train":
            await asyncio.to_thread((await _target(msg)).reset_training)
            await websocket.send_json({"type": "train_reset", "model": msg.get("model")})
        elif t == "training_delta":  # how far weights moved during the last train, region vs rest (off the loop — walks params)
            session = await _target(msg)
            region = await asyncio.to_thread(_resolve_region, session, msg["region"]) if msg.get("region") else None
            res = await asyncio.to_thread(session.training_delta, region)
            await websocket.send_json({"type": "training_delta", "model": msg.get("model"), "delta": res})


def _watch_parent():
    """Die with the desktop app that spawned us (covers force-quit/crash, where kill() never runs)."""
    import threading
    import time
    ppid = os.getppid()

    def loop():
        while True:
            time.sleep(2)
            if os.getppid() != ppid:  # parent gone → we were reparented
                os._exit(0)

    threading.Thread(target=loop, daemon=True).start()


def serve(model_id, host="127.0.0.1", port=8000):
    import uvicorn

    from parametic_studio.kernel.model_session import ModelSession

    # windows: ppid는 부모 사망 후에도 불변 — 고아 감시는 posix 전용, RunEvent::Exit kill이 주 방어선
    if os.environ.get("PARAMETIC_STUDIO_PARENT_WATCH") == "1" and os.name == "posix":
        _watch_parent()

    global SESSION, DEFAULT_MODEL
    SESSION = ModelSession.from_pretrained(model_id)
    SESSIONS[model_id] = SESSION  # default model is addressable by id too
    DEFAULT_MODEL = model_id
    # Disable websocket keepalive pings: a long blocking op (eval, spot, sweep) runs one-at-a-time on a
    # connection and can delay pong handling past the default 20s ping timeout, so uvicorn would close the
    # socket mid-eval. That close makes the UI flash "kernel offline" and the reconnect resync then re-opens
    # models — whose requests queue behind the still-running op — freezing the tab. No server pings ⇒ no
    # spurious close; the frontend already reconnects on a genuinely dead socket.
    uvicorn.run(app, host=host, port=port, ws_ping_interval=None, ws_ping_timeout=None)


if __name__ == "__main__":
    # host/port overridable for remote serving — VESSL/exposed-port needs 0.0.0.0 (127.0.0.1
    # isn't reachable through the port proxy); ssh-tunnel path keeps the 127.0.0.1 default.
    serve(os.environ.get("PARAMETIC_STUDIO_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
          host=os.environ.get("PARAMETIC_STUDIO_HOST", "127.0.0.1"),
          port=int(os.environ.get("PARAMETIC_STUDIO_PORT", "8000")))
