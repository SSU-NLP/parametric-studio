import json
import math
import os
import re
from pathlib import Path

import torch

from parametic_studio.device import dtype_for, pick_device

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(?!.*lora_)(.+)$")  # lora_* params never enter spot/probe grids


def _gpt2_byte_decoder():
    """GPT-2 byte-level unicode→byte map. Some tokenizers' .decode() leaks the byte-level alphabet
    (Ġ=space, Ċ=newline, …) instead of reconstructing bytes; this inverts it. ponytail: standard
    GPT-2 table, built once."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _gpt2_byte_decoder()


class ModelSession:
    """A live inference session: model resident, manual token-by-token decode.

    Manual greedy loop (not TextIteratorStreamer) so each step can be stopped
    and, later, have attention captured. ponytail: greedy + eos + stop only.
    """

    @classmethod
    def from_pretrained(cls, model_id, device="auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = pick_device(device)  # "cuda:1" honored; "auto" lands on the freest card
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype_for(dev), attn_implementation="eager"
        )
        return cls(model, tok, dev, model_id=model_id)

    def __init__(self, model, tokenizer, device, model_id=None):
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.device = device
        self.model_id = model_id  # None (tests/anonymous) → no disk persistence
        self._stop = False
        self.last_raw = None  # per-layer [heads, kv] of the last decoded step (for drilldown)
        self._acts = {}       # (layer, module) -> last-token output norm
        self.regions = {}         # LRU cache: name -> region masks (lazy; loaded from disk on demand, cap 2)
        self.region_grids = {}    # name -> [L][M] importance grid (lazy from meta; None for legacy/cell regions)
        self._region_files = {}   # name -> Path(.pt) — every saved region (masks NOT loaded at boot)
        self._region_meta = {}    # name -> {"count": int, "grid": [[..]]|None} from the .meta.json sidecar
        self._region_order = []   # load order for LRU eviction of self.regions
        self._imp_cache = None    # {"key": <examples hash>, "acc": {param_name: CPU importance Tensor}} — cleared on train
        d = self._region_dir()
        if d and d.exists():
            for f in d.glob("*.pt"):  # scan only — masks (~0.36GB each) stay on disk until get_region
                self._region_files[f.stem] = f
                meta = f.with_suffix(".meta.json")
                if not meta.exists():
                    self._migrate_meta(f, meta)  # one-time: legacy .pt with no sidecar → compute + write, then free
                self._region_meta[f.stem] = json.loads(meta.read_text())
                self.region_grids[f.stem] = self._region_meta[f.stem].get("grid")
        self.leaf_modules_probed = self._leaf_modules()
        self.modules_probed = self._register_activation_hooks()

    def _migrate_meta(self, pt_path, meta_path):
        """Legacy .pt without a sidecar: load once → count/grid → write meta → drop the mask (1-time at boot)."""
        blob = torch.load(pt_path, map_location="cpu")
        if isinstance(blob, dict) and "masks" in blob:  # v2/v3: masks + grid (+ base_topk in v3)
            masks, grid = blob["masks"], blob.get("grid")
        else:  # v1: plain mask dict
            masks, grid = blob, None
        count = sum(int(v.sum()) for v in masks.values())
        meta = {"count": count, "grid": grid}
        if isinstance(blob, dict) and blob.get("base_topk") is not None:  # v3 keeps re-threshold reach
            meta["base_topk"] = blob["base_topk"]
        meta_path.write_text(json.dumps(meta))
        del blob, masks

    def free_memory(self):
        """Release transient device memory after heavy ops (the mps caching allocator hoards freed blocks)."""
        self.model.zero_grad(set_to_none=True)
        if self.device.type == "mps":
            torch.mps.empty_cache()

    def _region_dir(self):
        if not self.model_id:
            return None
        root = os.environ.get("PARAMETIC_STUDIO_HOME", os.path.expanduser("~/.parametic_studio"))
        return Path(root) / "regions" / self.model_id.replace("/", "__")

    def _register_activation_hooks(self):
        names = ["self_attn", "mlp"]
        self._hook_handles = []
        for i, layer in enumerate(self.model.model.layers):
            for name in names:
                self._hook_handles.append(getattr(layer, name).register_forward_hook(self._act_hook(i, name)))
        self._leaf_out = {}  # (layer, submodule path) -> full-sequence output tensor, for activation/spot grids
        seen = set()
        for i, layer in enumerate(self.model.model.layers):
            for path in self.leaf_modules_probed:  # weight+bias of the same leaf share one output
                if (i, path) in seen:
                    continue
                seen.add((i, path))
                target = layer
                for part in path.split("."):
                    target = getattr(target, part)
                self._hook_handles.append(target.register_forward_hook(self._leaf_hook(i, path)))
        return names

    def _leaf_hook(self, i, path):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out  # [b, seq, hidden|out_features]
            self._leaf_out[(i, path)] = h[0].detach()
        return hook

    def close(self):
        # remove hooks so the model (and this session) can be GC'd / freed.
        for h in getattr(self, "_hook_handles", []):
            h.remove()
        self._hook_handles = []

    def _act_hook(self, i, name):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out  # [b, seq, hidden]
            self._acts[(i, name)] = float(h[0].detach().float().norm(dim=-1).sum())
        return hook

    def _modules(self):
        return [m.group(2) for name, _ in self.model.named_parameters()
                if (m := _LAYER_RE.match(name)) and int(m.group(1)) == 0]

    def _leaf_modules(self):
        modules = []
        seen = set()
        for mod in self._modules():
            path = re.sub(r"\.(weight|bias)$", "", mod)
            if path not in seen:
                seen.add(path)
                modules.append(path)
        return modules

    def activation_detail_grid(self):
        """Fine activation grid for the latest forward pass: [layer, leaf module].

        Coarse activation shows only self_attn/mlp. This view exposes parameter-adjacent
        leaf modules such as q_proj, k_proj, gate_proj, and layer norms so the UI can
        drill into a clicked layer without another model pass.
        """
        L = len(self.model.model.layers)
        modules = self.leaf_modules_probed
        grid = [[0.0] * len(modules) for _ in range(L)]
        for l in range(L):
            for c, path in enumerate(modules):
                out = self._leaf_out.get((l, path))
                if out is not None:
                    grid[l][c] = float(out.detach().float().norm(dim=-1).sum())
        return {"layers": L, "modules": modules, "grid": grid}

    def spot_step(self, example, acc, n):
        """Accumulate |grad×param| for one example into acc (mutated); return running grid normalized by n.

        Split out of compute_spot so the API can stream one step at a time off the event loop.
        """
        ids = torch.tensor([self.tok.encode(example)], device=self.device)
        self.model.zero_grad(set_to_none=True)
        self.model(ids, labels=ids).loss.backward()
        for name, p in self.model.named_parameters():
            m = _LAYER_RE.match(name)
            if m and p.grad is not None:
                key = (int(m.group(1)), m.group(2))
                acc[key] = acc.get(key, 0.0) + float((p.grad * p.data).abs().sum())
        self.model.zero_grad(set_to_none=True)
        L, modules = len(self.model.model.layers), self._modules()
        grid = [[acc.get((l, mod), 0.0) / n for mod in modules] for l in range(L)]
        return {"layers": L, "modules": modules, "grid": grid}

    def _get_importance(self, examples, progress=None, should_stop=None, extra_devices=None):
        """Per-param accumulated |grad×param| (CPU tensors), cached per examples set.

        The importance is always relative to the *current* weights. A cache hit (same examples,
        weights unchanged since — train_steps/reset_training invalidate) returns without any backward.
        Miss: run one backward per example, accumulate on CPU, cache, return. should_stop() → cancel
        (caches what was accumulated so far so a resumed/repeat call still reuses it).

        extra_devices: additional CUDA devices ("cuda:1", ...) to help — examples are split across
        self.device + extra_devices and run concurrently on transient full-model replicas (freed
        when done). Multi-device runs skip the live per-cell heatmap (progress grid is None mid-run
        — safely merging partial cell sums across threads isn't worth it for a one-shot compute);
        the final grid (compute_spot's own reduce over the complete acc) is unaffected either way."""
        key = hash(tuple(examples))
        if self._imp_cache is not None and self._imp_cache["key"] == key:
            if progress:
                progress(len(examples), len(examples), None)  # UI: instant "done" without recomputing
            return self._imp_cache["acc"]
        devs = [str(self.device)] + [str(d) for d in (extra_devices or []) if str(d) != str(self.device)]
        if len(devs) <= 1:
            acc, done = self._importance_single(examples, progress, should_stop)
        else:
            acc, done = self._importance_parallel(examples, devs, progress, should_stop)
        if done:  # don't cache an empty (fully-cancelled) pass
            self._imp_cache = {"key": key, "acc": acc, "n": done}  # one set at a time — new examples replace the old
        return acc

    def _importance_single(self, examples, progress, should_stop):
        acc = {}
        total = len(examples)
        done = 0
        L, modules = len(self.model.model.layers), self._modules()  # for the running spot-view grid
        cell = {}
        for i, text in enumerate(examples):
            if should_stop and should_stop():
                break
            ids = torch.tensor([self.tok.encode(text)], device=self.device)
            self.model.zero_grad(set_to_none=True)
            self.model(ids, labels=ids).loss.backward()
            for name, p in self.model.named_parameters():
                m = _LAYER_RE.match(name)
                if m and p.grad is not None:
                    a = (p.grad * p.data).abs().cpu()  # CPU: acc is model-param-sized, don't hoard GPU memory
                    acc[name] = a if name not in acc else acc[name] + a
                    if progress:  # running per-cell sum for the live heatmap (cheap: one scalar per param)
                        k = (int(m.group(1)), m.group(2))
                        cell[k] = cell.get(k, 0.0) + float(a.sum())
            done = i + 1
            if progress:
                grid = [[cell.get((l, mod), 0.0) / done for mod in modules] for l in range(L)]
                progress(done, total, {"layers": L, "modules": modules, "grid": grid})
        self.model.zero_grad(set_to_none=True)
        return acc, done

    def _importance_parallel(self, examples, devs, progress, should_stop):
        """Split examples round-robin across devs; every device other than self.device gets a
        transient full-model replica (freed right after). torch releases the GIL during CUDA
        kernels, so these threads genuinely overlap compute across GPUs, not just interleave on one."""
        import copy
        import threading
        chunks = {d: examples[i::len(devs)] for i, d in enumerate(devs)}
        replicas = {d: (self.model if d == str(self.device) else copy.deepcopy(self.model).to(d)) for d in devs}
        lock = threading.Lock()
        counter = [0]
        total = len(examples)

        def report():
            with lock:
                counter[0] += 1
                n = counter[0]
            if progress:
                progress(n, total, None)  # no live grid in multi-device mode — see docstring above

        results = {}

        def run(d):
            acc, done = {}, 0
            for text in chunks[d]:
                if should_stop and should_stop():
                    break
                ids = torch.tensor([self.tok.encode(text)], device=d)
                model = replicas[d]
                model.zero_grad(set_to_none=True)
                model(ids, labels=ids).loss.backward()
                for name, p in model.named_parameters():
                    m = _LAYER_RE.match(name)
                    if m and p.grad is not None:
                        a = (p.grad * p.data).abs().cpu()
                        acc[name] = a if name not in acc else acc[name] + a
                model.zero_grad(set_to_none=True)
                done += 1
                report()
            results[d] = (acc, done)

        threads = [threading.Thread(target=run, args=(d,)) for d in devs if chunks[d]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for d, replica in replicas.items():
            if replica is not self.model:
                del replica
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
        acc, done = {}, 0
        for d in devs:
            partial, n = results.get(d, ({}, 0))
            done += n
            for name, val in partial.items():
                acc[name] = val if name not in acc else acc[name] + val
        return acc, done

    def compute_spot(self, examples, progress=None, should_stop=None, extra_devices=None):
        """Coding-Spot importance: accumulate |grad×param| over examples, reduced to [layer, module].
        Populates the per-param importance cache so a following save/knob reuses it (no recompute).
        extra_devices: additional GPUs to help — see _get_importance."""
        L, modules = len(self.model.model.layers), self._modules()
        if not examples:  # no examples → all-zero grid
            return {"layers": L, "modules": modules, "grid": [[0.0] * len(modules) for _ in range(L)]}
        acc = self._get_importance(examples, progress=progress, should_stop=should_stop, extra_devices=extra_devices)
        n = self._imp_cache["n"] if self._imp_cache else len(examples)
        cell = {}
        for name, score in acc.items():
            m = _LAYER_RE.match(name)
            cell[(int(m.group(1)), m.group(2))] = cell.get((int(m.group(1)), m.group(2)), 0.0) + float(score.sum()) / n
        grid = [[cell.get((l, mod), 0.0) for mod in modules] for l in range(L)]
        return {"layers": L, "modules": modules, "grid": grid}

    def spot_activation_grid(self, region):
        """[L, M] grid: mean |output value| over all current sequence tokens at neurons touched by an already-identified
        spot region — NOT the whole module (that's the coarse 'activation' probe). 2D weight masks
        reduce to output rows (any selected input in that row); 1D weight/bias masks index the output
        directly. 0.0 where a module has no selected weights or hasn't produced output yet this run."""
        L, modules = len(self.model.model.layers), self._modules()
        grid = [[0.0] * len(modules) for _ in range(L)]
        for name, mask in region.items():
            m = _LAYER_RE.match(name)
            if not m or not mask.any():
                continue
            l, mod = int(m.group(1)), m.group(2)
            if mod not in modules:
                continue
            path = re.sub(r"\.(weight|bias)$", "", mod)
            out = self._leaf_out.get((l, path))
            if out is None:
                continue
            sel = mask.any(dim=1) if mask.dim() == 2 else mask  # 2D: row touched by any selected weight
            vals = out[:, sel.to(out.device)]
            if vals.numel():
                grid[l][modules.index(mod)] = float(vals.abs().mean())
        return {"layers": L, "modules": modules, "grid": grid}

    def _spot_rows(self, region):
        """(layer, module) -> bool row mask over output features (Definition A helper). 2D weight →
        rows any selected weight touches; 1D weight/bias → the mask itself. Skips empty/foreign params."""
        modules = self._modules()
        rows = {}
        for name, mask in region.items():
            m = _LAYER_RE.match(name)
            if not m or not mask.any():
                continue
            l, mod = int(m.group(1)), m.group(2)
            if mod in modules:
                rows[(l, mod)] = mask.any(dim=1) if mask.dim() == 2 else mask
        return rows

    def region_usage(self, region, prompt, spot_rows=None):
        """Activation share (Definition A) of `region` while processing `prompt`, per token:
        usage_t = Σ|act| over the spot's output rows / Σ|act| over ALL rows in those modules, at token t.
        One forward pass; temporary hooks capture every position (not just the last). Returns
        {overall, per_layer: [L], tokens: [{text, usage}]}. A code-bearing prompt lights the coding
        spot (high share); unrelated prose barely uses it (low share).

        spot_rows (optional): precomputed {(layer,module): row mask} to reuse across many prompts —
        deriving it from a full-model mask is the expensive part, so a batch computes it once."""
        L = len(self.model.model.layers)
        if spot_rows is None:
            spot_rows = self._spot_rows(region)
        if not spot_rows:
            return {"overall": 0.0, "per_layer": [0.0] * L, "tokens": []}
        needed = {(l, re.sub(r"\.(weight|bias)$", "", mod)) for (l, mod) in spot_rows}
        captured, handles = {}, []

        def mk(l, path):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[(l, path)] = h[0].detach()  # [seq, features] — all positions this time
            return hook
        for (l, path) in needed:
            target = self.model.model.layers[l]
            for part in path.split("."):
                target = getattr(target, part)
            handles.append(target.register_forward_hook(mk(l, path)))
        ids = torch.tensor([self.tok.encode(prompt)], device=self.device)
        try:
            with torch.no_grad():
                self.model(ids)
        finally:
            for h in handles:
                h.remove()
        seq = ids.shape[1]
        num, den = torch.zeros(seq), torch.zeros(seq)
        lnum, lden = {}, {}
        for (l, mod), rows in spot_rows.items():
            out = captured.get((l, re.sub(r"\.(weight|bias)$", "", mod)))
            if out is None:
                continue
            a = out.abs().float().cpu()                # [seq, features]
            spot_sum = a[:, rows.to(a.device)].sum(dim=1)  # [seq]
            all_sum = a.sum(dim=1)                      # [seq]
            num += spot_sum
            den += all_sum
            lnum[l] = lnum.get(l, 0.0) + float(spot_sum.sum())
            lden[l] = lden.get(l, 0.0) + float(all_sum.sum())
        usage = (num / den.clamp(min=1e-9)).tolist()
        toks = [self.tok.decode([t]) for t in ids[0].tolist()]
        tokens = [{"text": toks[i], "usage": usage[i]} for i in range(seq)]
        per_layer = [(lnum.get(l, 0.0) / lden[l]) if lden.get(l) else 0.0 for l in range(L)]
        overall = float(num.sum() / den.sum().clamp(min=1e-9))
        return {"overall": overall, "per_layer": per_layer, "tokens": tokens}

    # ---- knob (B1): Locate × Edit × Evaluate. A Region is {param_name: bool mask}. ----

    def locate_cell(self, layer, module):
        """Locate: a whole per-layer weight, e.g. (0, 'mlp.gate_proj.weight'). Masks live on CPU."""
        name = f"model.layers.{layer}.{module}"
        p = dict(self.model.named_parameters())[name]
        return {name: torch.ones(p.shape, dtype=torch.bool)}

    def _region_from_acc(self, acc, topk, n, return_grid):
        # ponytail: per-param exact top-k via topk indices — no tie over-selection, no global flatten copy.
        # acc lives on CPU (model-sized), but topk over a 7B model is slow on CPU — do it per-param on the
        # model's device (transient: one param tensor at a time), masks come back to CPU.
        dev = self.device
        region = {}
        for name, score in acc.items():
            flat = score.flatten().to(dev, non_blocking=True)
            count = int(topk * flat.numel())
            if count <= 0:
                continue
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[torch.topk(flat, count, largest=True).indices] = True
            region[name] = mask.reshape(score.shape).cpu()  # masks live on CPU; ops move them per use
            del flat, mask  # free the device copy before the next param
        if not return_grid:
            return region
        L, modules = len(self.model.model.layers), self._modules()  # per-cell importance sums — spot heatmap numbers
        cell = {}
        for name, score in acc.items():
            m = _LAYER_RE.match(name)
            cell[(int(m.group(1)), m.group(2))] = float(score.sum()) / max(n, 1)
        grid = [[cell.get((l, mod), 0.0) for mod in modules] for l in range(L)]
        return region, grid

    def locate_cached(self, topk=0.01, return_grid=False):
        """Threshold the LAST computed importance (the displayed spot) — no examples, no backward.
        Returns None if nothing is cached. This is what save/knob use so they never recompute the
        spot just because the caller's example list didn't hash-match the compute call."""
        if self._imp_cache is None:
            return None
        return self._region_from_acc(self._imp_cache["acc"], topk, self._imp_cache.get("n", 1), return_grid)

    def control_region(self, kind, topk=0.01, seed=0):
        """Matched control mask from the cached importance: same per-param count as the top-k% spot,
        but positions chosen by `kind` — 'random' (seeded uniform) or 'bottom' (lowest importance).
        This is the causal control from the paper: damage an equal-size region that is NOT the spot.
        Returns None if no spot was computed this session."""
        if self._imp_cache is None:
            return None
        acc = self._imp_cache["acc"]
        gen = torch.Generator().manual_seed(seed)
        region = {}
        for name, score in acc.items():
            flat = score.flatten()
            count = int(topk * flat.numel())
            if count <= 0:
                continue
            mask = torch.zeros(flat.numel(), dtype=torch.bool)
            if kind == "random":
                idx = torch.randperm(flat.numel(), generator=gen)[:count]
            elif kind == "bottom":
                idx = torch.topk(flat, count, largest=False).indices  # least-important positions
            else:
                raise ValueError(f"unknown control kind: {kind}")
            mask[idx] = True
            region[name] = mask.reshape(score.shape)
        return region

    def concentration_curve(self, points=200):
        """Lorenz/CDF of the cached per-param importance over ALL parameters: how concentrated is the
        signal. Returns {points: [[frac_params, frac_importance], ...], total_params}. frac_params is
        the top X fraction (by importance, descending); frac_importance is the cumulative share they
        hold. A steep early rise = a few weights carry most of the importance. None if no spot cached."""
        if self._imp_cache is None:
            return None
        acc = self._imp_cache["acc"]
        vals = torch.cat([score.flatten().float() for score in acc.values()])
        n = vals.numel()
        total = float(vals.sum()) or 1.0
        sorted_desc, _ = torch.sort(vals, descending=True)
        csum = torch.cumsum(sorted_desc, dim=0)
        # downsample to `points` log-ish steps so the steep head is well resolved without shipping n values
        xs = sorted(set(int(round(n * (i / points) ** 0.5)) for i in range(1, points + 1)) | {n})
        curve = [[k / n, float(csum[min(k, n) - 1]) / total] for k in xs if k >= 1]
        return {"points": [[0.0, 0.0]] + curve, "total_params": n}

    def locate_spot(self, examples, topk=0.01, return_grid=False, progress=None, extra_devices=None):
        """Locate: top-k% mask *within each layer param* by accumulated |grad×param|
        (per-param, not a global threshold).
        return_grid=True also returns the [L][M] importance cell grid (what the spot view shows).
        progress(i, total): called after each example's backward+accumulate (i is 1-based).
        extra_devices: additional GPUs to help — see _get_importance."""
        if not examples:
            return ({}, []) if return_grid else {}
        acc = self._get_importance(examples, progress=progress, extra_devices=extra_devices)  # cached: no backward on a repeat examples set
        result = self._region_from_acc(acc, topk, len(examples), return_grid)
        # acc is the CPU-resident cache — keep it so a %-only change re-thresholds without another backward.
        self.free_memory()  # release backward leftovers on the device (cache stays on CPU)
        return result

    def tensor_list(self):
        """Metadata of every parameter (HF safetensors-viewer style): name, shape, dtype."""
        return [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype).removeprefix("torch.")}
                for n, p in self.model.named_parameters()]

    def _region_dense(self, region_name, param_name):
        """Dense param-shaped grid of a saved region: the stored |g×w| importance at each selected
        position, 0 everywhere else (legacy regions with no importance → 1.0 at selected positions =
        a plain mask). Lets the tensor inspector show WHERE the spot sits inside a weight. Raises if
        the region or the tensor-within-region is missing."""
        if not region_name or region_name not in self._region_files:
            raise ValueError(f"region not found: {region_name}")
        blob = torch.load(self._region_files[region_name], map_location="cpu")
        masks = blob["masks"] if isinstance(blob, dict) and "masks" in blob else blob
        if param_name not in masks:
            raise ValueError(f"'{param_name}' is not selected in region '{region_name}'")
        mask = masks[param_name].to(torch.bool)
        idx = mask.reshape(-1).nonzero(as_tuple=True)[0]  # True positions, flattened row-major
        imp = blob.get("importance", {}).get(param_name) if isinstance(blob, dict) else None
        dense = torch.zeros(mask.numel(), dtype=torch.float32)
        dense[idx] = imp.float() if imp is not None else 1.0  # importance aligns to mask.nonzero() order
        return dense.reshape(mask.shape)

    def tensor_values(self, name, r0=0, c0=0, rows=48, cols=48, source="weights", region=None):
        """A windowed slice of one parameter as a 2D grid, for the interactive tensor inspector. A single
        weight is millions of values — never ship the whole thing: return at most rows×cols (hard-capped)
        from offset (r0, c0). 1D params (norms/biases) render as one row; >2D params are flattened to
        [dim0, -1]. Stats are over the FULL tensor so the view shows the real distribution, not just the
        window.

        source selects WHICH values to show at each position of the same tensor:
          - "weights"    : the model's actual parameter values θ (signed). Reflects applied knob/damage.
          - "importance" : the live cached |grad×weight| from the last spot / prompt-importance (≥0).
          - "region"     : a saved region's stored importance, 0 outside the spot (≥0) — see _region_dense.
        """
        params = dict(self.model.named_parameters())
        if name not in params:
            raise ValueError(f"unknown tensor: {name}")
        p = params[name]
        shape = list(p.shape)
        signed = source == "weights"
        if source == "weights":
            t = p.detach().float().cpu()
        elif source == "importance":
            if self._imp_cache is None:
                raise ValueError("no importance yet — compute a spot or run prompt importance first")
            acc = self._imp_cache["acc"]
            if name not in acc:
                raise ValueError(f"'{name}' has no importance (only transformer-layer weights are scored)")
            t = acc[name].float().cpu()
        elif source == "region":
            t = self._region_dense(region, name)
        else:
            raise ValueError(f"unknown source: {source}")
        flat2d = t.reshape(1, -1) if t.dim() <= 1 else (t if t.dim() == 2 else t.reshape(shape[0], -1))
        R, C = flat2d.shape
        rows, cols = min(int(rows), 128), min(int(cols), 128)  # hard cap on payload
        r0 = max(0, min(int(r0), max(0, R - 1)))
        c0 = max(0, min(int(c0), max(0, C - 1)))
        window = flat2d[r0:r0 + rows, c0:c0 + cols]
        return {"name": name, "shape": shape, "dtype": str(p.dtype).removeprefix("torch."),
                "source": source, "signed": signed,
                "rows_total": R, "cols_total": C, "r0": r0, "c0": c0,
                "values": window.tolist(), "flattened": t.dim() > 2,
                "stats": {"min": float(t.min()), "max": float(t.max()), "mean": float(t.mean()),
                          "std": float(t.std()), "absmax": float(t.abs().max())}}

    def _cache_region(self, name, region):
        """Insert into the LRU mask cache (cap 2) — evict the oldest *reloadable* region past the cap.
        Memory-only regions (no model_id → no backing .pt) are never evicted: there's nowhere to reload them."""
        if name in self.regions:
            self._region_order.remove(name)
        self.regions[name] = region
        self._region_order.append(name)
        evictable = [n for n in self._region_order if n in self._region_files]
        while len(evictable) > 2:  # cap 2: region masks are big; hold at most two disk-backed resident
            victim = evictable.pop(0)
            self._region_order.remove(victim)
            self.regions.pop(victim, None)

    def get_region(self, name, topk=None):
        """Return a saved region's masks, loading from disk (v1/v2/v3) on cache miss. LRU-cached (cap 2).

        topk (v3 only): re-threshold to a *smaller* top-k% without recomputing. A saved region at
        base_topk holds the union of every tighter %, so keeping the top new_count by the stored
        importance in each param yields exactly the fresh top-topk% mask. topk None → full saved mask;
        topk > base_topk (or no importance) → full saved mask (can't grow past what was saved)."""
        if topk is None and name in self.regions:
            return self.regions[name]
        blob = torch.load(self._region_files[name], map_location="cpu") if name in self._region_files else None
        if blob is None:  # memory-only region (no backing .pt): full mask, no re-threshold data
            return self.regions[name]
        region = blob["masks"] if isinstance(blob, dict) and "masks" in blob else blob  # v3/v2 | v1
        if topk is not None and isinstance(blob, dict) and "importance" in blob:
            base = blob.get("base_topk")
            if base is not None and topk <= base:
                region = self._rethreshold(region, blob["importance"], base, topk)
                return region  # derived mask — don't pollute the LRU cache of full masks
        if topk is None:
            self._cache_region(name, region)
        return region

    def _rethreshold(self, region, importance, base, topk):
        """Per-param: keep the top new_count of the saved base-mask by stored importance → top-topk% mask."""
        out = {}
        for pname, mask in region.items():
            new_count = max(1, round(mask.numel() * topk))          # min 1: never empty a selected param
            idx = mask.nonzero(as_tuple=True)[0] if mask.dim() == 1 else mask.reshape(-1).nonzero(as_tuple=True)[0]
            imp = importance[pname]                                  # 1D, aligned to mask.nonzero() order
            new_count = min(new_count, imp.numel())                 # can't keep more than base selected
            keep = torch.topk(imp, new_count, largest=True).indices  # tightest new_count of the base set
            new_flat = torch.zeros(mask.numel(), dtype=torch.bool)
            new_flat[idx[keep]] = True
            out[pname] = new_flat.reshape(mask.shape)
        return out

    def import_region(self, name, dir_path):
        """Import an external mask directory (one bool-tensor .pt per parameter, filename =
        param name) as a Studio region under `name`. No grid/importance (imported masks don't
        carry them) — the
        region still works for intervene/activations-spot, just without the spot-view heatmap or
        re-threshold-by-%. Raises ValueError if the directory has no .pt matching this model's params
        (e.g. wrong model loaded, or an empty/bad path) so the caller sees a clear error, not a silent no-op."""
        d = Path(os.path.expanduser(str(dir_path)))
        if not d.is_dir():
            raise ValueError(f"not a directory: {d}")
        params = dict(self.model.named_parameters())
        region, skipped = {}, []
        for f in sorted(d.glob("*.pt")):
            pname = f.stem
            if pname not in params:
                skipped.append(f.name)
                continue
            mask = torch.load(f, map_location="cpu")
            if mask.shape != params[pname].shape:
                skipped.append(f.name)  # shape mismatch — almost certainly the wrong model
                continue
            region[pname] = mask.to(torch.bool)
        if not region:
            raise ValueError(f"no matching parameter masks found in {d} for this model")
        self.save_region(name, region)
        return {"count": sum(int(m.sum()) for m in region.values()), "imported": len(region), "skipped": skipped}

    def region_meta(self):
        """Saved-region list without loading masks: [{name, count, base_topk?}] from the meta sidecars.
        base_topk (v3) is the upper bound for re-thresholding a saved spot to a smaller %."""
        return [{"name": n, "count": m.get("count", 0), "base_topk": m.get("base_topk")}
                for n, m in self._region_meta.items()]

    def save_region(self, name, region, grid=None, importance=None, base_topk=None):
        """Locate: keep a named mask (+ its importance grid, for faithful viewing) in the workspace and on disk.

        importance (v3): {param_name: 1D float tensor} — the |grad×param| value at each mask-True
        position, ordered to match mask.nonzero(). Persisted only in the .pt (per-param, too big for
        the meta sidecar). With base_topk it lets get_region re-threshold to any smaller top-k%."""
        self._cache_region(name, region)
        self.region_grids[name] = grid
        count = sum(int(v.sum()) for v in region.values())
        meta = {"count": count, "grid": grid}
        if importance is not None and base_topk is not None:
            meta["base_topk"] = base_topk  # importance itself stays in the .pt — per-param, large
        self._region_meta[name] = meta
        d = self._region_dir()
        if d:
            d.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^\w.-]", "_", name)
            pt = d / f"{safe}.pt"
            blob = {"masks": {k: v.cpu() for k, v in region.items()}, "grid": grid}
            if importance is not None and base_topk is not None:  # v3
                blob["importance"] = {k: v.cpu() for k, v in importance.items()}
                blob["base_topk"] = base_topk
            # ponytail: bool dump ≈ 1 byte/param (~0.4GB for a 0.5B full region) — pack indices if disk matters.
            torch.save(blob, pt)
            (d / f"{safe}.meta.json").write_text(json.dumps(meta))
            self._region_files[name] = pt

    def delete_region(self, name):
        """Remove a saved region from the workspace and disk (no-op if unknown)."""
        self.regions.pop(name, None)
        if name in self._region_order:
            self._region_order.remove(name)
        self.region_grids.pop(name, None)
        self._region_meta.pop(name, None)
        self._region_files.pop(name, None)
        d = self._region_dir()
        if d:
            safe = re.sub(r"[^\w.-]", "_", name)
            (d / f"{safe}.pt").unlink(missing_ok=True)
            (d / f"{safe}.meta.json").unlink(missing_ok=True)

    def _saved_importance_grid(self, masks, importance, topk=None):
        L, modules = len(self.model.model.layers), self._modules()
        cell = {}
        for pname, mask in masks.items():
            imp = importance.get(pname) if isinstance(importance, dict) else None
            if imp is None:
                continue
            vals = imp
            if topk is not None:
                count = min(max(1, round(mask.numel() * topk)), vals.numel())
                vals = torch.topk(vals, count, largest=True).values
            m = _LAYER_RE.match(pname)
            if m:
                key = (int(m.group(1)), m.group(2))
                cell[key] = cell.get(key, 0.0) + float(vals.sum())
        return [[cell.get((l, mod), 0.0) for mod in modules] for l in range(L)]

    def region_grid(self, name, topk=None):
        """Visualize a saved region: selection-fraction grid + the importance grid captured at save time."""
        blob = torch.load(self._region_files[name], map_location="cpu") if name in self._region_files else None
        effective_topk = topk
        importance_grid = self.region_grids.get(name)
        if isinstance(blob, dict) and "masks" in blob and "importance" in blob:
            base = blob.get("base_topk")
            if effective_topk is not None and (base is None or effective_topk > base):
                effective_topk = None
            importance_grid = self._saved_importance_grid(blob["masks"], blob["importance"], effective_topk)
        region = self.get_region(name, effective_topk)
        L, modules = len(self.model.model.layers), self._modules()
        return {"layers": L, "modules": modules, "grid": self._cell_grid(region, L, modules),
                "importance": importance_grid,
                "base_topk": self._region_meta.get(name, {}).get("base_topk"),
                "view_topk": effective_topk,
                "count": sum(int(v.sum()) for v in region.values())}

    def _cell_grid(self, region, L, modules):
        grid = [[0.0] * len(modules) for _ in range(L)]
        for pname, mask in region.items():
            m = _LAYER_RE.match(pname)
            if m and m.group(2) in modules:
                grid[int(m.group(1))][modules.index(m.group(2))] = float(mask.sum()) / mask.numel()
        return grid

    def region_compare(self, names):
        """Cross-dataset spot comparison: per-region cell grids, all-intersection grid, pairwise Jaccard."""
        from itertools import combinations
        L, modules = len(self.model.model.layers), self._modules()
        regions = {n: self.get_region(n) for n in names}  # local — may exceed LRU cap for the compare only
        # display grids: prefer the saved importance heatmap (what the spot view showed) — the
        # selection-fraction grid is ~uniform by construction (per-param top-k%).
        grids = {n: self.region_grids.get(n) or self._cell_grid(r, L, modules) for n, r in regions.items()}
        kinds = {n: ("importance" if self.region_grids.get(n) else "fraction") for n in names}
        jaccard = {}
        for a, b in combinations(names, 2):
            inter = union = 0
            for p in set(regions[a]) | set(regions[b]):
                ma, mb = regions[a].get(p), regions[b].get(p)
                if ma is None or mb is None:
                    union += int((ma if mb is None else mb).sum())
                else:
                    inter += int((ma & mb).sum())
                    union += int((ma | mb).sum())
            jaccard[f"{a}|{b}"] = inter / union if union else 0.0
        inter_region = {}
        for p in set.intersection(*(set(r) for r in regions.values())) if regions else set():
            acc = regions[names[0]][p]
            for n in names[1:]:
                acc = acc & regions[n][p]
            inter_region[p] = acc
        inter_grid = self._cell_grid(inter_region, L, modules)
        # lift = observed ∩-fraction / expected under independence (product of per-region fractions).
        # top-k% spots make raw fractions near-uniform and tiny — lift is the readable signal.
        frac = {n: self._cell_grid(r, L, modules) for n, r in regions.items()}
        lift = [[0.0] * len(modules) for _ in range(L)]
        for l in range(L):
            for c in range(len(modules)):
                expected = 1.0
                for n in names:
                    expected *= frac[n][l][c]
                lift[l][c] = inter_grid[l][c] / expected if expected > 0 else 0.0
        return {"layers": L, "modules": modules, "grids": grids, "kinds": kinds,
                "intersection": inter_grid, "intersection_lift": lift, "jaccard": jaccard}

    def intervene(self, region, op="scale", alpha=0.0, key="default"):
        """Edit: reversibly modify selected weights under `key`. op = scale|zero|mean|random.

        Multiple keys coexist (a mixing board of knobs). Re-using a key re-applies from the
        original weights. ponytail: assumes knobs target disjoint regions (per-cell); overlapping
        regions get last-write-wins on restore.
        """
        active = getattr(self, "_active", {})
        if key in active:
            self._restore(active.pop(key))  # re-adjust: undo this knob first, apply from original
        params = dict(self.model.named_parameters())
        backup = {}
        with torch.no_grad():
            for name, cmask in region.items():
                p = params[name]
                mask = cmask.to(p.device)
                sel = p.data[mask]
                backup[name] = sel.clone()  # only selected values → memory ∝ |region|
                if op == "scale":
                    p.data[mask] = sel * alpha
                elif op == "zero":
                    p.data[mask] = 0.0
                elif op == "mean":
                    p.data[mask] = sel.mean()
                elif op == "random":
                    p.data[mask] = torch.randn_like(sel) * sel.std() + sel.mean()
                else:
                    raise ValueError(f"unknown op: {op}")
        active[key] = (region, backup, op, alpha)
        self._active = active

    def _restore(self, entry):
        region, backup = entry[0], entry[1]
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, mask in region.items():
                params[name].data[mask.to(params[name].device)] = backup[name]

    def suspend(self):
        """A/B: restore original weights but keep knob entries (resume re-applies)."""
        for entry in getattr(self, "_active", {}).values():
            self._restore(entry)

    def resume(self):
        """A/B: re-apply all suspended knobs from the original weights.
        ponytail: op='random' redraws on resume — a new sample, not the identical one."""
        for key, (region, _b, op, alpha) in list(getattr(self, "_active", {}).items()):
            self.intervene(region, op, alpha, key=key)

    def clear(self, key=None):
        """Edit: restore weights. key → that knob only; None → all knobs (no-op if none)."""
        active = getattr(self, "_active", {})
        if key is None:
            for entry in active.values():  # disjoint regions → restore order-independent
                self._restore(entry)
            self._active = {}
        elif key in active:
            self._restore(active.pop(key))

    # ---- region-aware training: full | spot-freeze | spot-only | lora. Reversible via reset_training. ----

    def train_steps(self, examples, mode="full", steps=50, lr=1e-4, region=None, lora_dim=8):
        """Generator: one AdamW step per next(), yields {"step", "loss"}. stop() ends early.

        spot-freeze = gradients blocked inside region (train everything *except* the spot).
        spot-only   = only region weights train (element-masked). lora = base frozen, adapters train.
        """
        if getattr(self, "_trained", None) is not None:
            self.reset_training()  # a previous training is still applied → restore the clean base and start fresh
        self.clear()  # invariant: knob backups are always relative to the current base weights
        self._stop = False
        params = dict(self.model.named_parameters())
        hooks = []
        if mode == "lora":
            from parametic_studio.kernel.lora import convert_to_lora
            convert_to_lora(self.model, lora_dim)
            for n, p in self.model.named_parameters():
                p.requires_grad_("lora_" in n)     # adapters are the *only* trainables (embed/norm/lm_head too)
            self._trained = ("lora", None)         # base weights never change → nothing to back up
        elif mode == "spot-freeze":
            for name, mask in region.items():
                dm = mask.to(params[name].device)  # hook runs on the grad's device
                hooks.append(params[name].register_hook(lambda g, m=dm: g.masked_fill(m, 0)))
            self._trained = (mode, {n: p.detach().to("cpu", copy=True) for n, p in params.items()})
        elif mode == "spot-only":
            for n, p in params.items():
                p.requires_grad_(n in region)
            for name, mask in region.items():
                dm = mask.to(params[name].device)
                hooks.append(params[name].register_hook(lambda g, m=dm: g * m))
            self._trained = (mode, (region, {n: params[n].data[m.to(params[n].device)].clone() for n, m in region.items()}))
        else:  # full
            self._trained = ("full", {n: p.detach().to("cpu", copy=True) for n, p in params.items()})
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        # weight_decay must be 0: decoupled decay moves weights even where the gradient is masked to zero,
        # which would silently violate spot-freeze/spot-only region guarantees.
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
        try:
            for step in range(steps):
                if self._stop:
                    return
                ids = torch.tensor([self.tok.encode(examples[step % len(examples)])], device=self.device)
                loss = self.model(ids, labels=ids).loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)  # bf16 NaN insurance
                opt.step()
                yield {"step": step, "loss": float(loss.detach())}
        finally:
            self._imp_cache = None  # weights moved → cached importance is stale
            for h in hooks:
                h.remove()  # leaked hooks would silently corrupt every later spot map
            self.model.zero_grad(set_to_none=True)
            del opt
            for p in self.model.parameters():
                p.requires_grad_(True)  # locate_spot needs grads everywhere
            if self.device.type == "mps":
                torch.mps.empty_cache()  # release optimizer state

    def reset_training(self):
        """Restore pre-training weights (no-op if not trained)."""
        trained = getattr(self, "_trained", None)
        if trained is None:
            return
        self._imp_cache = None  # weights restored → cached importance is stale
        self.clear()  # same invariant as train_steps
        mode, payload = trained
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            if mode == "lora":
                from parametic_studio.kernel.lora import remove_lora
                remove_lora(self.model)  # base params were frozen all along → bit-exact
            elif mode == "spot-only":
                region, backup = payload
                for n, m in region.items():
                    params[n].data[m.to(params[n].device)] = backup[n]
            else:
                for n, p in params.items():
                    p.data.copy_(payload[n].to(self.device))
        self._trained = None

    def training_delta(self, region=None):
        """How far the weights moved during the last training, split by the trained region vs the rest —
        shows what actually changed (and confirms a freeze held: a frozen side reports ~0). RMS of
        (current − pre-train) per side, plus the max abs move. region: {param: bool mask} to split by
        (the region that was trained/frozen). Returns None if nothing was trained (or lora — base frozen)."""
        trained = getattr(self, "_trained", None)
        if trained is None:
            return None
        mode, payload = trained
        params = dict(self.model.named_parameters())
        rs = rn = os_ = on = 0.0
        rmax = omax = 0.0
        if mode in ("full", "spot-freeze"):
            for name, base in payload.items():
                p = params.get(name)
                if p is None:
                    continue
                d = (p.detach().cpu() - base).flatten()
                if region and name in region:
                    m = region[name].reshape(-1)
                    din, dout = d[m], d[~m]
                    rs += float((din.double() ** 2).sum()); rn += din.numel(); rmax = max(rmax, float(din.abs().max()) if din.numel() else 0)
                    os_ += float((dout.double() ** 2).sum()); on += dout.numel(); omax = max(omax, float(dout.abs().max()) if dout.numel() else 0)
                else:
                    os_ += float((d.double() ** 2).sum()); on += d.numel(); omax = max(omax, float(d.abs().max()) if d.numel() else 0)
        elif mode == "spot-only":
            reg, backup = payload
            for name, m in reg.items():
                cur = params[name].detach().cpu()[m.cpu()]  # only the trained (selected) values
                d = (cur - backup[name].cpu()).flatten()
                rs += float((d.double() ** 2).sum()); rn += d.numel(); rmax = max(rmax, float(d.abs().max()) if d.numel() else 0)
        else:  # lora — base weights never moved
            return {"mode": mode, "region_rms": None, "other_rms": None}
        import math
        return {"mode": mode,
                "region_rms": math.sqrt(rs / rn) if rn else None, "region_max": rmax if rn else None, "region_params": int(rn),
                "other_rms": math.sqrt(os_ / on) if on else None, "other_max": omax if on else None, "other_params": int(on)}

    def ppl(self, examples):
        """Evaluate: mean-loss perplexity over examples (quantifies an intervention's damage)."""
        losses = []
        with torch.no_grad():
            for text in examples:
                ids = torch.tensor([self.tok.encode(text)], device=self.device)
                losses.append(float(self.model(ids, labels=ids).loss))
        return math.exp(sum(losses) / max(1, len(losses)))

    def _greedy_complete(self, model, device, prompt, max_tokens, temperature):
        """Minimal greedy/temperature decode of a raw code prompt on a given (model, device), cut at the
        first HumanEval stop. Self-contained (no probes/hooks read) so it's safe to run on a replica in a
        worker thread. Uses the shared tokenizer (encode/decode are thread-safe)."""
        from parametic_studio.kernel.humaneval import STOP_SEQUENCES, truncate_completion
        ids = torch.tensor([self.tok.encode(prompt)], device=device)
        eos = self.tok.eos_token_id
        out, text = [], ""
        # KV cache: decode is O(n), not O(n^2). This matters most for a DAMAGED model — with its coding
        # ability zeroed it rarely emits EOS or a clean function boundary, so it runs to the full max_tokens
        # every problem; without the cache that tail made eval-after-delete pathologically slow. The eager
        # attention the model is loaded with supports use_cache, and eval needs no attention-matrix probe.
        past, step = None, ids
        for _ in range(max_tokens):
            with torch.no_grad():
                o = model(step, past_key_values=past, use_cache=True)
            past = o.past_key_values
            logits = o.logits[0, -1]
            nxt = (int(torch.multinomial(torch.softmax(logits.float() / temperature, -1), 1))
                   if temperature and temperature > 0 else int(logits.argmax()))
            if nxt == eos:
                break
            out.append(nxt)
            # early stop the moment the completion runs past the target function (next def/class/comment/
            # main-guard/print) — cuts a healthy model to a short function.
            text = self._decode(out)
            if any(s in text for s in STOP_SEQUENCES):
                break
            step = torch.tensor([[nxt]], device=device)  # feed only the new token; cache holds the prefix
        return truncate_completion(text, STOP_SEQUENCES)

    def eval_pass_at_1(self, rows, max_tokens=512, temperature=0.0, extra_devices=None, progress=None, should_stop=None):
        """HumanEvalPack pass@1 over `rows` under the CURRENT weights (an applied knob/damage counts —
        replicas are deep-copied AFTER damage, so extra GPUs evaluate the same damaged model). Each row:
        generate a completion, run its unit tests in a subprocess, count pass. progress(done, total,
        passed) fires as problems finish. extra_devices: extra CUDA cards to split the problems across
        (transient replicas, freed after) — same 'help GPUs' idea as compute_spot. should_stop(): checked
        before each problem — return True to bail early (e.g. the client socket closed). Returns (passed, total)."""
        from parametic_studio.kernel.humaneval import build_program, check_correctness
        total = len(rows)
        devs = [str(self.device)] + [str(d) for d in (extra_devices or []) if str(d) != str(self.device)]

        if len(devs) <= 1:
            passed = 0
            for i, row in enumerate(rows):
                if should_stop and should_stop():
                    break
                comp = self._greedy_complete(self.model, self.device, row["prompt"], max_tokens, temperature)
                ok, _ = check_correctness(build_program(row, comp))
                passed += int(ok)
                if progress:
                    progress(i + 1, total, passed)
            return passed, total

        import copy
        import threading
        # replicas capture the current (damaged) weights; problems split round-robin across the cards
        replicas = {d: (self.model if d == str(self.device) else copy.deepcopy(self.model).to(d).eval()) for d in devs}
        chunks = {d: rows[i::len(devs)] for i, d in enumerate(devs)}
        lock = threading.Lock()
        state = {"done": 0, "passed": 0}
        results = {}

        def run(d):
            p = 0
            for row in chunks[d]:
                if should_stop and should_stop():
                    break
                comp = self._greedy_complete(replicas[d], torch.device(d), row["prompt"], max_tokens, temperature)
                ok, _ = check_correctness(build_program(row, comp))
                p += int(ok)
                with lock:  # update shared counters, then report OUTSIDE the lock so sends don't serialize compute
                    state["done"] += 1
                    state["passed"] += int(ok)
                    done, passed = state["done"], state["passed"]
                if progress:
                    progress(done, total, passed)
            results[d] = p

        threads = [threading.Thread(target=run, args=(d,)) for d in devs if chunks[d]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for d, r in replicas.items():
            if r is not self.model:
                del r
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
        return sum(results.values()), total

    def stop(self):
        self._stop = True

    def drilldown(self, layer):
        return self.last_raw[layer]  # [heads, kv] of the last step

    def _decode(self, ids):
        """Decode token ids to text. Falls back to the GPT-2 byte decoder when the tokenizer's own
        .decode() leaks the byte-level alphabet (Ġ/Ċ/… — seen on some code tokenizers like
        deepseek-coder), so streamed output is real text, not raw BPE markers."""
        text = self.tok.decode(ids)
        if "Ġ" in text or "Ċ" in text or "ĉ" in text:
            try:
                return bytearray(_BYTE_DECODER[c] for c in text).decode("utf-8", "replace")
            except KeyError:
                return text  # not pure byte-level (mixed alphabet) → leave as-is
        return text

    def format_prompt(self, prompt):
        tmpl = getattr(self.tok, "apply_chat_template", None)
        if tmpl and getattr(self.tok, "chat_template", None):
            return tmpl([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        return prompt

    def generate_text(self, prompt, max_tokens, probes=("attention",), temperature=0.0, spot_topk=0.01, spot_region_name=None):
        prompt = self.format_prompt(prompt)
        ids = torch.tensor([self.tok.encode(prompt)])
        yield from self.generate(ids, max_tokens, probes, temperature=temperature, spot_topk=spot_topk, spot_region_name=spot_region_name)

    def complete_code(self, prompt, max_tokens=512, temperature=0.0, stops=None):
        """Continue a raw code `prompt` (no chat template) and return the completion text,
        cut at the first stop sequence. Uses the plain decode loop so the current knob/damage
        state applies verbatim. ponytail: reuse generate(), collect text, truncate."""
        from parametic_studio.kernel.humaneval import truncate_completion, STOP_SEQUENCES
        stops = STOP_SEQUENCES if stops is None else stops
        ids = torch.tensor([self.tok.encode(prompt)])
        out = "".join(ev["text"] for ev in self.generate(ids, max_tokens, probes=(), temperature=temperature))
        return truncate_completion(out, stops)

    def generate(self, input_ids, max_tokens, probes=("attention",), temperature=0.0, spot_topk=0.01, spot_region_name=None):
        self._stop = False
        ids = input_ids.to(self.device)
        eos = self.tok.eos_token_id
        want_attn = "attention" in probes
        want_logit = "logitlens" in probes
        want_spot_act = "spot_activation" in probes
        # region is invariant across the whole run (weights/cache don't change mid-generation) —
        # resolve once, not per token: both paths below do model-wide work (topk / disk load).
        # spot_region_name (a saved region) → its full saved mask, exactly as saved (no re-threshold);
        # else the just-computed spot (_imp_cache), thresholded at spot_topk.
        spot_region = None
        if want_spot_act:
            spot_region = self.get_region(spot_region_name) if spot_region_name else self.locate_cached(spot_topk)
        gen_ids, prev_text = [], ""  # decode the running sequence, emit the delta — byte-level BPE
        # (Ġ/Ċ markers) only reconstructs correctly across whole tokens, not one id at a time.
        for step in range(max_tokens):
            if self._stop:
                return
            with torch.no_grad():
                # ponytail: no KV cache, full recompute each step — O(n²) but fine for demo lengths.
                out = self.model(ids, output_attentions=want_attn, output_hidden_states=want_logit)
            logits = out.logits[0, -1]
            if temperature and temperature > 0:  # temperature sampling; 0 = greedy
                nxt = int(torch.multinomial(torch.softmax(logits.float() / temperature, -1), 1))
            else:
                nxt = int(logits.argmax())
            if nxt == eos:
                return
            gen_ids.append(nxt)
            full = self._decode(gen_ids)
            piece, prev_text = full[len(prev_text):], full
            event = {"step": step, "token_id": nxt, "text": piece}
            if want_logit:
                # decode each layer's residual through the final norm + lm_head → top-1 prediction.
                norm, head = self.model.model.norm, self.model.lm_head
                with torch.no_grad():
                    rows = []
                    for h in out.hidden_states[1:]:  # [1:] skips the embedding layer
                        p = torch.softmax(head(norm(h[0, -1])).float(), -1)
                        tid = int(p.argmax())
                        rows.append({"token": self.tok.decode([tid]), "prob": float(p[tid])})
                event["logit"] = rows
            if want_attn:
                self.last_raw = [a[0, :, -1, :].float() for a in out.attentions]  # last query per-head (drilldown)
                if step == 0:
                    # prefill: emit every prompt query row (0..P-1) so the matrix is the full
                    # causal square N×N, not just the generated-token band. (no KV cache → full attn available)
                    P = ids.shape[1]
                    event["attn_rows"] = [torch.stack([a[0, :, q, : q + 1].mean(0).float() for a in out.attentions]) for q in range(P)]
                else:
                    event["attn_rows"] = [torch.stack([r.mean(0) for r in self.last_raw])]  # one new row [L, kv]
            if "activation" in probes:
                L = len(self.model.model.layers)
                event["act"] = torch.tensor(
                    [[self._acts[(l, m)] for m in self.modules_probed] for l in range(L)]
                )  # [L, M]
                event["act_detail"] = self.activation_detail_grid()
            if spot_region:
                event["spot_act"] = self.spot_activation_grid(spot_region)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=self.device)], dim=1)
            yield event
