import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import CodeMirror, { EditorView, keymap } from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import { json } from '@codemirror/lang-json'
import { javascript } from '@codemirror/lang-javascript'
import { java } from '@codemirror/lang-java'
import { cpp } from '@codemirror/lang-cpp'
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language'
import { tags } from '@lezer/highlight'

// Tauri bridge — no-ops in the browser (non-Tauri) so the studio runs either way.
const inTauri = () => typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window
async function tauriInvoke(cmd: string) {
  if (!inTauri()) return
  try { const { invoke } = await import('@tauri-apps/api/core'); await invoke(cmd) } catch { /* not in tauri */ }
}
// like tauriInvoke but takes args and surfaces success/failure — for commands the caller must await (ssh_connect/disconnect).
async function tauriInvokeResult(cmd: string, args?: Record<string, unknown>): Promise<void> {
  if (!inTauri()) throw new Error('not running in the desktop app')
  const { invoke } = await import('@tauri-apps/api/core')
  await invoke(cmd, args)
}
// like tauriInvokeResult but returns the command's value — for query commands (kernel_env_status/discover_pythons/probe_python).
async function tauriInvokeValue<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (!inTauri()) throw new Error('not running in the desktop app')
  const { invoke } = await import('@tauri-apps/api/core')
  return await invoke<T>(cmd, args)
}
// setup/onboarding contract (§0) — mirrors setup.rs PyInfo / EnvStatus.
type PyInfo = { path: string; version: string; source: string; pip: boolean; missing: string[]; ready: boolean }
type EnvStatus = { python_path: string | null; deps_ok: boolean; missing: string[] }
async function tauriListen(event: string, cb: (payload: string) => void): Promise<() => void> {
  if (!inTauri()) return () => {}
  try {
    const { listen } = await import('@tauri-apps/api/event')
    return await listen<string>(event, (e) => cb(e.payload))
  } catch { return () => {} }
}

const DEFAULT_WS = 'ws://127.0.0.1:8000/ws'  // IPv4 explicit — kernel binds 127.0.0.1 only; 'localhost' resolves to ::1 first on macOS and WKWebView won't fall back
const SSH_TUNNEL_WS = 'ws://localhost:8422/ws'  // P7: local end of the Rust-owned SSH tunnel to a remote kernel
const WS_URL = localStorage.getItem('ps_kernel_url') || DEFAULT_WS
const WS_TOKEN = localStorage.getItem('ps_kernel_token') || ''
const isRemoteConnected = () => WS_URL === SSH_TUNNEL_WS  // URL points at the tunnel (was in remote mode)
// The SSH tunnel is Rust-owned and dies with the process, so it survives a webview reload but NOT an
// app restart. sessionStorage has the same lifetime (kept on reload, cleared on a fresh app launch),
// so it tells the two apart: tunnel is actually live only if the flag is still here.
const isTunnelLive = () => isRemoteConnected() && sessionStorage.getItem('ps_tunnel_live') === '1'
const DEFAULT = { id: 'Qwen/Qwen2.5-1.5B-Instruct', label: 'Qwen2.5-1.5B' }
const SPOT_TOPK_OPTIONS = [0.005, 0.01, 0.05] as const
// fraction → percent label without spurious rounding: 0.005 → "0.5", 0.01 → "1", 0.05 → "5".
// fraction → percent label. 3 significant figures so tiny paper-scale values survive (0.000025 → "0.0025",
// not "0" as a fixed-2-decimal round would give), while whole values stay clean (0.05 → "5", 0.01 → "1").
const pctLabel = (k: number) => String(+((k * 100).toPrecision(3)))
const VIEWS = ['output', 'attention', 'importance', 'activations', 'activations-spot', 'usage', 'logitlens', 'spot', 'tensors', 'control', 'train', 'eval', 'log'] as const
type View = typeof VIEWS[number]
const VIEW_LABEL: Record<string, string> = {
  output: 'Output',
  attention: 'Attention',
  importance: 'Parameter Importance',
  activations: 'Activations',
  'activations-spot': 'Spot activation',
  usage: 'Region usage',
  logitlens: 'Logit lens',
  spot: 'Spot',
  tensors: 'Tensors',
  control: 'Region control',
  train: 'Train',
  eval: 'Eval',
  log: 'Log',
}
const viewLabel = (v: string) => VIEW_LABEL[v] ?? v
const DEFAULT_DATASET = 'def add(a, b):\n    return a + b\nfor i in range(10):\n    print(i)\nx = [3, 1, 2]\nx.sort()'
const GENERAL_SET = 'The weather was pleasant and the streets were quiet.\nShe walked to the market to buy fresh vegetables.\nHistory teaches us patience and perspective.\nThe orchestra played beautifully through the evening.'
const PRESETS: Record<string, string> = {
  python: 'def add(a, b):\n    return a + b\nfor i in range(10):\n    print(i)\nx = [3, 1, 2]\nx.sort()\nwith open("f") as fp:\n    data = fp.read()\nresult = [n * n for n in nums if n > 0]\ntry:\n    v = int(s)\nexcept ValueError:\n    v = 0',
  java: 'public int add(int a, int b) {\n    return a + b;\n}\nfor (int i = 0; i < 10; i++) {\n    System.out.println(i);\n}\nList<Integer> x = new ArrayList<>();\nx.sort(Comparator.naturalOrder());\ntry {\n    int v = Integer.parseInt(s);\n} catch (NumberFormatException e) {\n    v = 0;\n}',
  cpp: 'int add(int a, int b) {\n    return a + b;\n}\nfor (int i = 0; i < 10; ++i) {\n    std::cout << i << std::endl;\n}\nstd::vector<int> x = {3, 1, 2};\nstd::sort(x.begin(), x.end());\nstd::ifstream fp("f.txt");\nstd::string data((std::istreambuf_iterator<char>(fp)), {});\ntry {\n    int v = std::stoi(s);\n} catch (const std::invalid_argument& e) {\n    v = 0;\n}',
  javascript: 'function add(a, b) {\n    return a + b;\n}\nfor (let i = 0; i < 10; i++) {\n    console.log(i);\n}\nconst x = [3, 1, 2];\nx.sort((a, b) => a - b);\nconst data = await fs.readFile("f.txt", "utf8");\nconst result = nums.filter(n => n > 0).map(n => n * n);\ntry {\n    v = parseInt(s, 10);\n} catch (e) {\n    v = 0;\n}',
}

// CodeMirror: theme wired to the app's CSS variables. Syntax palette lives in index.css as
// --syn-* tokens (VS Code Dark+/Light+ colors), so dark/light follow the app theme automatically.
const cmTheme = EditorView.theme({
  '&': { backgroundColor: 'var(--bg-0)', color: 'var(--text-0)', height: '100%', fontSize: '13px' },
  '.cm-content': { fontFamily: 'var(--mono)', caretColor: 'var(--text-0)' },
  '.cm-gutters': { backgroundColor: 'var(--bg-0)', color: 'var(--text-2)', border: 'none', borderRight: '1px solid var(--line)' },
  '.cm-activeLine': { backgroundColor: 'color-mix(in srgb, var(--text-0) 4%, transparent)' },
  '.cm-activeLineGutter': { backgroundColor: 'transparent', color: 'var(--text-0)' },
  '&.cm-focused .cm-selectionBackground, .cm-selectionBackground': { backgroundColor: 'color-mix(in srgb, var(--accent) 25%, transparent)' },
  '.cm-cursor': { borderLeftColor: 'var(--text-0)' },
  '&.cm-focused': { outline: 'none' },
})
const cmHighlight = syntaxHighlighting(HighlightStyle.define([
  { tag: [tags.keyword, tags.controlKeyword, tags.moduleKeyword, tags.operatorKeyword], color: 'var(--syn-kw)' },
  { tag: [tags.definitionKeyword, tags.bool, tags.null, tags.atom, tags.self], color: 'var(--syn-def)' },
  { tag: [tags.string, tags.special(tags.string), tags.regexp], color: 'var(--syn-str)' },
  { tag: tags.number, color: 'var(--syn-num)' },
  { tag: [tags.comment, tags.lineComment, tags.blockComment], color: 'var(--syn-com)', fontStyle: 'italic' },
  { tag: [tags.function(tags.variableName), tags.function(tags.propertyName)], color: 'var(--syn-fn)' },
  { tag: [tags.typeName, tags.className, tags.namespace, tags.standard(tags.variableName)], color: 'var(--syn-type)' },
  { tag: [tags.variableName, tags.propertyName, tags.attributeName, tags.definition(tags.variableName)], color: 'var(--syn-var)' },
]))
function cmLangExt(name: string) {
  const n = name.toLowerCase()
  if (n.endsWith('.py') || n === 'python') return [python()]
  if (n.endsWith('.js') || n.endsWith('.ts') || n === 'javascript') return [javascript()]
  if (n.endsWith('.java') || n === 'java') return [java()]
  if (/\.(cpp|cc|cxx|h|hpp)$/.test(n) || n === 'cpp') return [cpp()]
  if (n.endsWith('.json') || n.endsWith('.jsonl')) return [json()]
  return []
}

type Logit = { token: string; prob: number }
type Spot = { layers: number; modules: string[]; grid: number[][] }
type Cond = { code_ppl: number; general_ppl: number | null }
type KnobRow = { key: string; kind: 'cell' | 'spot' | 'named'; layer?: number; module?: string; name?: string; topk?: number; op: string; alpha: number }
type ModelData = {
  output: string; frames: number[][][]; act: number[][] | null; logit: Logit[] | null;
  actDetail: Spot | null; promptImportance: Spot | null; promptImportanceProg: { i: number; total: number } | null;
  spot: Spot | null; spotAct: Spot | null; spotProg: { i: number; total: number } | null; perhead: { layer: number; data: number[][] } | null
  actFrames: number[][][]; actDetailFrames: number[][][]; spotActFrames: number[][][]; tokenTexts: string[]  // per-token grids + labels → cumulative + timeline views
  promptTokens: string[]  // decoded prompt tokens → attention axes labelled with real text (prompt ⧺ generated)
  concentration: number[][] | null  // Lorenz/CDF: [[frac_params, frac_importance], ...]
  contrast: { topk: number; clean?: Cond; spot?: Cond; random?: Cond; bottom?: Cond } | null; contrastProg: string | null
  // k-sweep control experiment (paper Table 1): zero spot/random/bottom at each top-k%, measure code+general PPL
  sweep: { topks: number[]; clean?: Cond; rows: { topk: number; cond: string; code_ppl: number; general_ppl: number | null }[] } | null; sweepProg: string | null
  usage: { rows: { prompt: string; overall: number }[]; running: boolean; detail: { prompt: string; overall: number; per_layer: number[]; tokens: { text: string; usage: number }[] } | null }
  knobs: KnobRow[]; kppl: { base: number | null; inter: number | null }; ab: { base: string | null; inter: string | null }
  regions: { name: string; count: number; base_topk?: number }[]; evals: { code: number | null; general: number | null }
  evalProg: { i: number; total: number; passed: number } | null
  evalResult: { dataset: string; passed: number; total: number; pass_at_1: number; damaged: boolean; knobCount: number } | null
  evalPrev: { dataset: string; passed: number; total: number; pass_at_1: number; damaged: boolean; knobCount: number } | null
  train: { losses: number[]; total: number; running: boolean; trained: boolean; before: { code: number | null; general: number | null } | null; error: string | null }
  tensors: { name: string; shape: number[]; dtype: string }[] | null
  count: number; busy: boolean; loading: boolean
  lastRunId: string | null; framesFlushed: boolean
  download: { pct: number; done_mb: number; total_mb: number } | null  // HF download stream (open only)
}
const emptyTrain = (): ModelData['train'] => ({ losses: [], total: 0, running: false, trained: false, before: null, error: null })
const empty = (): ModelData => ({ output: '', frames: [], act: null, actDetail: null, promptImportance: null, promptImportanceProg: null, logit: null, spot: null, spotAct: null, spotProg: null, perhead: null, actFrames: [], actDetailFrames: [], spotActFrames: [], tokenTexts: [], promptTokens: [], concentration: null, contrast: null, contrastProg: null, sweep: null, sweepProg: null, usage: { rows: [], running: false, detail: null }, knobs: [], kppl: { base: null, inter: null }, ab: { base: null, inter: null }, regions: [], evals: { code: null, general: null }, evalProg: null, evalResult: null, evalPrev: null, train: emptyTrain(), tensors: null, count: 0, busy: false, loading: false, lastRunId: null, framesFlushed: false, download: null })

type Tile = { id: number; model: string; tabs: string[]; active: number; h: number }  // tabs: View | `data:<name>`
type Col = { id: number; w: number; tiles: Tile[] }

// theme-aware colormaps: dark base → bright ramp in dark mode, light base → deep ramp in light mode
const isLight = () => document.documentElement.dataset.theme === 'light'
function ramp(t: number, from: [number, number, number], to: [number, number, number]) {
  return `rgb(${Math.round(from[0] + (to[0] - from[0]) * t)},${Math.round(from[1] + (to[1] - from[1]) * t)},${Math.round(from[2] + (to[2] - from[2]) * t)})`
}
function cellColor(v: number, max: number) {  // blue ramp (attention/activations)
  const t = max > 0 ? v / max : 0
  return isLight() ? ramp(t, [0xE9, 0xE7, 0xE4], [0x00, 0x4F, 0xB0]) : ramp(t, [0x20, 0x1d, 0x1d], [0x00, 0x7a, 0xff])
}
function ampColor(v: number, max: number) {   // amber ramp (spot importance)
  const t = max > 0 ? v / max : 0
  return isLight() ? ramp(t, [0xE9, 0xE7, 0xE4], [0xB0, 0x6A, 0x00]) : ramp(t, [0x20, 0x1d, 0x1d], [0xe0, 0xa8, 0x5e])
}
// per-region hues for spot comparison — each saved region renders in its own color
const REGION_HUES: [number, number, number][] = [[0x00, 0x7a, 0xff], [0xe0, 0xa8, 0x5e], [0x30, 0xd1, 0x58], [0xff, 0x64, 0x82]]
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = l - c / 2
  const [r, g, b] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x] : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x]
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)]
}
// distinct hue per compare region, unbounded: curated palette first, then golden-angle HSL so any
// number of regions each get their own colour (no cap on how many spots you compare).
const hueAt = (i: number): [number, number, number] => i < REGION_HUES.length ? REGION_HUES[i] : hslToRgb((i * 137.508) % 360, 0.62, 0.62)
const hueRamp = (rgb: [number, number, number]) => (v: number, max: number) => {
  const t = max > 0 ? v / max : 0
  return ramp(t, isLight() ? [0xE9, 0xE7, 0xE4] : [0x20, 0x1d, 0x1d], rgb)
}
const hueCss = (rgb: [number, number, number]) => `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`
function ScaleBar({ max, color, label }: { max: number; color: (v: number, max: number) => string; label?: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, color: 'var(--text-2)', margin: '4px 0' }}>
      {label && <span>{label}</span>}
      <span>0</span>
      <div style={{ width: 90, height: 8, borderRadius: 6, background: `linear-gradient(to right, ${color(0, 1)}, ${color(0.5, 1)}, ${color(1, 1)})` }} />
      <span>{Number.isFinite(max) ? max.toPrecision(3) : '—'}</span>
    </div>
  )
}
function normalizedScoreGrid(grid: number[][]): { grid: number[][]; max: number } {
  const max = grid.flat().reduce((a, v) => Math.max(a, Number(v) || 0), 0)
  return { max, grid: grid.map((row) => row.map((v) => (max > 0 ? ((Number(v) || 0) / max) * 100 : 0))) }
}
// Same, but scaled against an external reference max (the region's importance mass at its base top-k%)
// instead of the grid's own max. Re-thresholding a region to a smaller % captures less importance mass,
// so on a fixed reference the whole map visibly dims — otherwise per-threshold re-normalization pins the
// max at 100 every time and the heatmap looks identical at 1% vs 5%. `max` is still this grid's own raw
// max (for the "raw max |g×w|" readout). refMax ≤ 0 → fall back to self-normalization.
function scoreGridAgainst(grid: number[][], refMax: number): { grid: number[][]; max: number } {
  const max = grid.flat().reduce((a, v) => Math.max(a, Number(v) || 0), 0)
  const denom = refMax > 0 ? refMax : max
  return { max, grid: grid.map((row) => row.map((v) => (denom > 0 ? ((Number(v) || 0) / denom) * 100 : 0))) }
}
function moduleAlignedDiffGrid(a: number[][], aModules: string[], b: number[][], bModules: string[]): number[][] {
  const bIdx = new Map(bModules.map((m, i) => [m, i]))
  return a.map((row, l) => row.map((v, c) => v - (b[l]?.[bIdx.get(aModules[c]) ?? -1] ?? 0)))
}
function Grid({ rows, cols, rowH, onRow, onRowEnter, onLeave, cellTitle }: { rows: number[][]; cols: number; rowH: number; onRow?: (i: number) => void; onRowEnter?: (i: number) => void; onLeave?: () => void; cellTitle?: (i: number, k: number, v: number) => string }) {
  const max = Math.max(...rows.flat())
  return (
    <div onMouseLeave={onLeave} style={{ display: 'grid', gridTemplateRows: `repeat(${rows.length}, ${rowH}px)`, gap: 1 }}>
      {rows.map((row, i) => (
        <div key={i} onClick={onRow ? () => onRow(i) : undefined} onMouseEnter={onRowEnter ? () => onRowEnter(i) : undefined} className={onRow ? 'hover-row' : undefined} title={onRow ? `layer ${i} — hover to preview · click to pin` : undefined}
          style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: 1, cursor: onRow ? 'pointer' : 'default' }}>
          {Array.from({ length: cols }, (_, k) => <div key={k} title={cellTitle && k < row.length ? cellTitle(i, k, row[k]) : undefined} style={{ background: k < row.length ? cellColor(row[k], max) : 'var(--bg-2)' }} />)}
        </div>
      ))}
    </div>
  )
}
// short axis label for a module param name so captures are legible without hovering:
// self_attn.q_proj.weight → q · mlp.gate_proj.weight → gate · input_layernorm.weight → ln1 · .bias → +b
function moduleAbbrev(name: string): string {
  const bias = name.endsWith('.bias')
  const s = name.replace(/\.(weight|bias)$/, '')
  const map: Record<string, string> = {
    'self_attn.q_proj': 'q', 'self_attn.k_proj': 'k', 'self_attn.v_proj': 'v', 'self_attn.o_proj': 'o',
    'mlp.gate_proj': 'gate', 'mlp.up_proj': 'up', 'mlp.down_proj': 'down',
    'input_layernorm': 'ln1', 'post_attention_layernorm': 'ln2',
  }
  return (map[s] ?? s.split('.').filter((p) => p !== 'proj').pop() ?? s) + (bias ? '+b' : '')
}
function moduleFullLabel(name: string): string {
  const bias = name.endsWith('.bias')
  const s = name.replace(/\.(weight|bias)$/, '')
  const map: Record<string, string> = {
    'self_attn.q_proj': 'query',
    'self_attn.k_proj': 'key',
    'self_attn.v_proj': 'value',
    'self_attn.o_proj': 'output',
    'mlp.gate_proj': 'gate',
    'mlp.up_proj': 'up',
    'mlp.down_proj': 'down',
    'input_layernorm': 'input norm',
    'post_attention_layernorm': 'post-attn norm',
  }
  return (map[s] ?? (s.split('.').filter((p) => p !== 'proj').join(' ') || s)) + (bias ? ' bias' : '')
}
function SpotGrid({ grid, modules, onCell, selected, color = ampColor, cellTitle, onHover, hovered, labels = true }: { grid: number[][]; modules: string[]; onCell?: (l: number, module: string) => void; selected?: Set<string>; color?: (v: number, max: number) => string; cellTitle?: (l: number, module: string, v: number) => string; onHover?: (cell: string | null) => void; hovered?: string | null; labels?: boolean }) {
  // coerce to finite numbers — a stray NaN/Infinity/null in the grid must not throw during render (white screen).
  const flat = grid.flat().map((v) => (typeof v === 'number' && isFinite(v) ? v : 0))
  const max = flat.reduce((a, b) => (b > a ? b : a), 0)   // reduce, not spread — spread overflows the stack on huge arrays
  const sorted = [...flat].sort((a, b) => b - a)
  const thr = sorted[Math.max(0, Math.floor(sorted.length * 0.05) - 1)] ?? Infinity
  const L = grid.length
  const step = Math.max(1, Math.ceil(L / 12))   // ~12 layer ticks max — every row labelled would be unreadable at 9px
  const cols = `${labels ? '15px ' : ''}repeat(${modules.length}, 1fr)`
  const lab = { fontSize: 8, color: 'var(--text-2)', lineHeight: '9px', overflow: 'hidden', whiteSpace: 'nowrap' } as const
  return (
    <div>
      {labels && (
        <div style={{ display: 'grid', gridTemplateColumns: cols, gap: 1, marginBottom: 2 }}>
          <div />
          {modules.map((m, c) => <div key={c} title={m} style={{ ...lab, textAlign: 'center' }}>{moduleAbbrev(String(m ?? ''))}</div>)}
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateRows: `repeat(${L}, 9px)`, gap: 1 }}
        onMouseLeave={onHover ? () => onHover(null) : undefined}>
        {grid.map((row, l) => (
          <div key={l} style={{ display: 'grid', gridTemplateColumns: cols, gap: 1 }}>
            {labels && <div style={{ ...lab, textAlign: 'right', paddingRight: 3 }}>{(l % step === 0 || l === L - 1) ? l : ''}</div>}
            {row.map((rv, c) => {
              const v = typeof rv === 'number' && isFinite(rv) ? rv : 0   // guard: bad cell value must not crash render
              const mod = modules[c] ?? '?'
              const key = `${l}.${mod}`
              const sel = selected?.has(key)
              const hov = hovered === key
              return <div key={c} onClick={onCell ? () => onCell(l, mod) : undefined}
                onMouseEnter={onHover ? () => onHover(key) : undefined}
                title={cellTitle ? cellTitle(l, mod, v) : `L${l} · ${mod} · ${v.toExponential(2)}${onCell ? ' — click → knob' : ''}`}
                style={{ background: color(v, max), cursor: onCell ? 'pointer' : 'default', outline: hov ? '1.5px solid var(--accent)' : sel ? '1.5px solid var(--accent)' : v >= thr ? `1px solid ${isLight() ? '#1A1717' : '#FDFCFC'}` : 'none', outlineOffset: (hov || sel) ? -1 : 0 }} />
            })}
          </div>
        ))}
      </div>
    </div>
  )
}

// horizontal labelled bars, each showing value as a % of the total — reads left→right, biggest first.
function BarList({ items, color, valueLabel }: { items: { label: string; value: number; title?: string }[]; color: (v: number, max: number) => string; valueLabel?: (v: number, total: number) => string }) {
  const max = items.reduce((a, b) => Math.max(a, b.value), 0)
  const total = items.reduce((a, b) => a + b.value, 0) || 1
  return (
    <div style={{ display: 'grid', gap: 6 }}>
      {items.map((it) => (
        <div key={it.label} title={it.title} style={{ display: 'grid', gridTemplateColumns: '52px 1fr 38px', gap: 8, alignItems: 'center', fontSize: 11 }}>
          <span className="mono" style={{ color: 'var(--text-1)', overflow: 'hidden', textOverflow: 'ellipsis' }}>{it.label}</span>
          <div style={{ background: 'var(--bg-2)', borderRadius: 5, height: 11, overflow: 'hidden' }}>
            <div style={{ height: 11, width: `${max ? (it.value / max) * 100 : 0}%`, background: color(it.value, max), borderRadius: 5, transition: 'width var(--ease)' }} />
          </div>
          <span style={{ color: 'var(--text-2)', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{valueLabel ? valueLabel(it.value, total) : `${((it.value / total) * 100).toFixed(0)}%`}</span>
        </div>
      ))}
    </div>
  )
}
// Concentration Lorenz/CDF: filled area under the cumulative curve, with the equality diagonal as a
// reference. A curve that bows hard toward the top-left = importance concentrated in a few weights.
function LineChart({ points, width = 300, height = 150 }: { points: number[][]; width?: number; height?: number }) {
  const pad = 6, W = width, H = height
  const px = (x: number) => pad + x * (W - 2 * pad)
  const py = (y: number) => H - pad - y * (H - 2 * pad)
  const line = points.map((p, i) => `${i ? 'L' : 'M'}${px(p[0]).toFixed(1)} ${py(p[1]).toFixed(1)}`).join(' ')
  const area = `${line} L${px(points.at(-1)![0]).toFixed(1)} ${py(0).toFixed(1)} L${px(0).toFixed(1)} ${py(0).toFixed(1)} Z`
  return (
    <svg width={W} height={H} style={{ display: 'block' }}>
      <line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)} stroke="var(--line-strong)" strokeWidth={1} strokeDasharray="3 3" />
      <path d={area} fill="var(--accent-soft)" />
      <path d={line} fill="none" stroke="var(--accent)" strokeWidth={1.8} strokeLinejoin="round" />
    </svg>
  )
}
// Causal contrast: for each condition (clean / spot / random / bottom), a log-scaled bar per metric
// (code PPL, general PPL). The story: spot damage explodes code PPL while controls + general stay flat.
function ContrastBars({ contrast }: { contrast: NonNullable<ModelData['contrast']> }) {
  const conds = ['clean', 'spot', 'random', 'bottom'] as const
  const color: Record<string, string> = { clean: 'var(--text-2)', spot: 'var(--danger)', random: 'var(--accent)', bottom: 'var(--live)' }
  const metrics: { key: 'code_ppl' | 'general_ppl'; label: string }[] = [{ key: 'code_ppl', label: 'code PPL' }, { key: 'general_ppl', label: 'general PPL' }]
  const allVals = conds.flatMap((c) => metrics.map((m) => contrast[c]?.[m.key])).filter((v): v is number => typeof v === 'number' && v > 0)
  const lmax = Math.log10(Math.max(...allVals, 10))
  const clean = contrast.clean
  return (
    <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }}>
      {metrics.map((m) => (
        <div key={m.key} style={{ flex: '1 1 200px', minWidth: 190 }}>
          <div style={{ ...hint, fontSize: 11, marginBottom: 8 }}>{m.label}</div>
          <div style={{ display: 'grid', gap: 7 }}>
            {conds.map((c) => {
              const v = contrast[c]?.[m.key]
              const ratio = clean && typeof v === 'number' && clean[m.key] ? v / (clean[m.key] as number) : null
              const w = typeof v === 'number' && v > 0 ? (Math.log10(v) / lmax) * 100 : 0
              return (
                <div key={c} style={{ display: 'grid', gridTemplateColumns: '52px 1fr 96px', gap: 8, alignItems: 'center', fontSize: 11 }}>
                  <span style={{ color: color[c], fontWeight: c === 'spot' ? 600 : 500 }}>{c}</span>
                  <div style={{ background: 'var(--bg-2)', borderRadius: 5, height: 12, overflow: 'hidden' }}>
                    <div style={{ height: 12, width: `${Math.max(2, w)}%`, background: color[c], borderRadius: 5, transition: 'width var(--ease)' }} />
                  </div>
                  <span className="mono" style={{ color: 'var(--text-1)', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                    {typeof v === 'number' ? v.toPrecision(4) : '…'}{ratio && ratio >= 1.5 ? <span style={{ color: 'var(--danger)' }}> ×{ratio < 100 ? ratio.toFixed(1) : Math.round(ratio)}</span> : ''}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}
// Symmetric N×N overlap heatmap of pairwise Jaccard (|A∩B|/|A∪B|) between saved spots — reveals a
// shared coding core (high off-diagonal) vs language-specific spots (low). Diagonal is self (—).
function JaccardMatrix({ names, jaccard }: { names: string[]; jaccard: Record<string, number> }) {
  const val = (a: string, b: string) => (a === b ? 1 : (jaccard[`${a}|${b}`] ?? jaccard[`${b}|${a}`] ?? 0))
  const short = (n: string) => n.replace(/-spot$/, '').replace(/^qwen[\d.]*-[\d.]*-/i, '')
  const cell = 40
  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ display: 'grid', gridTemplateColumns: `72px repeat(${names.length}, ${cell}px)`, gap: 3, fontSize: 10 }}>
        <div />
        {names.map((n) => <div key={n} title={n} style={{ textAlign: 'center', color: 'var(--text-2)', overflow: 'hidden', whiteSpace: 'nowrap' }}>{short(n)}</div>)}
        {names.map((r) => (
          <Fragment key={r}>
            <div title={r} style={{ color: 'var(--text-2)', textAlign: 'right', paddingRight: 5, whiteSpace: 'nowrap', overflow: 'hidden', alignSelf: 'center' }}>{short(r)}</div>
            {names.map((c) => {
              const v = val(r, c)
              return <div key={c} title={`${r} ∩ ${c} = ${(v * 100).toFixed(1)}%`}
                style={{ height: cell, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 6, background: r === c ? 'var(--bg-2)' : ampColor(v, 1), color: v > 0.5 && r !== c ? (isLight() ? '#fff' : '#1a1717') : 'var(--text-1)', fontVariantNumeric: 'tabular-nums' }}>{r === c ? '—' : (v * 100).toFixed(0)}</div>
            })}
          </Fragment>
        ))}
      </div>
    </div>
  )
}
// Given a layer×module importance grid, derive the two summary breakdowns that tell the spot's story:
// which module types hold it, and at which network depth it concentrates.
function SpotSummary({ grid, modules }: { grid: number[][]; modules: string[] }) {
  const byModule = modules
    .map((m, c) => ({ label: moduleAbbrev(String(m ?? '')), value: grid.reduce((a, row) => a + (Number(row[c]) || 0), 0), title: String(m) }))
    .filter((x) => x.value > 0)
    .sort((a, b) => b.value - a.value)
  const L = grid.length
  const layerSum = grid.map((row) => row.reduce((a, b) => a + (Number(b) || 0), 0))
  const bands = Math.min(6, L)
  const bandSize = Math.ceil(L / bands)
  const byLayer = Array.from({ length: bands }, (_, b) => {
    const lo = b * bandSize, hi = Math.min(L, lo + bandSize)
    return { label: hi - lo === 1 ? `L${lo}` : `L${lo}–${hi - 1}`, value: layerSum.slice(lo, hi).reduce((a, c) => a + c, 0) }
  }).filter((x) => x.value > 0)
  return (
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 12 }}>
      <div style={{ flex: '1 1 240px', minWidth: 220, maxWidth: 300 }}>
        <VizCard title="By module" accent="var(--accent)"><BarList items={byModule} color={ampColor} /></VizCard>
      </div>
      <div style={{ flex: '1 1 240px', minWidth: 220, maxWidth: 300 }}>
        <VizCard title="By layer depth" accent="var(--accent)"><BarList items={byLayer} color={ampColor} /></VizCard>
      </div>
    </div>
  )
}
// mean of per-token [L,M] grids → one cumulative [L,M] grid (the cumulative activation map).
function meanFrames(frames: number[][][]): number[][] {
  if (!frames.length) return []
  const L = frames[0].length, M = frames[0][0]?.length ?? 0
  const out = Array.from({ length: L }, () => Array(M).fill(0))
  for (const f of frames) for (let l = 0; l < L; l++) for (let c = 0; c < M; c++) out[l][c] += (f[l]?.[c] ?? 0)
  for (let l = 0; l < L; l++) for (let c = 0; c < M; c++) out[l][c] /= frames.length
  return out
}
// per-token [L,M] grids → layer×token matrix (cell = that layer's total activation at that token).
function layerTimeMatrix(frames: number[][][]): number[][] {
  if (!frames.length) return []
  const L = frames[0].length
  return Array.from({ length: L }, (_, l) => frames.map((f) => (f[l] ?? []).reduce((a, b) => a + (b || 0), 0)))
}
// layer×token heatmap with a fixed column width (so long runs scroll horizontally) + rotated token
// labels under each column so you can read WHICH token each column is, not just on hover.
function ActivationTimeline({ matrix, tokens, label }: { matrix: number[][]; tokens: string[]; label: string }) {
  const L = matrix.length, T = matrix[0]?.length ?? 0
  const colW = 15, rowH = 7
  const max = matrix.reduce((a, r) => Math.max(a, ...r), 1e-9)
  const fmt = (t: string) => (t ?? '').replace(/\n/g, '⏎').replace(/\t/g, '⇥').replace(/ /g, '·') || '∅'
  return (
    <div>
      <ScaleBar max={max} color={cellColor} label={label} />
      <div style={{ overflowX: 'auto', paddingBottom: 6 }}>
        <div style={{ width: T * (colW + 1), minWidth: '100%' }}>
          <div style={{ display: 'grid', gridTemplateRows: `repeat(${L}, ${rowH}px)`, gap: 1 }}>
            {matrix.map((row, l) => (
              <div key={l} style={{ display: 'grid', gridTemplateColumns: `repeat(${T}, ${colW}px)`, gap: 1 }}>
                {row.map((v, t) => <div key={t} title={`L${l} · tok ${t} "${fmt(tokens[t])}" · ${v.toFixed(2)}`} style={{ background: cellColor(v, max), height: rowH }} />)}
              </div>
            ))}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${T}, ${colW}px)`, gap: 1, marginTop: 4, height: 56 }}>
            {Array.from({ length: T }, (_, t) => (
              <div key={t} title={fmt(tokens[t])} style={{ position: 'relative', width: colW }}>
                <span style={{ position: 'absolute', top: 1, left: Math.round(colW / 2) + 3, transform: 'rotate(90deg)', transformOrigin: 'left top', fontSize: 9, lineHeight: `${colW}px`, color: 'var(--text-2)', whiteSpace: 'nowrap', fontFamily: 'var(--mono)' }}>{fmt(tokens[t])}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
// [Cumulative | Timeline] pill toggle for the activation views.
function ModeToggle({ mode, set }: { mode: 'cumulative' | 'timeline'; set: (m: 'cumulative' | 'timeline') => void }) {
  return (
    <span style={{ display: 'inline-flex', gap: 4 }}>
      {(['cumulative', 'timeline'] as const).map((m) => (
        <span key={m} onClick={() => set(m)} className={`chip${mode === m ? ' on' : ''}`} style={{ textTransform: 'capitalize' }}>{m}</span>
      ))}
    </span>
  )
}
// Linear-style content card: accent dot + title + optional right slot, then padded body.
function VizCard({ title, subtitle, accent, right, children }: { title: string; subtitle?: string; accent?: string; right?: ReactNode; children: ReactNode }) {
  return (
    <div className="card" style={{ maxWidth: 560 }}>
      <div className="card-head">
        <span style={{ width: 8, height: 8, borderRadius: 3, background: accent ?? 'var(--accent)', flexShrink: 0 }} />
        <span className="card-title">{title}</span>
        {right && <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>{right}</span>}
      </div>
      <div style={{ padding: '14px 16px' }}>
        {subtitle && <div style={{ color: 'var(--text-2)', fontSize: 12, marginBottom: 14, lineHeight: 1.55 }}>{subtitle}</div>}
        {children}
      </div>
    </div>
  )
}
const hint = { color: 'var(--text-2)' as const }
const iconBtn = { background: 'transparent', border: 'none', color: 'var(--text-2)', cursor: 'pointer', padding: '0 5px', fontSize: 11 }
// common bordered action button — mechanical replacement target for the ~30 inline `[...]` buttons
function Btn({ onClick, children, color = 'var(--text-1)', title, disabled, style }: { onClick?: () => void; children: React.ReactNode; color?: string; title?: string; disabled?: boolean; style?: React.CSSProperties }) {
  return (
    <button className="btn" onClick={onClick} title={title} disabled={disabled}
      style={{ padding: '3px 12px', cursor: disabled ? 'default' : 'pointer', fontSize: 11, color, ...style }}>
      {children}
    </button>
  )
}
// inline 14px stroke icons — name→path map, no icon dependency. color inherits from .tree-icon.
const ICON_PATHS: Record<string, React.ReactNode> = {
  cube: <><path d="M12 2 21 7v10l-9 5-9-5V7z" /><path d="M12 2v10m0 0 9-5m-9 5-9-5" /></>,
  folder: <path d="M3 6a1 1 0 0 1 1-1h5l2 2h8a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z" />,
  file: <><path d="M6 3h8l4 4v14H6z" /><path d="M14 3v4h4M9 12h6M9 16h6" /></>,
  database: <><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5" /><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></>,
  link: <><path d="M9 15l6-6" /><path d="M10.5 6.5 13 4a4 4 0 0 1 6 6l-2.5 2.5M13.5 17.5 11 20a4 4 0 0 1-6-6l2.5-2.5" /></>,
  diamond: <path d="M12 2 22 12 12 22 2 12z" />,
  tensor: <rect x="6" y="6" width="12" height="12" rx="1.5" />,
}
function Icon({ name, size = 14 }: { name: string; size?: number }) {
  return (
    <svg className="tree-icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round">
      {ICON_PATHS[name] ?? ICON_PATHS.file}
    </svg>
  )
}
function Chevron({ open }: { open: boolean }) {
  return (
    <svg className={`tree-chevron${open ? ' open' : ''}`} width={12} height={12} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 6l6 6-6 6" />
    </svg>
  )
}
type MenuItem = { label: string; onClick: () => void; key?: string; danger?: boolean; disabled?: boolean } | 'sep'
function ContextMenu({ x, y, items, onClose }: { x: number; y: number; items: MenuItem[]; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState({ x, y })
  useEffect(() => {  // clamp inside the viewport once measured
    const el = ref.current; if (!el) return
    const r = el.getBoundingClientRect()
    setPos({ x: Math.min(x, window.innerWidth - r.width - 6), y: Math.min(y, window.innerHeight - r.height - 6) })
  }, [x, y])
  useEffect(() => {
    const close = () => onClose()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    // defer so the opening right-click / click doesn't immediately close it
    const t = window.setTimeout(() => {
      window.addEventListener('click', close)
      window.addEventListener('contextmenu', close)
      window.addEventListener('scroll', close, true)
    }, 0)
    window.addEventListener('keydown', onKey)
    return () => { clearTimeout(t); window.removeEventListener('click', close); window.removeEventListener('contextmenu', close); window.removeEventListener('scroll', close, true); window.removeEventListener('keydown', onKey) }
  }, [onClose])
  return (
    <div ref={ref} className="ctx-menu" style={{ left: pos.x, top: pos.y }} onClick={(e) => e.stopPropagation()} onContextMenu={(e) => { e.preventDefault(); e.stopPropagation() }}>
      {items.map((it, i) => it === 'sep'
        ? <div key={i} className="ctx-sep" />
        : <div key={i} className={`ctx-item${it.danger ? ' danger' : ''}${it.disabled ? ' disabled' : ''}`} onClick={() => { if (!it.disabled) it.onClick() }}>
            <span>{it.label}</span>{it.key && <span className="ctx-key">{it.key}</span>}
          </div>)}
    </div>
  )
}

// PENDING_OPS: request ops that get a ⟳-pending badge until their matching response (or error) arrives
const PENDING_OPS = new Set(['ppl', 'intervene', 'save_region', 'import_region', 'drilldown', 'region_compare', 'region_info', 'eval_code', 'causal_contrast', 'region_usage', 'region_usage_batch', 'tensor_values', 'causal_sweep', 'training_delta'])

// Dataset content → training/spot examples. Understands JSON arrays / JSONL (records), with an
// optional per-dataset field selection (e.g. context+question for benchmark files); plain text
// falls back to one-example-per-line.
function parseRecords(text: string): unknown[] | null {
  const t = text.trim()
  if (t.startsWith('[')) {
    try { const j = JSON.parse(t); if (Array.isArray(j)) return j } catch { /* not a JSON array */ }
  }
  const lines = t.split('\n').map((s) => s.trim()).filter(Boolean)
  if (lines.length && lines.every((l) => l.startsWith('{') || (l.startsWith('"') && l.endsWith('"')))) {
    try { return lines.map((l) => JSON.parse(l)) } catch { /* not JSONL */ }
  }
  return null
}
function recordText(r: unknown, fields?: string[]): string | null {
  if (typeof r === 'string') return r
  if (!r || typeof r !== 'object') return null
  const obj = r as Record<string, unknown>
  const str = (v: unknown) => (typeof v === 'string' ? v : v == null ? null : JSON.stringify(v))
  // ponytail: multi-field examples join with a space so they stay one-per-line when materialized.
  if (fields?.length) return fields.map((f) => str(obj[f])).filter(Boolean).join(' ')
  for (const k of ['text', 'content', 'code', 'prompt', 'input', 'question']) if (typeof obj[k] === 'string') return obj[k] as string
  let best: string | null = null  // auto fallback: the longest string field
  for (const v of Object.values(obj)) if (typeof v === 'string' && (best == null || v.length > best.length)) best = v
  return best
}
function toExamples(text: string, fields?: string[]): string[] {
  const recs = parseRecords(text)
  if (recs) return recs.map((r) => recordText(r, fields)).filter((x): x is string => !!x && !!x.trim())
  return text.split('\n').map((s) => s.trim()).filter(Boolean)
}

type LogEntry = { ts: string; model: string; kind: 'action' | 'result'; text: string; py: string }
function regionPy(r: any): string {
  if (!r) return 'None'
  if (r.kind === 'cell') return `s.locate_cell(${r.layer}, ${JSON.stringify(r.module)})`
  if (r.kind === 'spot') return `s.locate_spot(examples, topk=${r.topk ?? 0.01})`
  return `s.regions[${JSON.stringify(r.name)}]`
}
// ponytail: examples arrays can hit ~1MB (2000 lines) — truncate the log's python column, not the actual WS payload.
function pyList(arr: string[]): string {
  const full = JSON.stringify(arr)
  return full.length <= 2000 ? full : `examples  # ${arr.length} examples (elided — full data in datasets store)`
}
// ponytail: the python equivalent of each WS action — copy the py column to reproduce a session outside the studio.
function actionPy(m: any): string | null {
  switch (m.type) {
    case 'generate': return `list(s.generate_text(${JSON.stringify(m.prompt)}, max_tokens=${m.max_tokens ?? 64}))`
    case 'spot': return `s.compute_spot(${pyList(m.examples)})`
    case 'intervene': return `s.intervene(${regionPy(m.region)}, ${JSON.stringify(m.op ?? 'scale')}, alpha=${m.alpha ?? 0}, key=${JSON.stringify(m.key ?? 'default')})`
    case 'clear': return m.key ? `s.clear(${JSON.stringify(m.key)})` : 's.clear()'
    case 'suspend': return 's.suspend()'
    case 'resume': return 's.resume()'
    case 'ppl': return `s.ppl(${pyList(m.examples)})${m.tag ? `  # ${m.tag}` : ''}`
    case 'save_region': return `s.save_region(${JSON.stringify(m.name)}, ${regionPy(m.region)})`
    case 'import_region': return `s.import_region(${JSON.stringify(m.name)}, ${JSON.stringify(m.path)})`
    case 'train': return `list(s.train_steps(${pyList(m.examples)}, mode=${JSON.stringify(m.mode ?? 'full')}, steps=${m.steps ?? 50}, lr=${m.lr ?? 1e-4}${m.region ? `, region=${regionPy(m.region)}` : ''}))`
    case 'stop_train': return 's.stop()'
    case 'reset_train': return 's.reset_training()'
    default: return null  // catalog/open/close/drilldown/stop — not experiment actions
  }
}
function download(name: string, text: string, type: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([text], { type }))
  a.download = name
  a.click()
  URL.revokeObjectURL(a.href)
}

export default function App() {
  const [prompt, setPrompt] = useState('write a quicksort in python')
  const [theme, setTheme] = useState<'dark' | 'light'>(() => (localStorage.getItem('parametic-theme') === 'light' ? 'light' : 'dark'))
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('parametic-theme', theme) }, [theme])
  const [sync, setSync] = useState(true)
  const [explorerOpen, setExplorerOpen] = useState(true)
  const [explorerW, setExplorerW] = useState(() => Number(localStorage.getItem('parametic-explorer-w')) || 230)
  const [chatW, setChatW] = useState(() => Number(localStorage.getItem('parametic-chat-w')) || 320)
  useEffect(() => { localStorage.setItem('parametic-explorer-w', String(explorerW)); localStorage.setItem('parametic-chat-w', String(chatW)) }, [explorerW, chatW])
  function dragSidebar(e: React.MouseEvent, side: 'left' | 'right') {
    e.preventDefault()
    const x0 = e.clientX, w0 = side === 'left' ? explorerW : chatW
    const move = (ev: MouseEvent) => {
      const d = ev.clientX - x0
      if (side === 'left') setExplorerW(Math.min(480, Math.max(150, w0 + d)))
      else setChatW(Math.min(600, Math.max(200, w0 - d)))
    }
    const up = () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
    window.addEventListener('mousemove', move); window.addEventListener('mouseup', up)
  }
  const [focusModel, setFocusModel] = useState(DEFAULT.id)
  const [open, setOpen] = useState<{ id: string; label: string }[]>([DEFAULT])
  const [catalog, setCatalog] = useState<{ id: string; label: string; installed?: boolean; size_mb?: number | null }[]>([])
  const [data, setData] = useState<Record<string, ModelData>>({ [DEFAULT.id]: empty() })
  const [cols, setCols] = useState<Col[]>([{ id: 1, w: 1, tiles: [{ id: 1, model: DEFAULT.id, tabs: ['output', 'importance', 'attention'], active: 0, h: 1 }] }])
  const [maxTile, setMaxTile] = useState<number | null>(null)  // maximized tile id: fills the whole workspace (hides other panes + chat dock)
  const [ds] = useState(DEFAULT_DATASET)  // seed examples; spot is dataset-driven now, so this never changes
  const [layer, setLayer] = useState(0)
  const [hoverLayer, setHoverLayer] = useState<number | null>(null)  // attention: hover=preview, click=pin
  const [attnCell, setAttnCell] = useState<{ q: number; k: number } | null>(null)  // attention matrix: hovered (query, key) cell
  const [attnZoom, setAttnZoom] = useState(15)  // attention matrix cell size (px) — zoom in/out
  const [activMode, setActivMode] = useState<'cumulative' | 'timeline'>('cumulative')  // activations views: cumulative vs per-token timeline
  type TrainDelta = { mode: string; region_rms: number | null; region_max?: number | null; region_params?: number; other_rms: number | null; other_max?: number | null; other_params?: number }
  const [trainDelta, setTrainDelta] = useState<Record<string, TrainDelta | null>>({})
  const [trainMode, setTrainMode] = useState('spot-freeze')
  const [trainRegion, setTrainRegion] = useState('')
  const [trainRegionTopk, setTrainRegionTopk] = useState(0.01)  // freeze/train the region re-thresholded to this top-k% (≤ its saved base)
  const [trainEpochs, setTrainEpochs] = useState(2)  // passes over the selected examples; steps = epochs × #examples
  const [trainLr, setTrainLr] = useState(2e-5)  // safe fine-tuning default; 1e-4 often diverges (loss climbs) on a 1.5B model
  const [trainDsName, setTrainDsName] = useState('')  // benchmark/dataset to fine-tune on (each record = one example)
  const [trainLimit, setTrainLimit] = useState(30)     // # examples from the dataset to use
  // presets are client-side seeds; server entries live in $PARAMETIC_STUDIO_HOME/datasets (content lazy-fetched)
  const [datasets, setDatasets] = useState<{ name: string; content: string | null; fields?: string[]; server?: boolean; size?: number; link?: string | null }[]>([
    { name: 'python', content: PRESETS.python }, { name: 'java', content: PRESETS.java },
    { name: 'cpp', content: PRESETS.cpp }, { name: 'javascript', content: PRESETS.javascript },
    { name: 'general', content: GENERAL_SET },
  ])
  const [spotN, setSpotN] = useState('')                 // '' = all examples
  const [spotPick, setSpotPick] = useState<'first' | 'random'>('first')
  const [sweepKs, setSweepKs] = useState('0.0025, 0.01, 0.09, 0.25')  // percent values for the Table-1 k-sweep (paper's range)
  const [spotTopk, setSpotTopk] = useState<number>(0.01)  // parameter-region mask size: default top 1%
  const [evalDsName, setEvalDsName] = useState('')       // '' = eval kppl on spot data (dsExamples); else a loaded dataset name
  const [namedRegionPick, setNamedRegionPick] = useState('')   // knob board: "+ from saved region" picker
  const [namedRegionTopk, setNamedRegionTopk] = useState(0.01) // its topk %, capped by the picked region's base_topk
  // P13: pass@k (HumanEvalPack) eval view — separate dataset pick + gen params from the spot kppl "eval on"
  const [codeEvalDsName, setCodeEvalDsName] = useState('')
  const [codeEvalTemp, setCodeEvalTemp] = useState(0)
  const [codeEvalMaxTokens, setCodeEvalMaxTokens] = useState(512)
  const [codeEvalLimit, setCodeEvalLimit] = useState('')  // '' = full dataset
  const sample = (ex: string[]) => {
    const n = Number(spotN)
    if (!n || n >= ex.length) return ex
    if (spotPick === 'first') return ex.slice(0, n)
    const pool = [...ex]                                  // partial Fisher–Yates: n random picks
    for (let i = 0; i < n; i++) { const j = i + Math.floor(Math.random() * (pool.length - i)); [pool[i], pool[j]] = [pool[j], pool[i]] }
    return pool.slice(0, n)
  }
  const [regionInfo, setRegionInfo] = useState<Record<string, { layers: number; modules: string[]; grid: number[][]; importance: number[][] | null; count: number; base_topk?: number; view_topk?: number | null }>>({})
  // Per-region importance reference max: the running-max raw |g×w| cell seen across every top-k% viewed.
  // The heatmap normalizes against this (not each threshold's own max) so lowering the % visibly dims it.
  const [regionImpMax, setRegionImpMax] = useState<Record<string, number>>({})
  // Tensor inspector (layer → tensor → values). Selection + last-fetched window, keyed by model.
  type TensorView = { name: string; shape: number[]; dtype: string; source: string; signed: boolean; rows_total: number; cols_total: number; r0: number; c0: number; values: number[][]; flattened: boolean; stats: { min: number; max: number; mean: number; std: number; absmax: number } }
  // value source: 'weights' (original model θ) | 'importance' (live |g×w|) | a saved region name
  const [tvSource, setTvSource] = useState<Record<string, string>>({})
  const [tvSel, setTvSel] = useState<Record<string, { layer: string; tensor: string | null }>>({})
  const [tvData, setTvData] = useState<Record<string, TensorView>>({})
  const [tvHover, setTvHover] = useState<{ i: number; j: number; v: number } | null>(null)  // hovered heatmap cell → value readout; null when the mouse leaves
  const tvFetched = useRef<Set<string>>(new Set())  // guards the one-shot tensor-list fetch per model in the view
  // Region control tab: pick a saved region and adjust it (op/alpha/topk) with a clean-vs-adjusted PPL compare
  const [ctrlSel, setCtrlSel] = useState<Record<string, string>>({})       // selected region name per model
  const [ctrlOp, setCtrlOp] = useState('zero')                             // scale | zero | mean | random
  const [ctrlAlpha, setCtrlAlpha] = useState(0)                           // scale factor (only used by op=scale)
  const [ctrlTopk, setCtrlTopk] = useState<Record<string, number>>({})     // topk fraction per region (capped by base_topk)
  const [ctrlApplied, setCtrlApplied] = useState<Record<string, { name: string; op: string; alpha: number; topk: number } | null>>({})
  const [ctrlPpl, setCtrlPpl] = useState<Record<string, { cleanCode?: number; cleanGen?: number; adjCode?: number; adjGen?: number }>>({})
  const [ctrlEvalDs, setCtrlEvalDs] = useState('')                         // HumanEval dataset for the benchmark
  const [ctrlEvalLimit, setCtrlEvalLimit] = useState(20)                   // # problems (paper used 20; keeps it fast)
  const [ctrlEval, setCtrlEval] = useState<Record<string, { clean?: { pass_at_1: number; passed: number; total: number }; deleted?: { pass_at_1: number; passed: number; total: number } }>>({})
  const ctrlEvalSlot = useRef<Record<string, 'clean' | 'deleted'>>({})     // routes the next eval_result into clean|deleted for the control tab
  // Train tab benchmark: pass@1 of the model base (pre-train) vs trained. eval fans out over GPUs (unlike training).
  type PassResult = { pass_at_1: number; passed: number; total: number }
  const [trainEval, setTrainEval] = useState<Record<string, { base?: PassResult; trained?: PassResult }>>({})
  const trainEvalSlot = useRef<Record<string, 'base' | 'trained'>>({})
  const [trainEvalDs, setTrainEvalDs] = useState('')
  const [trainEvalLimit, setTrainEvalLimit] = useState(20)
  const [trainEvalGpus, setTrainEvalGpus] = useState<Record<string, number[]>>({})
  const [ctrlEvalGpus, setCtrlEvalGpus] = useState<Record<string, number[]>>({})  // extra GPUs to split the benchmark across
  const ctrlTopkRef = useRef(0.01)                                         // latest slider top-k (avoids stale closure on drag-release re-apply)
  const [compareSel, setCompareSel] = useState<string[]>([])  // selected region names (a toggle set; hue is fixed per region, see regHue)
  const [compareHover, setCompareHover] = useState<string | null>(null)  // shared "L.module" cell — cross-highlights every compare grid
  const [importanceCompareRegion, setImportanceCompareRegion] = useState('')
  const [interMetric, setInterMetric] = useState<'shared' | 'lift' | 'fraction'>('shared')  // intersection grid coloring
  const [compareData, setCompareData] = useState<{ names: string[]; layers: number; modules: string[]; grids: Record<string, number[][]>; kinds: Record<string, string>; intersection: number[][]; intersectionLift: number[][] | null; jaccard: Record<string, number> } | null>(null)
  // parse each dataset ONCE per change — parsing in render paths re-chewed megabytes of JSONL on
  // every token-stream re-render (GB-scale GC churn).
  const dsMeta = useMemo(() => {
    const meta: Record<string, { count: number; examples: string[] }> = {}
    for (const d of datasets) if (d.content != null) { const ex = toExamples(d.content, d.fields); meta[d.name] = { count: ex.length, examples: ex } }
    return meta
  }, [datasets])
  const dsExamples = useMemo(() => toExamples(ds), [ds])  // spot-view editor content, parsed once per edit
  // the exact example array the last spot ran on. region()/save_region reuse THIS, not toExamples(ds):
  // multi-line examples (code) don't survive the ex.join('\n')→toExamples split round-trip, so re-parsing
  // the editor yields a different set → importance cache miss → save recomputes. Keeping the array fixes it.
  const [spotExamples, setSpotExamples] = useState<Record<string, string[]>>({})  // per-model: each keeps the array ITS spot ran on
  // extra GPUs to help compute_spot's backward passes, per model — home GPU is always included, this is on TOP of it
  const [spotExtraGpus, setSpotExtraGpus] = useState<Record<string, number[]>>({})
  const emitSpot = (mid: string, ex: string[]) => {
    setSpotExamples((s) => ({ ...s, [mid]: ex }))
    const extra = (spotExtraGpus[mid] ?? []).map((i) => `cuda:${i}`)
    sendTo(mid, { type: 'spot', examples: ex, ...(extra.length ? { extra_devices: extra } : {}) })
  }
  // activations-spot source, per model: '' = the just-computed spot; else a saved region name (no recompute needed)
  const [spotActRegion, setSpotActRegion] = useState<Record<string, string>>({})
  // region-usage: per-model region source ('' = current spot) + the prompt set to compare
  const [usageRegion, setUsageRegion] = useState<Record<string, string>>({})
  const [usagePrompts, setUsagePrompts] = useState<Record<string, string>>({})
  const [regionName, setRegionName] = useState<Record<string, string>>({})  // per-model "Save region" name input (controlled — was a fragile getElementById)
  const USAGE_DEFAULT = 'write a quicksort in python\nexplain photosynthesis\nsummarize this paragraph\nfix this python bug: def f(x) return x+1'
  // spot is driven by a dataset pick (no free-text editor): pick → sample → compute. content lazy-loads,
  // so remember the pending pick and fire once it lands (see dataset_content handler).
  const [spotSrc, setSpotSrc] = useState('')            // dataset name the spot ran on (shown in the picker)
  const spotPendingRef = useRef<string | null>(null)
  const computeSpotFor = (mid: string, name: string) => {
    const dset = datasets.find((x) => x.name === name); if (!dset) return
    setSpotSrc(name)
    if (dset.content == null) { spotPendingRef.current = name; sendTo(mid, { type: 'read_dataset', name }); return }
    emitSpot(mid, sample(toExamples(dset.content, dset.fields)))
  }
  const [expModels, setExpModels] = useState<Set<string>>(new Set())  // models with their tensor tree expanded
  const [expPaths, setExpPaths] = useState<Set<string>>(new Set())    // expanded folder paths (model-id prefixed)
  // explorer selection: `${section}:${name}` — target of Del key + context menu ('model:'|'data:'|'region:')
  const [selected, setSelected] = useState<string | null>(null)
  const selectedRef = useRef(selected); selectedRef.current = selected
  // open context menu: screen coords + which row it targets. items are rebuilt each render so the
  // armed-delete label ("sure? — click again") stays live while the menu is open.
  const [menu, setMenu] = useState<{ x: number; y: number; target: string } | null>(null)
  const [dragging, setDragging] = useState(false)
  const [overTile, setOverTile] = useState<number | null>(null)
  const nextId = useRef(2)
  const drag = useRef<{ tid: number; idx: number } | null>(null)
  const sockets = useRef<Record<string, WebSocket>>({})
  const rowRef = useRef<HTMLDivElement>(null)
  // log panel autoscroll: pin to bottom on new entries, unless the user has scrolled up
  const logScrollRef = useRef<HTMLDivElement>(null)
  const logPinned = useRef(true)
  const abPhase = useRef<Record<string, 'base' | 'inter'>>({})  // A/B sequencing (refs: onmessage closure is created once)
  const promptImportancePending = useRef<Record<string, string>>({})
  const generatedText = useRef<Record<string, string>>({})
  const promptRef = useRef(prompt); promptRef.current = prompt
  const [genOpen, setGenOpen] = useState(false)
  const [maxTokens, setMaxTokens] = useState(256)
  const [temperature, setTemperature] = useState(0)
  const [probesOn, setProbesOn] = useState<Record<string, boolean>>({ attention: true, activation: true, logitlens: true, spot_activation: true })
  const genRef = useRef({ maxTokens, temperature }); genRef.current = { maxTokens, temperature }
  // kernel liveness: null=connecting, true=up, false=down (reconnecting with backoff)
  const [kernelUp, setKernelUp] = useState<boolean | null>(null)
  const [kernelStats, setKernelStats] = useState<{ rss_mb: number } | null>(null)
  // first-run kernel setup (§0/T3): if the local kernel doesn't connect in ~6s and deps are missing,
  // show an overlay to pick a Python + pip-install requirements-studio.txt, then respawn the kernel.
  const [setupOpen, setSetupOpen] = useState(false)
  const [envStatus, setEnvStatus] = useState<EnvStatus | null>(null)
  const [pythons, setPythons] = useState<PyInfo[] | null>(null)  // null = scanning
  const [selectedPy, setSelectedPy] = useState('')  // realpath of the selected candidate
  const [installing, setInstalling] = useState(false)
  const [installLog, setInstallLog] = useState<string[]>([])
  const [installExit, setInstallExit] = useState<{ ok: boolean; code: number } | null>(null)
  const [remoteStale, setRemoteStale] = useState(false)  // remote (SSH) kernel unreachable → banner, not the overlay
  // P16: GPU inventory — count + per-device mem/model occupancy, for the status bar and the load-device picker
  const [gpus, setGpus] = useState<{ count: number; devices: { index: number; name: string; mem_used_mb: number; mem_total_mb: number; models: string[] }[] } | null>(null)
  const [openDevice, setOpenDevice] = useState('')  // '' = Auto; else 'cuda:N' — picked in the "+ model" row
  // settings panel (menu-driven; browser gets a small status-bar button) — config + cached-model management
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [config, setConfig] = useState<Record<string, string>>({})
  // remote kernel: URL/token editable in settings, applied via reload (see WS_URL/WS_TOKEN module consts)
  const [kernelUrlInput, setKernelUrlInput] = useState(WS_URL)
  const [kernelTokenInput, setKernelTokenInput] = useState(WS_TOKEN)
  // P7 remote (SSH) kernel: Local|Remote mode toggle + connect form. Boots into 'remote' if already
  // tunneled (ps_kernel_url points at the SSH tunnel port) so reload keeps showing Disconnect.
  const [kernelMode, setKernelMode] = useState<'local' | 'remote'>(isRemoteConnected() ? 'remote' : 'local')
  // non-secret SSH connection fields persist to localStorage so a restart can reconnect without
  // re-entering everything; password/passphrase are state-only and never persisted (see below).
  const ls = (k: string, d = '') => localStorage.getItem(k) || d
  const [sshHost, setSshHost] = useState(() => ls('ps_ssh_host'))
  const [sshPort, setSshPort] = useState(() => ls('ps_ssh_port', '22'))
  const [sshUser, setSshUser] = useState(() => ls('ps_ssh_user'))
  const [sshAuth, setSshAuth] = useState<'password' | 'key'>(() => (ls('ps_ssh_auth') === 'key' ? 'key' : 'password'))
  const [sshPassword, setSshPassword] = useState('')  // state only — never persisted
  const [sshKeyPath, setSshKeyPath] = useState(() => ls('ps_ssh_key_path'))  // e.g. ~/.ssh/gpu.pem — not sensitive
  const [sshKeyPassphrase, setSshKeyPassphrase] = useState('')  // state only — never persisted
  const [sshRepoDir, setSshRepoDir] = useState(() => ls('ps_ssh_repo_dir'))
  const [sshModel, setSshModel] = useState(() => ls('ps_ssh_model'))
  const [sshPythonPath, setSshPythonPath] = useState(() => ls('ps_ssh_python_path'))  // e.g. /opt/conda/bin/python — GPU boxes keep torch in a non-default python
  const [sshHfHome, setSshHfHome] = useState(() => ls('ps_ssh_hf_home'))  // remote HF cache dir — point at a roomy volume (e.g. /shared/...) so big models don't fill the home disk
  useEffect(() => {
    const kv: Record<string, string> = {
      ps_ssh_host: sshHost, ps_ssh_port: sshPort, ps_ssh_user: sshUser, ps_ssh_auth: sshAuth,
      ps_ssh_key_path: sshKeyPath, ps_ssh_repo_dir: sshRepoDir, ps_ssh_model: sshModel,
      ps_ssh_python_path: sshPythonPath, ps_ssh_hf_home: sshHfHome,
    }
    for (const [k, v] of Object.entries(kv)) localStorage.setItem(k, v)
  }, [sshHost, sshPort, sshUser, sshAuth, sshKeyPath, sshRepoDir, sshModel, sshPythonPath, sshHfHome])
  const [sshConnecting, setSshConnecting] = useState(false)
  const [sshStatus, setSshStatus] = useState<{ state: string; detail?: string } | null>(null)
  const [installed, setInstalled] = useState<{ id: string; size_mb: number | null }[] | null>(null)
  // error toasts: stack of {id, text}, 5s auto-dismiss + click-to-dismiss, capped at 4 (oldest dropped)
  const [toasts, setToasts] = useState<{ id: number; text: string }[]>([])
  const toastId = useRef(0)
  const toast = (text: string) => setToasts((ts) => [...ts.slice(-3), { id: toastId.current++, text }])
  const dismissToast = (id: number) => setToasts((ts) => ts.filter((t) => t.id !== id))
  // pending registry: `${op}:${mid}` in flight since sendTo, cleared on matching response or error
  const [pending, setPending] = useState<Set<string>>(new Set())
  const pendingRef = useRef<Set<string>>(new Set())
  const setPendingKey = (key: string, on: boolean) => {
    const next = new Set(pendingRef.current)
    on ? next.add(key) : next.delete(key)
    pendingRef.current = next
    setPending(next)
  }
  // locate_progress: op → {i, total}, latest value while in flight, cleared when the op's response lands
  const [locateProg, setLocateProg] = useState<Record<string, { i: number; total: number }>>({})
  const reconnectAttempts = useRef(0)
  const reconnectTimer = useRef<number | null>(null)
  const splashClosed = useRef(false)  // fire close_splash exactly once (first kernelUp or 8s timeout)
  const authFailToasted = useRef(false)  // show the auth-failed toast once, not on every reconnect retry
  const openRef = useRef(open); openRef.current = open
  const kernelUpRef = useRef(kernelUp); kernelUpRef.current = kernelUp  // read latest liveness inside the mount-once 6s trigger
  const installLogRef = useRef<HTMLDivElement | null>(null)  // setup log box — auto-scroll to bottom on new pip lines
  const bootedRef = useRef(false)  // first catalog vs a reconnect resync — only the first adopts the kernel default
  function scheduleReconnect() {
    if (reconnectTimer.current != null) return
    const delay = Math.min(30000, 1000 * 2 ** reconnectAttempts.current)
    reconnectTimer.current = window.setTimeout(() => {
      reconnectTimer.current = null
      reconnectAttempts.current++
      const probe = new WebSocket(WS_URL)
      probe.onopen = () => {
        probe.close()
        setKernelUp(true)
        reconnectAttempts.current = 0
        // resync: fresh kernel has no sessions/state for us — reload models, catalog re-pulls regions+datasets
        sendTo(openRef.current[0]?.id ?? DEFAULT.id, { type: 'catalog' })
        sendTo(openRef.current[0]?.id ?? DEFAULT.id, { type: 'gpus' })
        for (const m of openRef.current) { patch(m.id, (d) => ({ ...d, loading: true })); sendTo(m.id, { type: 'open' }) }
      }
      probe.onerror = () => { try { probe.close() } catch { /* already closed */ } scheduleReconnect() }
    }, delay)
  }
  // after a kernel respawn: drop any 30s backoff so the fresh kernel is picked up in ~1s (not internals-changing — just resets the timer).
  const resumeReconnect = () => {
    reconnectAttempts.current = 0
    if (reconnectTimer.current) { clearTimeout(reconnectTimer.current); reconnectTimer.current = null }
    scheduleReconnect()
  }
  // webview-safe dialogs: inline inputs replace window.prompt, two-step "sure?" replaces window.confirm
  const [asking, setAsking] = useState<'hf' | 'editor' | 'path' | 'hf-dataset' | 'import-region' | null>(null)
  const [askValue, setAskValue] = useState('')
  const [askSplit, setAskSplit] = useState('')
  const [askConfig, setAskConfig] = useState('')          // HF dataset config (e.g. humanevalpack language)
  const [askFilter, setAskFilter] = useState('')           // 'col=value' row filter (e.g. tiny-codes programming_language=Python)
  const [askRegionPath, setAskRegionPath] = useState('')   // import-region: folder of per-param .pt masks
  const [hfLoading, setHfLoading] = useState<string | null>(null)  // repo id currently loading, for the Data section hint
  const [uploading, setUploading] = useState<string[]>([])  // dataset file names being uploaded to the kernel store (+ File/+ Folder)
  const [armed, setArmed] = useState<string | null>(null)
  const armedTimer = useRef<number | null>(null)
  function confirmClick(key: string, action: () => void) {
    if (armed === key) { setArmed(null); if (armedTimer.current) clearTimeout(armedTimer.current); action(); return }
    setArmed(key)
    if (armedTimer.current) clearTimeout(armedTimer.current)
    armedTimer.current = window.setTimeout(() => setArmed(null), 3000)  // arm expires
  }
  const loadStart = useRef<Record<string, number>>({})   // model id → load start ts (for the chip progress bar)
  const [tick, setTick] = useState(0)
  const anyLoading = open.some((m) => data[m.id]?.loading)
  useEffect(() => {
    if (!anyLoading) return
    const iv = setInterval(() => setTick((t) => t + 1), 1000)  // re-render each second while loading (elapsed counter)
    return () => clearInterval(iv)
  }, [anyLoading])
  void tick
  const [expLog, setExpLog] = useState<LogEntry[]>([])
  const [logPy, setLogPy] = useState(false)
  const logEntry = (model: string, kind: LogEntry['kind'], text: string, py = '') =>
    setExpLog((l) => [...l, { ts: new Date().toISOString(), model, kind, text, py }])
  useEffect(() => {
    const el = logScrollRef.current
    if (el && logPinned.current) el.scrollTop = el.scrollHeight
  }, [expLog, logPy])

  function patch(mid: string, f: (d: ModelData) => ModelData) { setData((all) => ({ ...all, [mid]: f(all[mid] ?? empty()) })) }
  function handleMessage(e: MessageEvent) {
    const m = JSON.parse(e.data)
    if (m.type === 'catalog') {
      setCatalog(m.models)
      const def = m.models.find((x: { id: string; label: string }) => x.id === m.default)
      // adopt the kernel's default model ONLY on the very first catalog. On a reconnect resync this must
      // NOT run — it would wipe data (incl. the computed spot) for every open model. (bug: spot vanished
      // after a mid-inference WS reconnect.)
      if (!bootedRef.current && def && def.id !== DEFAULT.id) {  // kernel booted with a different model — follow it if the UI is untouched
        setOpen((o) => (o.length === 1 && o[0].id === DEFAULT.id ? [def] : o))
        setData((all) => (all[DEFAULT.id] && !all[DEFAULT.id].count ? { [def.id]: empty() } : all))
        setCols((cs) => cs.map((c) => ({ ...c, tiles: c.tiles.map((t) => (t.model === DEFAULT.id ? { ...t, model: def.id } : t)) })))
        setFocusModel((f) => (f === DEFAULT.id ? def.id : f))
      }
      bootedRef.current = true
      sendTo(def?.id ?? DEFAULT.id, { type: 'regions' })   // disk-persisted regions appear on startup
      sendTo(def?.id ?? DEFAULT.id, { type: 'datasets' })  // …and the kernel-side dataset store
      return
    }
    if (m.type === 'datasets') {
      setDatasets((dd) => {
        const server = (m.items as { name: string; size: number; link: string | null }[]).map((it) => {
          const prev = dd.find((x) => x.name === it.name)
          return { name: it.name, size: it.size, link: it.link, server: true, content: prev?.content ?? null, fields: prev?.fields }
        })
        const names = new Set(server.map((s) => s.name))
        return [...dd.filter((x) => !x.server && !names.has(x.name)), ...server]  // client presets + server store
      })
      setHfLoading(null)
      return
    }
    if (m.type === 'dataset_content') {
      setDatasets((dd) => dd.map((x) => (x.name === m.name ? { ...x, content: m.content } : x)))
      if (spotPendingRef.current === m.name) {  // a spot pick was waiting on this content → compute now
        spotPendingRef.current = null
        const fields = datasets.find((x) => x.name === m.name)?.fields
        emitSpot(focused(), sample(toExamples(m.content, fields)))
      }
      return
    }
    if (m.type === 'dataset_saved') { setUploading((u) => u.filter((x) => x !== m.name)); sendTo(m.model, { type: 'datasets' }); return }
    if (m.type === 'loading_dataset') { setHfLoading(m.repo); return }
    if (m.type === 'stats') { setKernelStats({ rss_mb: m.rss_mb }); return }
    if (m.type === 'gpus') { setGpus({ count: m.count, devices: m.devices ?? [] }); return }
    if (m.type === 'config') { setConfig(m.config ?? {}); return }
    if (m.type === 'installed_models') { setInstalled(m.items ?? []); return }
    if (m.type === 'cached_deleted') { sendTo(focused(), { type: 'installed_models' }); return }
    const mid = m.model
    if (m.type === 'opened') { delete loadStart.current[mid]; patch(mid, (d) => ({ ...d, loading: false, download: null })); sendTo(mid, { type: 'regions' }); return }
    if (m.type === 'loading') { loadStart.current[mid] ??= Date.now(); patch(mid, (d) => ({ ...d, loading: true })); return }
    if (m.type === 'download_progress') { patch(mid, (d) => ({ ...d, download: { pct: m.pct, done_mb: m.done_mb, total_mb: m.total_mb } })); return }
    if (m.type === 'loading_weights') { patch(mid, (d) => ({ ...d, download: null })); return }  // download done, GPU load begins → drop to indeterminate
    if (m.type === 'load_failed') { delete loadStart.current[mid]; toast(`[open] ${openRef.current.find((o) => o.id === mid)?.label ?? mid} failed to load`); closeModel(mid); return }
    if (m.type === 'token') {
      generatedText.current[mid] = (generatedText.current[mid] ?? '') + m.text
      patch(mid, (d) => ({ ...d, output: d.output + m.text, count: d.count + 1, tokenTexts: [...d.tokenTexts, m.text] }))
    }
    else if (m.type === 'prompt_tokens') patch(mid, (d) => ({ ...d, promptTokens: m.tokens }))
    else if (m.type === 'attention') patch(mid, (d) => ({ ...d, frames: [...d.frames, m.data] }))
    else if (m.type === 'activation') patch(mid, (d) => {
      const detail = m.detail ? { layers: m.detail.layers, modules: m.detail.modules, grid: m.detail.grid } : null
      return {
        ...d,
        act: m.data,
        actDetail: detail ?? d.actDetail,
        actFrames: [...d.actFrames, m.data],
        actDetailFrames: detail ? [...d.actDetailFrames, detail.grid] : d.actDetailFrames,
      }
    })
    else if (m.type === 'spot_activation') patch(mid, (d) => ({ ...d, spotAct: { layers: m.layers, modules: m.modules, grid: m.grid }, spotActFrames: [...d.spotActFrames, m.grid] }))
    else if (m.type === 'logitlens') patch(mid, (d) => ({ ...d, logit: m.layers }))
    else if (m.type === 'spot_progress') patch(mid, (d) => ({ ...d, ...(m.grid ? { spot: { layers: m.layers, modules: m.modules, grid: m.grid } } : {}), spotProg: { i: m.i, total: m.total } }))  // multi-GPU runs omit the live grid (see spot_activation_grid docstring) — keep d.spot as-is then
    else if (m.type === 'spotmap') { patch(mid, (d) => ({ ...d, spot: { layers: m.layers, modules: m.modules, grid: m.grid }, spotProg: null, concentration: null, contrast: null })); sendTo(mid, { type: 'concentration' }) }  // fresh spot → pull its concentration curve, drop stale contrast
    else if (m.type === 'prompt_importance_progress') patch(mid, (d) => ({ ...d, ...(m.grid ? { promptImportance: { layers: m.layers, modules: m.modules, grid: m.grid } } : {}), promptImportanceProg: { i: m.i, total: m.total } }))
    else if (m.type === 'prompt_importance') patch(mid, (d) => ({ ...d, promptImportance: { layers: m.layers, modules: m.modules, grid: m.grid }, promptImportanceProg: null }))
    else if (m.type === 'concentration') patch(mid, (d) => ({ ...d, concentration: m.curve }))
    else if (m.type === 'contrast_progress') patch(mid, (d) => ({ ...d, contrastProg: m.stage, contrast: { ...(d.contrast ?? { topk: 0 }), [m.stage]: { code_ppl: m.code_ppl, general_ppl: m.general_ppl } } }))
    else if (m.type === 'contrast_result') { setPendingKey(`causal_contrast:${mid}`, false); patch(mid, (d) => ({ ...d, contrastProg: null, contrast: { topk: m.topk, clean: m.clean, spot: m.spot, random: m.random, bottom: m.bottom } })) }
    else if (m.type === 'sweep_progress') patch(mid, (d) => ({ ...d, sweepProg: m.cond === 'clean' ? 'clean' : `${(m.topk * 100).toFixed(4)}% · ${m.cond}`, sweep: m.cond === 'clean' ? { ...(d.sweep ?? { topks: [], rows: [] }), clean: { code_ppl: m.code_ppl, general_ppl: m.general_ppl } } : { ...(d.sweep ?? { topks: [], rows: [] }), rows: [...(d.sweep?.rows ?? []), { topk: m.topk, cond: m.cond, code_ppl: m.code_ppl, general_ppl: m.general_ppl }] } }))
    else if (m.type === 'sweep_result') { setPendingKey(`causal_sweep:${mid}`, false); patch(mid, (d) => ({ ...d, sweepProg: null, sweep: { topks: m.topks, clean: m.clean, rows: m.rows } })) }
    else if (m.type === 'usage_row') patch(mid, (d) => ({ ...d, usage: { ...d.usage, rows: [...d.usage.rows.filter((r) => r.prompt !== m.prompt), { prompt: m.prompt, overall: m.overall }] } }))
    else if (m.type === 'usage_batch_done') { setPendingKey(`region_usage_batch:${mid}`, false); patch(mid, (d) => ({ ...d, usage: { ...d.usage, running: false } })) }
    else if (m.type === 'region_usage') { setPendingKey(`region_usage:${mid}`, false); patch(mid, (d) => ({ ...d, usage: { ...d.usage, detail: { prompt: m.prompt, overall: m.overall, per_layer: m.per_layer, tokens: m.tokens } } })) }
    else if (m.type === 'ppl') {
      setPendingKey(`ppl:${mid}`, false)
      logEntry(mid, 'result', `ppl${m.tag ? `[${m.tag}]` : ''} = ${Number(m.value).toPrecision(5)}`)
      if (m.tag === 'code' || m.tag === 'general') patch(mid, (d) => ({ ...d, evals: { ...d.evals, [m.tag]: m.value } }))
      else if (typeof m.tag === 'string' && m.tag.startsWith('ctrl:')) { const key = m.tag.slice(5); setCtrlPpl((c) => ({ ...c, [mid]: { ...c[mid], [key]: m.value } })) }
      else patch(mid, (d) => ({ ...d, kppl: { ...d.kppl, [m.tag === 'base' ? 'base' : 'inter']: m.value } }))
    }
    else if (m.type === 'intervened') { setPendingKey(`intervene:${mid}`, false); setLocateProg((p) => { const n = { ...p }; delete n[`${mid}:intervene`]; return n }) }
    else if (m.type === 'region_saved') {
      setPendingKey(`save_region:${mid}`, false); setPendingKey(`import_region:${mid}`, false)
      setLocateProg((p) => { const n = { ...p }; delete n[`${mid}:save_region`]; return n })
      const detail = m.imported != null ? ` · imported ${m.imported} param(s)${m.skipped?.length ? `, skipped ${m.skipped.length}` : ''}` : ''
      logEntry(mid, 'result', `region "${m.name}" saved (${m.count} weights)${detail}`); sendTo(mid, { type: 'regions' })
    }
    else if (m.type === 'region_info') {
      setPendingKey(`region_info:${mid}`, false)
      setRegionInfo((ri) => ({ ...ri, [m.name]: { layers: m.layers, modules: m.modules, grid: m.grid, importance: m.importance ?? null, count: m.count, base_topk: m.base_topk, view_topk: m.view_topk ?? null } }))
      if (m.importance) {  // widen the fixed normalization reference to the largest importance mass seen (≈ base top-k%)
        const rawMax = (m.importance as number[][]).flat().reduce((a: number, v: number) => Math.max(a, Number(v) || 0), 0)
        setRegionImpMax((rm) => ({ ...rm, [m.name]: Math.max(rm[m.name] ?? 0, rawMax) }))
      }
    }
    else if (m.type === 'region_comparison') { setPendingKey(`region_compare:${mid}`, false); setCompareData({ names: m.names, layers: m.layers, modules: m.modules, grids: m.grids, kinds: m.kinds ?? {}, intersection: m.intersection, intersectionLift: m.intersection_lift ?? null, jaccard: m.jaccard }) }
    else if (m.type === 'regions') patch(mid, (d) => ({ ...d, regions: m.regions }))
    else if (m.type === 'train_step') patch(mid, (d) => ({ ...d, train: { ...d.train, losses: [...d.train.losses, m.loss], total: m.total, running: true } }))
    else if (m.type === 'trained') {
      logEntry(mid, 'result', `trained mode=${m.mode} steps=${m.steps} (${m.reason})`)
      // snapshot the pre-train evals as "before", then re-measure → evals become "after"
      patch(mid, (d) => ({ ...d, train: { ...d.train, running: false, trained: true, before: d.train.before ?? { ...d.evals } } }))
      sendTo(mid, { type: 'ppl', examples: PRESETS.python.split('\n').filter(Boolean), tag: 'code' })
      sendTo(mid, { type: 'ppl', examples: GENERAL_SET.split('\n').filter(Boolean), tag: 'general' })
    }
    else if (m.type === 'train_reset') { setTrainDelta((td) => ({ ...td, [mid]: null })); patch(mid, (d) => ({ ...d, train: emptyTrain() })) }
    else if (m.type === 'training_delta') { setPendingKey(`training_delta:${mid}`, false); setTrainDelta((td) => ({ ...td, [mid]: m.delta })) }
    else if (m.type === 'tensors') patch(mid, (d) => ({ ...d, tensors: m.tensors }))
    else if (m.type === 'tensor_values') { setPendingKey(`tensor_values:${mid}`, false); setTvData((tv) => ({ ...tv, [mid]: { name: m.name, shape: m.shape, dtype: m.dtype, source: m.source, signed: m.signed, rows_total: m.rows_total, cols_total: m.cols_total, r0: m.r0, c0: m.c0, values: m.values, flattened: m.flattened, stats: m.stats } })) }
    else if (m.type === 'locate_progress') setLocateProg((p) => ({ ...p, [`${mid}:${m.op}`]: { i: m.i, total: m.total } }))
    else if (m.type === 'error') {
      toast(`[${m.op ?? 'kernel'}] ${m.reason}`)
      if (m.op === 'load_hf_dataset') setHfLoading(null)
      if (m.op === 'save_dataset') setUploading([])  // error carries no name → clear the whole batch
      if (m.op) { setPendingKey(`${m.op}:${mid}`, false); setLocateProg((p) => { const n = { ...p }; delete n[`${mid}:${m.op}`]; return n }) }
      if (m.op === 'train') {
        // a stale training left on the kernel → auto-clear it so the next Train works (older kernels raise this)
        if (typeof m.reason === 'string' && m.reason.includes('reset_training')) { sendTo(mid, { type: 'reset_train' }); patch(mid, (d) => ({ ...d, train: { ...emptyTrain(), error: 'cleared a leftover training — press Train again' } })) }
        else patch(mid, (d) => ({ ...d, train: { ...d.train, running: false, error: m.reason } }))
      }
      if (m.op === 'eval_code') patch(mid, (d) => ({ ...d, evalProg: null }))
      if (m.op === 'region_usage_batch') patch(mid, (d) => ({ ...d, usage: { ...d.usage, running: false } }))  // clear the spinner on error
    }
    else if (m.type === 'perhead') { setPendingKey(`drilldown:${mid}`, false); patch(mid, (d) => ({ ...d, perhead: { layer: m.layer, data: m.data } })) }
    else if (m.type === 'eval_progress') patch(mid, (d) => ({ ...d, evalProg: { i: m.i, total: m.total, passed: m.passed } }))
    else if (m.type === 'eval_result') {
      setPendingKey(`eval_code:${mid}`, false)
      logEntry(mid, 'result', `eval[${m.dataset}] pass@1 = ${(m.pass_at_1 * 100).toFixed(2)}% (${m.passed}/${m.total})`)
      const slot = ctrlEvalSlot.current[mid]  // benchmark launched from the Region control tab → route into clean|deleted
      if (slot) { setCtrlEval((c) => ({ ...c, [mid]: { ...c[mid], [slot]: { pass_at_1: m.pass_at_1, passed: m.passed, total: m.total } } })); delete ctrlEvalSlot.current[mid] }
      const tslot = trainEvalSlot.current[mid]  // benchmark launched from the Train tab → route into base|trained
      if (tslot) { setTrainEval((c) => ({ ...c, [mid]: { ...c[mid], [tslot]: { pass_at_1: m.pass_at_1, passed: m.passed, total: m.total } } })); delete trainEvalSlot.current[mid] }
      patch(mid, (d) => {
        const damaged = d.knobs.length > 0
        const result = { dataset: m.dataset, passed: m.passed, total: m.total, pass_at_1: m.pass_at_1, damaged, knobCount: d.knobs.length }
        return { ...d, evalProg: null, evalResult: result, evalPrev: d.evalResult }
      })
    }
    else if (m.type === 'done') {
      const ph = abPhase.current[mid]
      if (ph === 'base') {        // A/B: baseline run finished → stash it, re-apply knobs, run intervened
        abPhase.current[mid] = 'inter'
        patch(mid, (d) => ({ ...d, ab: { ...d.ab, base: d.output }, output: '' }))
        sendTo(mid, { type: 'resume' })
        sendTo(mid, { type: 'generate', prompt: promptRef.current, max_tokens: genRef.current.maxTokens, temperature: genRef.current.temperature, probes: [] })
        return
      }
      if (ph === 'inter') {       // A/B: intervened run finished
        delete abPhase.current[mid]
        patch(mid, (d) => ({ ...d, ab: { ...d.ab, inter: d.output }, busy: false }))
        return
      }
      patch(mid, (d) => {
        if (d.frames.length > 32) sendTo(mid, { type: 'save_run', run: { prompt: promptRef.current, output: d.output, frames: d.frames, act: d.act, logit: d.logit, settings: { maxTokens: genRef.current.maxTokens, temperature: genRef.current.temperature }, reason: m.reason, ts: Date.now() } })
        return { ...d, busy: false }
      })
      const pendingPrompt = promptImportancePending.current[mid]
      if (pendingPrompt) {
        delete promptImportancePending.current[mid]
        sendTo(mid, { type: 'prompt_importance', prompt: pendingPrompt, completion: generatedText.current[mid] ?? '' })
      }
    }
    else if (m.type === 'run_saved') patch(mid, (d) => ({ ...d, lastRunId: m.id, framesFlushed: true, frames: d.frames.slice(-1) }))
    else if (m.type === 'run_data') patch(mid, (d) => ({ ...d, frames: m.run.frames ?? d.frames, framesFlushed: false }))
  }
  function socket(mid: string) {
    let s = sockets.current[mid]
    if (!s || s.readyState > WebSocket.OPEN) {
      s = new WebSocket(WS_URL); sockets.current[mid] = s; s.onmessage = handleMessage
      const sock = s as WebSocket & { _intentional?: boolean }
      if (WS_TOKEN) s.addEventListener('open', () => s.send(JSON.stringify({ type: 'auth', token: WS_TOKEN })), { once: true })
      sock.onopen = () => { setKernelUp(true); reconnectAttempts.current = 0 }
      sock.onclose = (e) => {
        if (sock._intentional) return  // unload closes are intentional
        if (e.code === 4401 && !authFailToasted.current) { authFailToasted.current = true; toast('[kernel] auth failed — check token in settings') }
        setKernelUp(false); scheduleReconnect()
      }
    }
    return s
  }
  function sendTo(mid: string, msg: object) {
    const s = socket(mid); const m = { ...msg, model: mid }
    const type = (m as unknown as { type: string }).type
    const py = actionPy(m)
    if (py) logEntry(mid, 'action', type, py)  // every experiment action lands in the log
    if (PENDING_OPS.has(type)) setPendingKey(`${type}:${mid}`, true)
    if (s.readyState === WebSocket.OPEN) s.send(JSON.stringify(m))
    else s.addEventListener('open', () => s.send(JSON.stringify(m)), { once: true })
  }
  useEffect(() => { sendTo(DEFAULT.id, { type: 'catalog' }); sendTo(DEFAULT.id, { type: 'gpus' }) }, [])
  // P7: subscribe once to SSH tunnel progress from the Rust side (no-op outside Tauri).
  useEffect(() => {
    let unlisten: (() => void) | undefined
    let cancelled = false
    tauriListen('ssh-status', (payload) => {
      let parsed: { state: string; detail?: string }
      try { parsed = JSON.parse(payload) } catch { return }
      setSshStatus(parsed)
      if (parsed.state === 'error') toast(`[ssh] ${parsed.detail ?? 'connection failed'}`)
    }).then((fn) => { if (cancelled) fn(); else unlisten = fn })
    return () => { cancelled = true; unlisten?.() }
  }, [])
  // T3: first-run kernel setup. Desktop app only — browser keeps today's reconnect-only behavior.
  // rescan() re-probes the available Pythons and preselects the first ready one (else the first entry).
  const rescan = () => {
    setPythons(null)
    tauriInvokeValue<PyInfo[]>('discover_pythons').then((list) => {
      setPythons(list)
      const pick = list.find((p) => p.ready) ?? list[0]
      if (pick) setSelectedPy(pick.path)
    }).catch((e) => { setPythons([]); toast(`[setup] ${e instanceof Error ? e.message : String(e)}`) })
  }
  // If the local kernel is down and its deps are missing, open the setup overlay. Fired from TWO
  // sources: (1) a 6s setTimeout — the fast path when the window is already visible; (2) the Rust
  // `check-setup` event. The event is the RELIABLE path: the main window is hidden at boot and
  // hidden-webview JS timers are suspended (see lib.rs), so when the kernel never comes up the window
  // stays hidden the full 8s and the setTimeout never fires — exactly the first-run case this is for.
  // Rust emits check-setup right after the 8s splash fallback shows the window, when JS is live again.
  const maybeOpenSetup = async () => {
    if (!inTauri()) return
    if (kernelUpRef.current === true) return  // kernel already came up — nothing to do
    if (isRemoteConnected()) { setRemoteStale(true); return }  // remote mode → banner, don't fight the SSH flow
    try {
      const st = await tauriInvokeValue<EnvStatus>('kernel_env_status')
      if (!st.deps_ok) { setEnvStatus(st); setSetupOpen(true); rescan() }
      // deps_ok but not up yet ⇒ transient outage; the "reconnecting…" status already covers it.
    } catch { /* command unavailable — leave the normal reconnect flow alone */ }
  }
  useEffect(() => {
    if (!inTauri()) return  // browser: unchanged
    const t = window.setTimeout(() => { void maybeOpenSetup() }, 6000)
    let unlisten: (() => void) | undefined
    let cancelled = false
    tauriListen('check-setup', () => { void maybeOpenSetup() })
      .then((fn) => { if (cancelled) fn(); else unlisten = fn })
    return () => { clearTimeout(t); cancelled = true; unlisten?.() }
  }, [])
  // kernel came up (fresh respawn or otherwise) ⇒ tear down the overlay/banner/installing state.
  useEffect(() => {
    if (kernelUp) { setSetupOpen(false); setRemoteStale(false); setInstalling(false) }
  }, [kernelUp])
  // T3: pip install progress + completion (mount-once, same pattern as ssh-status; no-op outside Tauri).
  useEffect(() => {
    let unlisten: (() => void) | undefined
    let cancelled = false
    tauriListen('deps-progress', (line) => setInstallLog((l) => [...l.slice(-499), line]))
      .then((fn) => { if (cancelled) fn(); else unlisten = fn })
    return () => { cancelled = true; unlisten?.() }
  }, [])
  useEffect(() => {
    let unlisten: (() => void) | undefined
    let cancelled = false
    tauriListen('deps-done', (payload) => {
      let r: { ok: boolean; code: number }
      try { r = JSON.parse(payload) } catch { return }
      setInstalling(false)
      setInstallExit(r)
      if (r.ok) {
        tauriInvokeResult('respawn_kernel').catch((e) => toast(`[setup] ${e instanceof Error ? e.message : String(e)}`))
        resumeReconnect()  // kill the 30s backoff so the fresh kernel is picked up in ~1s
      }
    }).then((fn) => { if (cancelled) fn(); else unlisten = fn })
    return () => { cancelled = true; unlisten?.() }
  }, [])
  useEffect(() => { const el = installLogRef.current; if (el) el.scrollTop = el.scrollHeight }, [installLog])
  useEffect(() => {
    if (!kernelUp) return
    const iv = setInterval(() => { sendTo(focused(), { type: 'stats' }); sendTo(focused(), { type: 'gpus' }) }, 15000)
    return () => clearInterval(iv)
  }, [kernelUp])
  // splash → main: hand off when the kernel first comes up, or after 8s regardless (Tauri only; no-op in browser)
  const closeSplash = () => { if (splashClosed.current) return; splashClosed.current = true; tauriInvoke('close_splash') }
  useEffect(() => { if (kernelUp) closeSplash() }, [kernelUp])
  useEffect(() => { const t = window.setTimeout(closeSplash, 8000); return () => clearTimeout(t) }, [])
  const anyBusy = Object.values(data).some((d) => d.busy)
  // Esc = stop generation, except while typing in an input/textarea
  useEffect(() => {
    if (!anyBusy) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return
      stop()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [anyBusy])
  // explorer keyboard: Del/Backspace = delete selected row (via armed confirm) · Cmd/Ctrl+S = save active data editor
  const colsRef = useRef(cols); colsRef.current = cols
  const datasetsRef = useRef(datasets); datasetsRef.current = datasets
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target as HTMLElement)?.isContentEditable) return
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedRef.current) { e.preventDefault(); deleteSelected() }
      else if ((e.metaKey || e.ctrlKey) && (e.key === 's' || e.key === 'S')) {
        let name: string | null = null
        for (const c of colsRef.current) for (const t of c.tiles) { const v = t.tabs[t.active]; if (v?.startsWith('data:')) name = v.slice(5) }
        if (!name) return  // no data editor active → let the browser handle it
        e.preventDefault()
        const dset = datasetsRef.current.find((x) => x.name === name)
        if (dset?.content != null) sendTo(focused(), { type: 'save_dataset', name, content: dset.content })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])  // deps via refs — handler reads latest cols/datasets/selected without re-binding
  // native menubar → UI actions (Tauri only)
  useEffect(() => {
    let off = () => {}
    tauriListen('menu', (id) => {
      if (id === 'settings') setSettingsOpen(true)
      else if (id === 'load-model') { setAsking('hf'); setAskValue('') }
      else if (id === 'unload-all') { for (const m of openRef.current) closeModel(m.id) }
      else if (id === 'toggle-theme') setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
      else if (id === 'toggle-explorer') setExplorerOpen((v) => !v)
    }).then((fn) => { off = fn })
    return () => off()
  }, [])
  // fetch config + installed list whenever the settings panel opens
  useEffect(() => { if (settingsOpen) { sendTo(focused(), { type: 'get_config' }); sendTo(focused(), { type: 'installed_models' }) } }, [settingsOpen])
  // toasts: 5s auto-dismiss each, independently
  useEffect(() => {
    if (toasts.length === 0) return
    const timers = toasts.map((t) => window.setTimeout(() => dismissToast(t.id), 5000))
    return () => timers.forEach(clearTimeout)
  }, [toasts])

  const focused = () => (open.some((m) => m.id === focusModel) ? focusModel : (open[0]?.id ?? DEFAULT.id))
  const targets = () => (sync ? open.map((m) => m.id) : [focused()])
  // P9: native file picker for the SSH key path — desktop app only (dynamic import keeps the
  // plugin out of the browser bundle's critical path).
  async function browseSshKeyPath() {
    if (!inTauri()) return
    try {
      const { open } = await import('@tauri-apps/plugin-dialog')
      const picked = await open({
        multiple: false, directory: false,
        filters: [{ name: 'SSH key', extensions: ['pem', 'key'] }, { name: 'All files', extensions: ['*'] }],
      })
      if (typeof picked === 'string') setSshKeyPath(picked)
    } catch { /* not in tauri / plugin unavailable */ }
  }
  // T3 setup actions — all Tauri-gated via the tauriInvoke* wrappers (throw/no-op in the browser).
  const startInstall = () => {
    setInstallLog([]); setInstallExit(null); setInstalling(true)
    tauriInvokeResult('install_kernel_deps', { pythonPath: selectedPy })
      .catch((e) => { setInstalling(false); toast(`[setup] ${e instanceof Error ? e.message : String(e)}`) })
  }
  const useThisPython = () => {
    tauriInvokeResult('respawn_kernel', { pythonPath: selectedPy })
      .then(() => resumeReconnect())
      .catch((e) => toast(`[setup] ${e instanceof Error ? e.message : String(e)}`))
  }
  // native file picker for a custom Python (reuses browseSshKeyPath's dynamic-import pattern, no filter).
  async function browsePython() {
    if (!inTauri()) return
    try {
      const { open } = await import('@tauri-apps/plugin-dialog')
      const picked = await open({ multiple: false, directory: false })
      if (typeof picked !== 'string') return
      const info = await tauriInvokeValue<PyInfo | null>('probe_python', { path: picked })
      if (!info) { toast(`[setup] not a usable python: ${picked}`); return }
      setPythons((list) => [info, ...(list ?? []).filter((p) => p.path !== info.path)])
      setSelectedPy(info.path)
    } catch (e) { toast(`[setup] ${e instanceof Error ? e.message : String(e)}`) }
  }
  // fall back to the local kernel: drop the remote/tunnel keys and reload.
  const useLocalKernel = () => {
    localStorage.removeItem('ps_kernel_url')
    localStorage.removeItem('ps_kernel_token')
    sessionStorage.removeItem('ps_tunnel_live')
    window.location.reload()
  }
  // P7: SSH remote kernel connect/disconnect. Rust owns the tunnel + kernel lifecycle; we just
  // point WS_URL at the local tunnel port and reload once it reports success.
  async function sshConnect() {
    if (!sshHost.trim() || !sshUser.trim()) { toast('[ssh] host and username are required'); return }
    if (sshAuth === 'key' ? !sshKeyPath.trim() : !sshPassword) { toast(`[ssh] ${sshAuth === 'key' ? 'key path' : 'password'} is required`); return }
    setSshConnecting(true)
    try {
      await tauriInvokeResult('ssh_connect', {
        host: sshHost.trim(), port: Number(sshPort) || 22, username: sshUser.trim(),
        password: sshAuth === 'key' ? '' : sshPassword,
        keyPath: sshAuth === 'key' ? sshKeyPath.trim() : '',
        keyPassphrase: sshAuth === 'key' ? sshKeyPassphrase : '',
        repoDir: sshRepoDir.trim(), pythonPath: sshPythonPath.trim(), model: sshModel.trim(),
        hfHome: sshHfHome.trim(),
      })
      localStorage.setItem('ps_kernel_url', SSH_TUNNEL_WS)  // ssh_* fields persist via the effect above
      sessionStorage.setItem('ps_tunnel_live', '1')  // tunnel now up; cleared on app restart (see isTunnelLive)
      window.location.reload()
    } catch (err) {
      setSshConnecting(false)
      toast(`[ssh] ${err instanceof Error ? err.message : String(err)}`)
    }
  }
  async function sshDisconnect() {
    try { await tauriInvokeResult('ssh_disconnect') } catch (err) { toast(`[ssh] ${err instanceof Error ? err.message : String(err)}`) }
    localStorage.removeItem('ps_kernel_url')
    localStorage.removeItem('ps_kernel_token')
    sessionStorage.removeItem('ps_tunnel_live')
    window.location.reload()
  }
  function send() {
    if (!prompt.trim()) return
    for (const mid of targets()) {
      patch(mid, (d) => ({ ...empty(), spot: d.spot, knobs: d.knobs, kppl: d.kppl, regions: d.regions, evals: d.evals, train: d.train, tensors: d.tensors, busy: true }))  // keep workspace (spot/knobs/regions/train) across re-runs
      const spotRegion = spotActRegion[mid]
      promptImportancePending.current[mid] = prompt
      generatedText.current[mid] = ''
      sendTo(mid, { type: 'generate', prompt, max_tokens: maxTokens, temperature, probes: ['attention', 'activation', 'logitlens', 'spot_activation'].filter((p) => probesOn[p]), spot_topk: spotTopk, ...(spotRegion ? { spot_region: spotRegion } : {}) })
    }
  }
  function stop() {
    for (const mid of targets()) {
      delete abPhase.current[mid]  // stopping mid-A/B must not chain into the next generation
      delete promptImportancePending.current[mid]
      delete generatedText.current[mid]
      sockets.current[mid]?.send(JSON.stringify({ type: 'stop', model: mid }))
    }
  }
  function openModel(id: string, label: string, device?: string) {
    if (open.some((m) => m.id === id)) return
    const wasEmpty = open.length === 0
    setOpen((o) => [...o, { id, label }]); patch(id, () => ({ ...empty(), loading: true }))
    sendTo(id, { type: 'open', ...(device ? { device } : {}) })
    if (wasEmpty) setCols((cs) => cs.map((c) => ({ ...c, tiles: c.tiles.map((t) => ({ ...t, model: id })) })))  // adopt the first model
  }
  function closeModel(id: string) {
    const rest = open.find((m) => m.id !== id)?.id ?? null  // null → studio goes empty
    const s = sockets.current[id] as (WebSocket & { _intentional?: boolean }) | undefined
    if (s) { s._intentional = true; s.send(JSON.stringify({ type: 'close', model: id })); s.close() }
    delete sockets.current[id]
    setOpen((o) => o.filter((m) => m.id !== id))
    setData((d) => { const n = { ...d }; delete n[id]; return n })
    if (rest) setCols((cs) => cs.map((c) => ({ ...c, tiles: c.tiles.map((t) => (t.model === id ? { ...t, model: rest } : t)) })))
    setFocusModel((f) => (f === id ? (rest ?? '') : f))
  }
  // ---- 2-level layout ops (cols × tiles) ----
  function tileCount() { return cols.reduce((n, c) => n + c.tiles.length, 0) }
  function updateTile(tid: number, f: (t: Tile) => Tile) { setCols((cs) => cs.map((c) => ({ ...c, tiles: c.tiles.map((t) => (t.id === tid ? f(t) : t)) }))) }
  function setActive(tid: number, idx: number) { updateTile(tid, (t) => ({ ...t, active: idx })) }
  function setTileModel(tid: number, model: string) { updateTile(tid, (t) => ({ ...t, model })) }
  function addTab(tid: number, view: View) { updateTile(tid, (t) => t.tabs.includes(view) ? t : { ...t, tabs: [...t.tabs, view], active: t.tabs.length }) }  // choose which view; no dup tabs
  function newTile(model: string, view: string): Tile { return { id: nextId.current++, model, tabs: [view], active: 0, h: 1 } }
  function openTabId(tabId: string) {  // editor-style: open/focus a dynamic tab in the first tile
    setCols((cs) => cs.map((c, ci) => ci !== 0 ? c : { ...c, tiles: c.tiles.map((t, ti) => {
      if (ti !== 0) return t
      const idx = t.tabs.indexOf(tabId)
      return idx >= 0 ? { ...t, active: idx } : { ...t, tabs: [...t.tabs, tabId], active: t.tabs.length }
    }) }))
  }
  function openDataTab(name: string) {
    const dset = datasets.find((x) => x.name === name)
    if (dset?.server && dset.content == null) sendTo(focused(), { type: 'read_dataset', name })  // lazy fetch from disk
    openTabId(`data:${name}`)
  }
  function openRegionTab(name: string) {
    // default view = top 1% (the paper's headline size), not whatever % was last clicked globally.
    // capped by the region's saved base_topk — a region saved tighter than 1% can't be widened to 1%.
    const base = (data[focused()]?.regions ?? []).find((r) => r.name === name)?.base_topk
    const k = base != null ? Math.min(0.01, base) : 0.01
    setSpotTopk(k)  // keep the chip highlight in sync with what we requested
    sendTo(focused(), { type: 'region_info', name, topk: k })  // refresh the grid every open (cheap)
    openTabId(`region:${name}`)
  }
  function toggleCompare(name: string) {
    setCompareSel((sel) => {
      const next = sel.includes(name) ? sel.filter((x) => x !== name) : [...sel, name]  // no cap — hueAt() colours any count
      if (next.length >= 2) sendTo(focused(), { type: 'region_compare', names: next })
      else setCompareData(null)
      return next
    })
  }
  // ---- explorer delete actions (shared by inline ×, context menu, and Del key) — each keeps its confirm ----
  function deleteDataset(dset: { name: string; server?: boolean; link?: string | null }) {
    if (!dset.server) { setDatasets((dd) => dd.filter((x) => x.name !== dset.name)); return }  // session preset — reload restores
    confirmClick(`data:${dset.name}`, () => { sendTo(focused(), { type: 'delete_dataset', name: dset.link ?? dset.name }); if (selectedRef.current === `data:${dset.name}`) setSelected(null) })
  }
  function deleteRegion(name: string) {
    confirmClick(`region:${name}`, () => {
      sendTo(focused(), { type: 'delete_region', name })
      setCompareSel((sel) => sel.filter((x) => x !== name)); setCompareData(null)
      if (selectedRef.current === `region:${name}`) setSelected(null)
    })
  }
  // Del key: dispatch on the selected row. datasets/regions run their armed delete; models unload.
  const deleteSelectedRef = useRef<() => void>(() => {})
  deleteSelectedRef.current = () => {
    const s = selectedRef.current; if (!s) return
    const [kind, ...rest] = s.split(':'); const name = rest.join(':')
    if (kind === 'data') { const dset = datasets.find((x) => x.name === name); if (dset) deleteDataset(dset) }
    else if (kind === 'region') deleteRegion(name)
    else if (kind === 'model') { closeModel(name); setSelected(null) }
  }
  const deleteSelected = () => deleteSelectedRef.current()
  function splitRight(tid: number) {
    setCols((cs) => {
      const ci = cs.findIndex((c) => c.tiles.some((t) => t.id === tid)); const tile = cs[ci].tiles.find((t) => t.id === tid)!
      return [...cs.slice(0, ci + 1), { id: nextId.current++, w: 1, tiles: [newTile(tile.model, tile.tabs[tile.active])] }, ...cs.slice(ci + 1)]
    })
  }
  function splitDown(tid: number) {
    setCols((cs) => cs.map((c) => {
      const ti = c.tiles.findIndex((t) => t.id === tid); if (ti < 0) return c
      const tile = c.tiles[ti]; const nt = newTile(tile.model, tile.tabs[tile.active])
      return { ...c, tiles: [...c.tiles.slice(0, ti + 1), nt, ...c.tiles.slice(ti + 1)] }
    }))
  }
  function pruneClose(tid: number) {
    if (tileCount() <= 1) return
    setCols((cs) => cs.map((c) => ({ ...c, tiles: c.tiles.filter((t) => t.id !== tid) })).filter((c) => c.tiles.length > 0))
  }
  function closeTab(tid: number, idx: number) {
    let emptied = false
    setCols((cs) => cs.map((c) => ({
      ...c, tiles: c.tiles.map((t) => {
        if (t.id !== tid) return t
        const tabs = t.tabs.filter((_, i) => i !== idx)
        if (tabs.length === 0) emptied = true
        return { ...t, tabs, active: Math.max(0, idx <= t.active ? t.active - 1 : t.active) }
      }).filter((t) => t.tabs.length > 0),
    })).filter((c) => c.tiles.length > 0))
    void emptied
  }
  function moveTab(toTid: number, toIdx?: number) {
    const d = drag.current; drag.current = null; setDragging(false); setOverTile(null)
    if (!d || (d.tid === toTid && toIdx === undefined)) return
    setCols((cs) => {
      let view: string | undefined
      cs.forEach((c) => c.tiles.forEach((t) => { if (t.id === d.tid) view = t.tabs[d.idx] }))
      if (view == null) return cs
      let next = cs.map((c) => ({ ...c, tiles: c.tiles.map((t) => (t.id === d.tid ? { ...t, tabs: t.tabs.filter((_, i) => i !== d.idx), active: Math.max(0, d.idx <= t.active ? t.active - 1 : t.active) } : t)) }))
      next = next.map((c) => ({ ...c, tiles: c.tiles.map((t) => {
        if (t.id !== toTid || t.tabs.includes(view!)) return t
        const tabs = [...t.tabs]; const at = toIdx ?? tabs.length; tabs.splice(at, 0, view!); return { ...t, tabs, active: Math.min(at, tabs.length - 1) }
      }) }))
      next = next.map((c) => ({ ...c, tiles: c.tiles.filter((t) => t.tabs.length > 0) })).filter((c) => c.tiles.length > 0)
      return next.length ? next : cs
    })
  }
  function resizeCols(ci: number, e: React.MouseEvent) {
    e.preventDefault(); const x0 = e.clientX; const w = rowRef.current?.clientWidth ?? 1
    const total = cols.reduce((s, c) => s + c.w, 0); const a0 = cols[ci].w, b0 = cols[ci + 1].w
    const move = (ev: MouseEvent) => { const dd = ((ev.clientX - x0) / w) * total; setCols((cs) => cs.map((c, i) => i === ci ? { ...c, w: Math.max(0.15, a0 + dd) } : i === ci + 1 ? { ...c, w: Math.max(0.15, b0 - dd) } : c)) }
    const up = () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
    window.addEventListener('mousemove', move); window.addEventListener('mouseup', up)
  }
  function resizeTiles(ci: number, ti: number, e: React.MouseEvent) {
    e.preventDefault(); const y0 = e.clientY; const H = (e.currentTarget.parentElement as HTMLElement)?.clientHeight ?? 1
    const tiles = cols[ci].tiles; const total = tiles.reduce((s, t) => s + t.h, 0); const a0 = tiles[ti].h, b0 = tiles[ti + 1].h
    const move = (ev: MouseEvent) => { const dd = ((ev.clientY - y0) / H) * total; setCols((cs) => cs.map((c, i) => i !== ci ? c : { ...c, tiles: c.tiles.map((t, j) => j === ti ? { ...t, h: Math.max(0.12, a0 + dd) } : j === ti + 1 ? { ...t, h: Math.max(0.12, b0 - dd) } : t) })) }
    const up = () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
    window.addEventListener('mousemove', move); window.addEventListener('mouseup', up)
  }

  function viewBody(mid: string, view: string) {
    const d = data[mid] ?? empty()
    if (view.startsWith('data:')) {  // dataset editor tab — model-independent
      const name = view.slice(5)
      const dset = datasets.find((x) => x.name === name)
      if (!dset) return <span style={hint}>dataset removed</span>
      if (dset.content == null) return <span style={hint}>loading {name}…</span>
      const examples = dsMeta[name]?.examples ?? []
      const recs = parseRecords(dset.content)
      const keys = recs?.length && recs[0] && typeof recs[0] === 'object' ? Object.keys(recs[0] as object) : null
      const toggleField = (k: string) => setDatasets((dd) => dd.map((x) => {
        if (x.name !== name) return x
        const f = x.fields ?? []
        return { ...x, fields: f.includes(k) ? f.filter((y) => y !== k) : [...f, k] }
      }))
      return (<>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6 }}>
          <span style={{ color: 'var(--text-1)' }}><span className="mono">▤ {name}</span><span style={hint}> · {examples.length} examples</span></span>
          <Btn onClick={() => emitSpot(focused(), sample(examples))} title="compute spot on the focused model with the parsed examples (sampling applies)" color="var(--accent)" style={{ padding: '2px 10px' }}>Spot</Btn>
          <Btn onClick={() => setTrainDsName(name)} title="select this dataset as the training benchmark (Train tab)" style={{ padding: '2px 10px' }}>→ Train data</Btn>
          <Btn onClick={() => sendTo(focused(), { type: 'save_dataset', name, content: dset.content })} title="write to ~/.parametic_studio/datasets (persists across restarts)" style={{ padding: '2px 10px' }}>{dset.server ? 'Save' : 'Save to disk'}</Btn>
        </div>
        {keys && (
          <div style={{ display: 'flex', gap: 4, alignItems: 'center', flexWrap: 'wrap', marginBottom: 6 }}>
            <span style={{ ...hint, fontSize: 11 }}>fields:</span>
            {keys.map((k) => {
              const on = dset.fields?.includes(k)
              return <span key={k} onClick={() => toggleField(k)} style={{ fontSize: 11, padding: '1px 8px', borderRadius: 10, cursor: 'pointer', border: '1px solid var(--line-strong)', color: on ? 'var(--accent)' : 'var(--text-2)', background: on ? 'var(--bg-2)' : 'transparent' }}>{on ? '● ' : ''}{k}</span>
            })}
            <span style={{ ...hint, fontSize: 11 }}>{dset.fields?.length ? '(joined per record)' : '(none = auto: text-ish or longest field)'}</span>
          </div>
        )}
        {keys && <div title={examples[0]} className="mono" style={{ ...hint, fontSize: 11, marginBottom: 6, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>parsed[0] → {examples[0] ?? '—'}</div>}
        <div style={{ width: '100%', height: keys ? 'calc(100% - 96px)' : 'calc(100% - 34px)', minHeight: 120, border: '1px solid var(--line-strong)', borderRadius: 10, overflow: 'hidden', background: 'var(--bg-0)' }}>
          <CodeMirror value={dset.content} height="100%" theme="none"
            onChange={(v) => setDatasets((dd) => dd.map((x) => (x.name === name ? { ...x, content: v } : x)))}
            extensions={[
              cmTheme, cmHighlight, ...cmLangExt(name),
              keymap.of([{ key: 'Mod-s', run: () => { sendTo(focused(), { type: 'save_dataset', name, content: dset.content }); return true } }]),
            ]} />
        </div>
      </>)
    }
    if (view === 'compare') {  // cross-dataset spot comparison (saved regions, one hue each)
      const regs = data[focused()]?.regions ?? []
      const cd = compareData
      // colour by the region's fixed slot in the saved list, not its position in the selection —
      // so toggling one region off doesn't recolour the others (order-independent, a real toggle).
      const regHue = (name: string) => hueAt(Math.max(0, regs.findIndex((r) => r.name === name)))
      return (<>
        <div style={{ color: 'var(--text-1)', marginBottom: 4 }}>region compare<span style={hint}> · pick 2+ saved regions (e.g. per-language spots)</span></div>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 10 }}>
          {regs.map((r) => {
            const on = compareSel.includes(r.name)
            return <span key={r.name} onClick={() => toggleCompare(r.name)} style={{ fontSize: 11, padding: '1px 8px', borderRadius: 10, cursor: 'pointer', border: `1px solid ${on ? hueCss(regHue(r.name)) : 'var(--line-strong)'}`, color: on ? hueCss(regHue(r.name)) : 'var(--text-2)' }}>{on ? '● ' : ''}◈ {r.name}</span>
          })}
          {regs.length < 2 && <span style={{ ...hint, fontSize: 11 }}>save regions in the spot view first (one per dataset)</span>}
        </div>
        {compareSel.length >= 2 && !cd && <span style={hint}>comparing…</span>}
        {cd && (<>
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(cd.names.length, 2)}, 1fr)`, gap: 12, marginBottom: 10 }}>
            {cd.names.map((n) => (
              <div key={n}>
                <div style={{ fontSize: 11, color: hueCss(regHue(n)), marginBottom: 3 }}>◈ {n}<span style={hint}> · {cd.kinds[n] === 'importance' ? '|g×w| importance' : 'selection fraction — legacy region, re-save to get the importance map'}</span></div>
                <SpotGrid grid={cd.grids[n]} modules={cd.modules} color={hueRamp(regHue(n))} onHover={setCompareHover} hovered={compareHover} />
              </div>
            ))}
          </div>
          {(() => {
            // intersection panel, 3 selectable metrics over the weights ALL regions selected:
            //  · shared  — geo-mean of per-region importance (where all agree the important weights live)
            //  · lift    — observed ∩ / chance; explodes on tiny data-independent cells (biases/norms)
            //  · fraction— raw shared-weight density
            // 'shared' is the honest default: lift/fraction saturate on biases and bury the real signal.
            const allImp = cd.names.every((n) => cd.kinds[n] === 'importance')
            const metric = interMetric === 'shared' && !allImp ? 'fraction' : interMetric
            const grid = metric === 'shared'
              ? cd.grids[cd.names[0]].map((row, l) => row.map((_, c) =>
                  cd.intersection[l][c] > 0 ? Math.exp(cd.names.reduce((s, n) => s + Math.log(cd.grids[n][l][c] || 1e-30), 0) / cd.names.length) : 0))
              : metric === 'lift' ? (cd.intersectionLift ?? cd.intersection) : cd.intersection
            const flat = grid.flat(); const lo = Math.min(...flat), hi = Math.max(...flat)
            const label = {
              shared: <>color = <b>importance all {cd.names.length} agree on</b> (geo-mean |g×w|) — biases/norms don't saturate it</>,
              lift: <>color = <b>enrichment vs chance</b> (lift; independent top-k ⇒ 1×) — ⚠ tiny data-independent cells (biases) blow up</>,
              fraction: <>color = <b>shared-weight density</b> (∩ fraction) — ~uniform by construction</>,
            }
            const Seg = ({ id, txt }: { id: 'shared' | 'lift' | 'fraction'; txt: string }) => (
              <Btn onClick={() => setInterMetric(id)} title={`color the intersection by ${id}`}
                color={interMetric === id ? 'var(--accent)' : 'var(--text-2)'}
                style={{ borderColor: interMetric === id ? 'var(--accent)' : 'var(--line-strong)', padding: '1px 8px', fontSize: 11 }}>{txt}</Btn>
            )
            return (<>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3, flexWrap: 'wrap' }}>
                <span style={{ color: 'var(--text-1)' }}>intersection<span style={hint}> · weights selected by ALL {cd.names.length} regions</span></span>
                <div style={{ display: 'flex', gap: 4 }}><Seg id="shared" txt="shared imp" /><Seg id="lift" txt="lift" /><Seg id="fraction" txt="∩ fraction" /></div>
              </div>
              <div style={{ ...hint, marginBottom: 3 }}>{label[metric]}{interMetric === 'shared' && !allImp && ' · (legacy region → fell back to fraction; re-save for importance)'}</div>
              <SpotGrid grid={grid} modules={cd.modules}
                color={(v: number) => ampColor(hi > lo ? (v - lo) / (hi - lo) : 0, 1)}
                cellTitle={(l, mod) => { const c = cd.modules.indexOf(mod); const lift = cd.intersectionLift?.[l]?.[c]; const g = grid[l]?.[c] ?? 0
                  return `L${l} · ${mod}${metric === 'shared' ? ` · shared imp ${g.toExponential(1)}` : ''} · ∩ ${((cd.intersection[l]?.[c] ?? 0) * 100).toFixed(2)}% of weights${lift != null ? ` · ${lift.toFixed(0)}× vs chance` : ''}` }}
                onHover={setCompareHover} hovered={compareHover} />
            </>)
          })()}
          {cd.names.length >= 2 && (() => {
            const pair = Object.values(cd.jaccard)
            const lo = pair.length ? Math.min(...pair) : 0, hi = pair.length ? Math.max(...pair) : 0
            return (
              <div style={{ marginTop: 14 }}>
                <VizCard title="Spot overlap" accent="var(--accent)"
                  subtitle="Pairwise Jaccard between the selected spots — how much of the coding region they share. High off-diagonal = a common coding core across languages; low = language-specific weights.">
                  <JaccardMatrix names={cd.names} jaccard={cd.jaccard} />
                  <div style={{ ...hint, fontSize: 11, marginTop: 10 }}>shared across all pairs: <span style={{ color: 'var(--accent)' }}>{(lo * 100).toFixed(0)}–{(hi * 100).toFixed(0)}%</span> Jaccard{lo > 0.3 ? ' — strong common core' : lo < 0.1 ? ' — largely language-specific' : ''}</div>
                </VizCard>
              </div>
            )
          })()}
        </>)}
      </>)
    }
    if (view.startsWith('region:')) {  // saved-region viewer tab
      const name = view.slice(7)
      if (!(data[focused()]?.regions ?? []).some((r) => r.name === name)) return <span style={hint}>region deleted</span>
      const info = regionInfo[name]
      if (!info) return <span style={hint}>loading region {name}…</span>
      const raw = info.importance ?? info.grid
      // fixed reference (largest mass seen ≈ base top-k%) so a smaller % dims the map instead of re-pinning it to 100
      const normalized = info.importance ? scoreGridAgainst(info.importance, regionImpMax[name] ?? 0) : null
      const g = normalized?.grid ?? raw
      const viewTopk = info.view_topk ?? info.base_topk ?? spotTopk
      const topkPct = pctLabel(viewTopk)
      const shown = Math.min(Math.max(0, hoverLayer ?? layer), Math.max(0, g.length - 1))
      return (<>
        <VizCard title={name} accent="var(--syn-num, #e0a85e)"
          right={<span style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
            {SPOT_TOPK_OPTIONS.map((k) => {
              const disabled = info.base_topk != null && k > info.base_topk
              return <span key={k} onClick={() => {
                if (disabled) return
                setSpotTopk(k)
                sendTo(mid, { type: 'region_info', name, topk: k })
              }} className={`chip${spotTopk === k ? ' on' : ''}`} title={disabled ? `saved at ${pctLabel(info.base_topk!)}%, cannot expand to ${pctLabel(k)}%` : `show top ${pctLabel(k)}% parameter region`}
                style={{ opacity: disabled ? 0.45 : 1, cursor: disabled ? 'default' : 'pointer' }}>{pctLabel(k)}%</span>
            })}
            <span className="badge" title="weights selected at the current top-k% (changes as you switch %)">top {topkPct}% · {info.count.toLocaleString()} weights</span>
          </span>}
          subtitle={info.importance
            ? '|gradient × weight| importance captured when this spot was computed, normalized to 0–100 for comparison. Rows = layers (0↑), cols = modules — bright cells are the coding spot.'
            : 'Selection fraction per (layer, module) — legacy region without a saved importance map (re-save to get one). Per-param top-k% makes this ~uniform by construction.'}>
          <ScaleBar max={info.importance ? 100 : Math.max(...g.flat())} color={ampColor} label={info.importance ? 'score 0–100' : 'fraction'} />
          <div style={{ marginTop: 4 }}><SpotGrid grid={g} modules={info.modules}
            onCell={(l) => setLayer(l)}
            selected={new Set([`${shown}.${info.modules[0] ?? ''}`])}
            onHover={(cell) => setHoverLayer(cell ? Number(cell.split('.')[0]) : null)}
            hovered={hoverLayer != null ? `${hoverLayer}.${info.modules[0] ?? ''}` : null}
            cellTitle={(l, mod, v) => info.importance
              ? `L${l} · ${mod} · Parameter Importance Score=${v.toFixed(1)}/100 · raw |g×w|=${(raw[l]?.[info.modules.indexOf(mod)] ?? 0).toExponential(3)}`
              : `L${l} · ${mod} · fraction=${v.toPrecision(3)}`} /></div>
          {normalized && <div style={{ ...hint, fontSize: 11, marginTop: 8 }}>raw max |g×w| = {normalized.max.toExponential(3)}</div>}
          {info.base_topk != null && <div style={{ ...hint, fontSize: 11, marginTop: 8 }}>showing top {((info.view_topk ?? info.base_topk) * 100).toFixed(0)}% · saved at top {(info.base_topk * 100).toFixed(3)}% — adjustable down from there as a knob, never up · use it: spot view → named-region knob · train view → region select</div>}
        </VizCard>
        <SpotSummary grid={g} modules={info.modules} />
      </>)
    }
    if (d.loading) return <span style={hint}>loading {open.find((m) => m.id === mid)?.label ?? 'model'}…</span>
    if (view === 'output') return <div className="mono" style={{ whiteSpace: 'pre-wrap' }}>{d.output || <span style={hint}>run a prompt below</span>}{d.busy && <span style={{ color: 'var(--accent)' }}>▌</span>}</div>
    if (view === 'attention') {
      const frames = d.frames
      if (!frames.length) return <span style={hint}>attention shows here while generating (with the attention probe on)</span>
      const L = frames.at(-1)?.length ?? 0
      const layerSel = Math.min(Math.max(0, hoverLayer ?? layer), Math.max(0, L - 1))
      const N = frames.length
      const toks = [...d.promptTokens, ...d.tokenTexts]
      const aligned = toks.length === N  // prompt ⧺ generated lines up with the frame count (no history trim)
      const label = (i: number) => (aligned ? toks[i] : undefined) ?? `#${i}`
      const disp = (t: string) => (t === '' ? '␀' : t.replace(/\n/g, '⏎').replace(/\t/g, '⇥').replace(/ /g, '·'))
      const short = (t: string) => { const s = disp(t); return s.length > 11 ? s.slice(0, 11) + '…' : s }
      const M = frames.map((f) => f[layerSel] ?? [])  // M[q] = attention weights q→k (causal, length q+1)
      let max = 1e-9
      for (const r of M) for (const v of r) if (v > max) max = v
      const CELL = attnZoom
      const hc = attnCell
      const readout = hc ? `query ${JSON.stringify(label(hc.q))} → key ${JSON.stringify(label(hc.k))} = ${(M[hc.q]?.[hc.k] ?? 0).toFixed(3)}` : 'hover a cell — row = the token doing the attending, column = the token it looks at'
      return (<>
        <div style={{ color: 'var(--text-1)', marginBottom: 8 }}>Attention matrix<span style={hint}> · row = query token, column = key token, brighter = stronger · causal (upper-right empty) · {N}×{N}{aligned ? '' : ' · labels off (re-run)'}</span></div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8, fontSize: 12 }}>
          <span style={hint}>layer</span>
          <input type="range" min={0} max={Math.max(0, L - 1)} value={layerSel} onChange={(e) => setLayer(Number(e.target.value))} style={{ width: 200 }} />
          <span className="mono" style={{ color: 'var(--text-1)' }}>L{layerSel} / {L - 1}</span>
          <Btn onClick={() => sendTo(mid, { type: 'drilldown', layer: layerSel })} disabled={pending.has(`drilldown:${mid}`)} style={{ padding: '1px 8px' }}>{pending.has(`drilldown:${mid}`) ? '⟳ ' : ''}Heads</Btn>
          <span style={{ ...hint, marginLeft: 6 }}>zoom</span>
          <input type="range" min={7} max={40} value={attnZoom} onChange={(e) => setAttnZoom(Number(e.target.value))} title="cell size" style={{ width: 110 }} />
          <span className="mono" style={{ color: 'var(--text-1)' }}>{attnZoom}px</span>
          <ScaleBar max={max} color={cellColor} label="attn" />
        </div>
        {d.framesFlushed && N <= 1 && <div style={{ marginBottom: 8 }}><Btn onClick={() => sendTo(mid, { type: 'load_run', id: d.lastRunId })} color="var(--accent)">Load full history</Btn></div>}
        <div className="mono" style={{ fontSize: 11.5, color: hc ? 'var(--text-1)' : 'var(--text-2)', marginBottom: 6, minHeight: 16 }}>{readout}</div>
        <div style={{ overflow: 'auto', height: 480, minHeight: 200, minWidth: 240, resize: 'both', border: '1px solid var(--line)', borderRadius: 8, background: 'var(--bg-0)' }}
          onMouseLeave={() => setAttnCell(null)}>
          <div style={{ display: 'grid', gridTemplateColumns: `104px repeat(${N}, ${CELL}px)`, gap: 1, padding: 6, width: 'max-content' }}>
            {/* header row: corner + key-token labels (vertical) */}
            <div />
            {toks.slice(0, N).map((_, k) => (
              <div key={`h${k}`} style={{ height: 76, display: 'flex', alignItems: 'flex-end', justifyContent: 'center', overflow: 'hidden' }}>
                <span title={JSON.stringify(label(k))} style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', fontFamily: 'var(--mono, monospace)', fontSize: 10, color: attnCell?.k === k ? 'var(--accent)' : 'var(--text-2)', whiteSpace: 'nowrap' }}>{short(label(k))}</span>
              </div>
            ))}
            {/* body: each row = a query token */}
            {M.map((rowvals, q) => (
              <Fragment key={q}>
                <div title={JSON.stringify(label(q))} className="mono" style={{ height: CELL, display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: 6, overflow: 'hidden', fontSize: 10.5, whiteSpace: 'nowrap', color: attnCell?.q === q ? 'var(--accent)' : 'var(--text-1)' }}>{short(label(q))}</div>
                {toks.slice(0, N).map((_, k) => {
                  const v = k <= q ? (rowvals[k] ?? 0) : null
                  const on = attnCell?.q === q && attnCell?.k === k
                  return <div key={k} onMouseEnter={v == null ? undefined : () => setAttnCell({ q, k })}
                    title={v == null ? undefined : `q ${q} → k ${k} · ${v.toFixed(3)}`}
                    style={{ width: CELL, height: CELL, background: v == null ? 'transparent' : cellColor(v, max), outline: on ? '1.5px solid var(--accent)' : 'none', outlineOffset: -1 }} />
                })}
              </Fragment>
            ))}
          </div>
        </div>
        <div style={{ ...hint, fontSize: 11, marginTop: 8 }}>diagonal = self-attention · each row sums to 1 · ·=space ⏎=newline · <b>drag the bottom-right corner to resize</b>, or use the zoom slider to enlarge cells</div>
        {d.perhead && <div style={{ marginTop: 14, borderTop: '1px solid var(--line)', paddingTop: 10 }}><div style={{ color: 'var(--text-1)', marginBottom: 6, fontSize: 12 }}>layer {d.perhead.layer} · per-head (rows = heads, cols = kv tokens)</div><Grid rows={d.perhead.data} cols={d.perhead.data[0].length} rowH={13} cellTitle={(i, k, v) => `head ${i} · kv ${k} · ${v.toFixed(3)}`} /></div>}
      </>)
    }
    if (view === 'importance') {
      const importance = d.promptImportance
      const impModules = importance?.modules ?? []
      const impGrid = importance?.grid ?? null
      const impScore = impGrid ? normalizedScoreGrid(impGrid) : null
      const rowCount = impGrid?.length ?? 1
      const shown = Math.min(Math.max(0, hoverLayer ?? layer), Math.max(0, rowCount - 1))
      const openRegions = Array.from(new Set(cols.flatMap((c) => c.tiles.flatMap((t) => t.tabs))
        .filter((t) => t.startsWith('region:')).map((t) => t.slice(7))))
      const compareName = importanceCompareRegion && openRegions.includes(importanceCompareRegion) ? importanceCompareRegion : (openRegions[0] ?? '')
      const compareInfo = compareName ? regionInfo[compareName] : null
      const compareScore = compareInfo ? scoreGridAgainst(compareInfo.importance ?? compareInfo.grid, compareInfo.importance ? (regionImpMax[compareName] ?? 0) : 0).grid : null
      const diffGrid = impScore && compareInfo && compareScore ? moduleAlignedDiffGrid(impScore.grid, impModules, compareScore, compareInfo.modules) : null
      const diffVals = diffGrid?.[shown] ?? []
      const meanAbsDiff = diffVals.length ? diffVals.reduce((a, v) => a + Math.abs(v), 0) / diffVals.length : 0
      // whole-grid single-number summary of how different the two importance maps are (all layers × modules).
      // MAD = mean |Δ| (avg score-point gap per cell); RMSE = √mean(Δ²) (weights large local gaps more).
      const allDiff = diffGrid ? diffGrid.flat() : []
      const overallMAD = allDiff.length ? allDiff.reduce((a, v) => a + Math.abs(v), 0) / allDiff.length : 0
      const overallRMSE = allDiff.length ? Math.sqrt(allDiff.reduce((a, v) => a + v * v, 0) / allDiff.length) : 0
      // cosine similarity of the two module-aligned score maps — scale-invariant, so it answers
      // "are the important parameters in the same PLACES?" regardless of each map's normalization base.
      let cosineSim = 0
      if (impScore && compareInfo && compareScore) {
        const bIdx = new Map(compareInfo.modules.map((m, i) => [m, i]))
        let dot = 0, na = 0, nb = 0
        impScore.grid.forEach((row, l) => row.forEach((av, c) => {
          const bv = compareScore[l]?.[bIdx.get(impModules[c]) ?? -1] ?? 0
          dot += av * bv; na += av * av; nb += bv * bv
        }))
        cosineSim = na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0
      }
      const sub = 'Request + response Parameter Importance Score: |gradient × weight| for the generated exchange, reduced to layer × parameter module, then normalized to 0–100 for comparison. Hover a layer to inspect its module scores; click to pin.'
      return (
        <VizCard title="Parameter Importance" accent="var(--syn-num, #e0a85e)" subtitle={sub}
          right={<span style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
            {SPOT_TOPK_OPTIONS.map((k) => (
              <span key={k} onClick={() => {
                setSpotTopk(k)
                setNamedRegionTopk(k)
                if (compareName) sendTo(mid, { type: 'region_info', name: compareName, topk: k })
              }} className={`chip${spotTopk === k ? ' on' : ''}`} title="top-k parameter region used for saved-region comparison">{pctLabel(k)}%</span>
            ))}
            {openRegions.length > 0 && <select value={compareName} onChange={(e) => {
              const next = e.target.value
              setImportanceCompareRegion(next)
              if (next) sendTo(mid, { type: 'region_info', name: next, topk: spotTopk })
            }} title="compare current prompt importance against an opened saved parameter region"
              style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '2px 6px', maxWidth: 180 }}>
              {openRegions.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>}
          </span>}>
          {d.promptImportanceProg && <div style={{ marginBottom: 8 }}><div style={{ display: 'flex', gap: 8, alignItems: 'center' }}><span style={hint}>computing request+response importance · {d.promptImportanceProg.i}/{d.promptImportanceProg.total}</span></div><div style={{ background: 'var(--bg-2)', borderRadius: 6, height: 4, marginTop: 3 }}><div style={{ height: 4, width: `${Math.round((d.promptImportanceProg.i / d.promptImportanceProg.total) * 100)}%`, background: 'var(--accent)', borderRadius: 6 }} /></div></div>}
          {!impGrid ? <span style={hint}>send a prompt and wait for generation to finish; this view will show only `|gradient × weight|` Parameter Importance Score, not activations.</span>
            : <>
              <ScaleBar max={100} color={ampColor} label="score 0–100" />
              <div style={{ marginTop: 4 }}><SpotGrid grid={impScore!.grid} modules={impModules} color={ampColor}
                onCell={(l) => setLayer(l)} selected={new Set([`${shown}.${impModules[0] ?? ''}`])}
                onHover={(cell) => setHoverLayer(cell ? Number(cell.split('.')[0]) : null)}
                hovered={hoverLayer != null ? `${hoverLayer}.${impModules[0] ?? ''}` : null}
                cellTitle={(l, mod, v) => `L${l} · ${mod} · Parameter Importance Score=${v.toFixed(1)}/100 · raw |g×w|=${(impGrid[l]?.[impModules.indexOf(mod)] ?? 0).toExponential(3)} · click layer for detail`} /></div>
              <div style={{ color: 'var(--text-1)', margin: '14px 0 8px' }}>layer {shown}{hoverLayer != null && hoverLayer !== layer ? <span style={hint}> · preview, pinned L{layer}</span> : <span style={hint}> · pinned</span>}<span style={hint}> · request+response Parameter Importance Score 0–100 · raw max {impScore!.max.toExponential(3)} · {impModules.length} parameter modules</span></div>
              <BarList items={impModules.map((m, c) => ({ label: moduleFullLabel(m), value: impScore!.grid[shown]?.[c] ?? 0, title: `${m} · raw |g×w|=${(impGrid[shown]?.[c] ?? 0).toExponential(3)}` }))} color={ampColor} valueLabel={(v) => v.toFixed(1)} />
              {openRegions.length === 0 ? <div style={{ ...hint, fontSize: 11, marginTop: 10 }}>Open a saved parameter region tab next to this view to show score differences for the hovered layer.</div>
                : !compareInfo || !diffGrid ? <div style={{ ...hint, fontSize: 11, marginTop: 10 }}>loading comparison region {compareName}…</div>
                  : <div style={{ marginTop: 14, borderTop: '1px solid var(--line)', paddingTop: 10 }}>
                    <div style={{ color: 'var(--text-1)', marginBottom: 8 }}>Δ vs <span className="mono">{compareName}</span><span style={hint}> · current prompt − saved region · layer {shown} · this layer mean |Δ| {meanAbsDiff.toFixed(1)}</span></div>
                    <div style={{ display: 'grid', gap: 6 }}>
                      {impModules.map((m, c) => ({ label: moduleFullLabel(m), value: diffGrid[shown]?.[c] ?? 0, title: m })).map((it) => {
                          const mag = Math.min(100, Math.abs(it.value))
                          return (
                            <div key={it.title} title={`${it.title} · Δ=${it.value >= 0 ? '+' : ''}${it.value.toFixed(1)} (${compareName})`} style={{ display: 'grid', gridTemplateColumns: '108px 1fr 48px', gap: 8, alignItems: 'center', fontSize: 11 }}>
                              <span className="mono" style={{ color: 'var(--text-1)', overflow: 'hidden', textOverflow: 'ellipsis' }}>{it.label}</span>
                              <div style={{ background: 'var(--bg-2)', borderRadius: 5, height: 11, overflow: 'hidden' }}>
                                <div style={{ height: 11, width: `${mag}%`, background: it.value >= 0 ? 'var(--accent)' : 'var(--danger)', borderRadius: 5, transition: 'width var(--ease)' }} />
                              </div>
                              <span style={{ color: it.value >= 0 ? 'var(--accent)' : 'var(--danger)', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{it.value >= 0 ? '+' : ''}{it.value.toFixed(1)}</span>
                            </div>
                          )
                        })}
                    </div>
                    {/* bottom-of-tab summary: the whole prompt↔region divergence as single numbers */}
                    <div style={{ marginTop: 16, borderTop: '2px solid var(--line-strong)', paddingTop: 12, display: 'flex', gap: 22, alignItems: 'baseline', flexWrap: 'wrap' }}>
                      <span style={{ color: 'var(--text-1)', fontSize: 12, fontWeight: 600 }}>Overall difference<span style={hint}> · all {(diffGrid.length)}×{impModules.length} cells vs <span className="mono">{compareName}</span></span></span>
                      <span style={{ display: 'flex', alignItems: 'baseline', gap: 6 }} title="mean |Δ| over ALL layers × modules — average score-point gap per cell (0–100 scale)">
                        <span style={hint}>mean |Δ|</span><span style={{ color: 'var(--accent)', fontSize: 20, fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>{overallMAD.toFixed(1)}</span></span>
                      <span style={{ display: 'flex', alignItems: 'baseline', gap: 6 }} title="√mean(Δ²) — like mean |Δ| but weights large local gaps more heavily">
                        <span style={hint}>RMSE</span><span style={{ color: 'var(--text-1)', fontSize: 20, fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>{overallRMSE.toFixed(1)}</span></span>
                      <span style={{ display: 'flex', alignItems: 'baseline', gap: 6 }} title="cosine similarity of the two score maps — scale-invariant pattern match (1.000 = identical layout of important modules, regardless of magnitude)">
                        <span style={hint}>cosine sim</span><span style={{ color: 'var(--text-1)', fontSize: 20, fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>{cosineSim.toFixed(3)}</span></span>
                    </div>
                  </div>}
            </>}
        </VizCard>
      )
    }
    if (view === 'activations') {
      const mods = ['self_attn', 'mlp']
      const frames = d.actDetailFrames.length ? d.actDetailFrames : d.actFrames
      const modules = d.actDetail?.modules ?? mods
      const rowCount = frames[0]?.length ?? 1
      const shown = Math.min(Math.max(0, hoverLayer ?? layer), Math.max(0, rowCount - 1))
      const sub = 'Forward activation magnitude accumulated over all tokens in the latest generation state. This is not Parameter Importance; it shows which modules produced large output activations for the request/response context.'
      return (
        <VizCard title="Activations" accent="var(--accent)" subtitle={sub}>
          {!frames.length ? <span style={hint}>generate a prompt with the activation probe on to see forward activations</span>
            : (() => { const g = frames.at(-1)!; return (<>
              <ScaleBar max={Math.max(...g.flat())} color={cellColor} label="Σ token ‖out‖" />
              <div style={{ marginTop: 4 }}><SpotGrid grid={g} modules={modules} color={cellColor}
                onCell={(l) => setLayer(l)} selected={new Set([`${shown}.${modules[0] ?? ''}`])}
                onHover={(cell) => setHoverLayer(cell ? Number(cell.split('.')[0]) : null)}
                hovered={hoverLayer != null ? `${hoverLayer}.${modules[0] ?? ''}` : null}
                cellTitle={(l, mod, v) => `L${l} · ${mod} · Σ token activation ‖out‖=${v.toFixed(2)}`} /></div>
              <div style={{ color: 'var(--text-1)', margin: '14px 0 8px' }}>layer {shown}{hoverLayer != null && hoverLayer !== layer ? <span style={hint}> · preview, pinned L{layer}</span> : <span style={hint}> · pinned</span>}<span style={hint}> · cumulative token activation · {modules.length} modules</span></div>
              <BarList items={modules.map((m, c) => ({ label: moduleFullLabel(m), value: g[shown]?.[c] ?? 0, title: m }))} color={cellColor} />
              <div style={{ ...hint, fontSize: 11, marginTop: 10 }}>Use this to compare which modules light up across prompts. For attribution or causal importance, use the Parameter Importance tab.</div>
            </>) })()}
        </VizCard>
      )
    }
    if (view === 'activations-spot') {
      const regionPick = spotActRegion[mid] ?? ''
      const regionSel = (
        <select value={regionPick} onChange={(e) => setSpotActRegion((s) => ({ ...s, [mid]: e.target.value }))}
          title="which identified spot to show activation for — a saved region needs no recompute"
          style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 'var(--radius-ctrl)', padding: '3px 8px' }}>
          <option value="">current spot</option>
          {d.regions.map((r) => <option key={r.name} value={r.name}>{r.name} ({r.count.toLocaleString()})</option>)}
        </select>
      )
      const ready = regionPick ? true : !!d.spot  // a saved region needs no computed spot this session
      const frames = d.spotActFrames
      const modules = d.spotAct?.modules ?? []
      const sub = activMode === 'cumulative'
        ? 'Mean |output value| at the spot\'s neurons over ALL generated tokens — 0 where a module holds no spot weights. The steady spot-firing profile for this run.'
        : 'Per-token: rows = layers, cols = generated tokens, brightness = the spot\'s total activation at each token. See exactly which tokens light the coding region.'
      const body = !ready
        ? <span style={hint}>compute a spot first (Spot tab), or pick a saved region →</span>
        : !frames.length
          ? <span style={hint}>generate a prompt to see live activation over the spot</span>
          : activMode === 'cumulative' ? (() => { const g = meanFrames(frames); return (<>
              <ScaleBar max={Math.max(...g.flat())} color={cellColor} label="spot activation" />
              <div style={{ marginTop: 4 }}><SpotGrid grid={g} modules={modules} color={cellColor} cellTitle={(l, mod, v) => `L${l} · ${mod} · mean spot act=${v.toFixed(3)}`} /></div>
            </>) })()
          : (<>
              <ActivationTimeline matrix={layerTimeMatrix(frames)} tokens={d.tokenTexts} label="spot act/token" />
              <div style={{ ...hint, fontSize: 10, marginTop: 4 }}>{frames.length} tokens · rows = layers · scroll → for long runs</div>
            </>)
      return (
        <VizCard title="Spot activation" accent="var(--accent)" right={<span style={{ display: 'flex', gap: 8, alignItems: 'center' }}><ModeToggle mode={activMode} set={setActivMode} />{regionSel}</span>} subtitle={sub}>
          {body}
        </VizCard>
      )
    }
    if (view === 'usage') {
      const regionPick = usageRegion[mid] ?? ''
      const promptText = usagePrompts[mid] ?? USAGE_DEFAULT
      const ready = regionPick ? true : !!d.spot
      const topk = d.knobs.find((k) => k.kind === 'spot')?.topk ?? spotTopk
      const regionMsg = regionPick ? { region_name: regionPick } : { topk }
      const runCompare = () => {
        const prompts = promptText.split('\n').map((p) => p.trim()).filter(Boolean)
        if (!prompts.length) return
        patch(mid, (dd) => ({ ...dd, usage: { ...dd.usage, rows: [], running: true, detail: null } }))
        sendTo(mid, { type: 'region_usage_batch', prompts, ...regionMsg })
      }
      const openDetail = (prompt: string) => sendTo(mid, { type: 'region_usage', prompt, ...regionMsg })
      const regionSel = (
        <select value={regionPick} onChange={(e) => setUsageRegion((s) => ({ ...s, [mid]: e.target.value }))} title="which spot to measure usage against"
          style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 'var(--radius-ctrl)', padding: '3px 8px' }}>
          <option value="">current spot</option>
          {d.regions.map((r) => <option key={r.name} value={r.name}>{r.name}</option>)}
        </select>
      )
      const rows = [...d.usage.rows].sort((a, b) => b.overall - a.overall)
      const detail = d.usage.detail
      return (
        <VizCard title="Region usage" accent="var(--accent)" right={regionSel}
          subtitle="Activation share (Definition A): of all activation flowing through the spot's modules while processing a prompt, how much runs through the spot itself. High = the prompt heavily engages the coding region. Click a prompt for its per-token timeline.">
          {!ready ? <span style={hint}>compute a spot first (Spot tab), or pick a saved region →</span> : (<>
            <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', marginBottom: 12 }}>
              <textarea value={promptText} onChange={(e) => setUsagePrompts((s) => ({ ...s, [mid]: e.target.value }))} rows={4} spellCheck={false}
                placeholder="one prompt per line" style={{ flex: 1, background: 'var(--bg-2)', border: '1px solid var(--line-strong)', borderRadius: 'var(--radius-ctrl)', padding: '6px 8px', outline: 'none', resize: 'vertical', fontSize: 12 }} />
              <Btn onClick={runCompare} disabled={pending.has(`region_usage_batch:${mid}`)} color="var(--accent)" style={{ padding: '5px 14px' }}>{d.usage.running ? '⟳ …' : 'Compare'}</Btn>
            </div>
            {rows.length > 0 && (
              <div style={{ display: 'grid', gap: 6, marginBottom: detail ? 16 : 0 }}>
                <div style={{ ...hint, fontSize: 11, display: 'grid', gridTemplateColumns: '1fr 120px 44px', gap: 8 }}><span>prompt</span><span>region usage</span><span style={{ textAlign: 'right' }}></span></div>
                {rows.map((r) => (
                  <div key={r.prompt} onClick={() => openDetail(r.prompt)} className="hover-row" title="click → per-token timeline"
                    style={{ display: 'grid', gridTemplateColumns: '1fr 120px 44px', gap: 8, alignItems: 'center', fontSize: 12, cursor: 'pointer', padding: '3px 4px', borderRadius: 6, background: detail?.prompt === r.prompt ? 'var(--accent-soft)' : 'transparent' }}>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-1)' }}>{r.prompt}</span>
                    <div style={{ background: 'var(--bg-2)', borderRadius: 5, height: 12, overflow: 'hidden' }}><div style={{ height: 12, width: `${Math.round(r.overall * 100)}%`, background: cellColor(r.overall, 1), borderRadius: 5 }} /></div>
                    <span className="mono" style={{ textAlign: 'right', color: 'var(--text-0)', fontVariantNumeric: 'tabular-nums' }}>{Math.round(r.overall * 100)}%</span>
                  </div>
                ))}
              </div>
            )}
            {detail && (
              <div style={{ borderTop: '1px solid var(--line)', paddingTop: 12 }}>
                <div style={{ color: 'var(--text-1)', marginBottom: 8, fontSize: 12 }}>Token timeline<span style={hint}> · {detail.prompt} · overall {Math.round(detail.overall * 100)}%</span></div>
                <div style={{ display: 'grid', gap: 4 }}>
                  {detail.tokens.map((t, i) => (
                    <div key={i} style={{ display: 'grid', gridTemplateColumns: '110px 1fr 40px', gap: 8, alignItems: 'center', fontSize: 11 }}>
                      <span className="mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-1)' }}>{t.text.replace(/\n/g, '⏎').replace(/ /g, '·') || '∅'}</span>
                      <div style={{ background: 'var(--bg-2)', borderRadius: 4, height: 10, overflow: 'hidden' }}><div style={{ height: 10, width: `${Math.round(t.usage * 100)}%`, background: cellColor(t.usage, 1), borderRadius: 4 }} /></div>
                      <span className="mono" style={{ textAlign: 'right', color: 'var(--text-2)', fontVariantNumeric: 'tabular-nums' }}>{Math.round(t.usage * 100)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>)}
        </VizCard>
      )
    }
    if (view === 'logitlens') return d.logit ? (<><div style={{ color: 'var(--text-1)', marginBottom: 8 }}>layer → top-1 · {d.logit.length} layers</div><div style={{ display: 'grid', gap: 3 }}>{d.logit.map((r, l) => <div key={l} style={{ display: 'grid', gridTemplateColumns: '34px 76px 1fr', gap: 8, alignItems: 'center' }}><span className="mono" style={hint}>L{l}</span><span className="mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.token}</span><div style={{ background: 'var(--bg-2)', borderRadius: 6, height: 9 }}><div style={{ height: 9, width: `${Math.round(r.prob * 100)}%`, background: 'var(--accent)', borderRadius: 6 }} /></div></div>)}</div></>) : <span style={hint}>logit lens while generating</span>
    if (view === 'tensors') {
      const ts = d.tensors
      if (!ts) {
        if (!tvFetched.current.has(mid)) { tvFetched.current.add(mid); sendTo(mid, { type: 'tensors' }) }
        return <span style={hint}>loading tensors…</span>
      }
      const layerOf = (name: string) => { const mm = name.match(/^model\.layers\.(\d+)\./); return mm ? mm[1] : 'other' }
      const keys = Array.from(new Set(ts.map((t) => layerOf(t.name))))
      const layers = keys.filter((k) => k !== 'other').sort((a, b) => Number(a) - Number(b))
      if (keys.includes('other')) layers.push('other')
      const sel = tvSel[mid] ?? { layer: layers[0] ?? 'other', tensor: null }
      const inLayer = ts.filter((t) => layerOf(t.name) === sel.layer)
      const shortName = (name: string) => sel.layer === 'other' ? name : name.replace(`model.layers.${sel.layer}.`, '')
      const src = tvSource[mid] ?? 'weights'
      const isRegionSrc = src !== 'weights' && src !== 'importance'
      const srcMsg = (name: string, r0: number, c0: number) => ({ type: 'tensor_values' as const, name, r0, c0, rows: 48, cols: 48, source: isRegionSrc ? 'region' : src, ...(isRegionSrc ? { region: src } : {}) })
      const pickLayer = (l: string) => setTvSel((s) => ({ ...s, [mid]: { layer: l, tensor: null } }))
      const load = (name: string, r0 = 0, c0 = 0) => { setTvHover(null); setTvSel((s) => ({ ...s, [mid]: { layer: sel.layer, tensor: name } })); setPendingKey(`tensor_values:${mid}`, true); sendTo(mid, srcMsg(name, r0, c0)) }
      const pickSource = (next: string) => { setTvSource((s) => ({ ...s, [mid]: next })); setTvHover(null); if (sel.tensor) { setPendingKey(`tensor_values:${mid}`, true); const isReg = next !== 'weights' && next !== 'importance'; sendTo(mid, { type: 'tensor_values', name: sel.tensor, r0: 0, c0: 0, rows: 48, cols: 48, source: isReg ? 'region' : next, ...(isReg ? { region: next } : {}) }) } }
      const tv = sel.tensor && tvData[mid]?.name === sel.tensor && tvData[mid]?.source === (isRegionSrc ? 'region' : src) ? tvData[mid] : null
      const loading = pending.has(`tensor_values:${mid}`)
      const absmax = tv?.stats.absmax || 1
      const posMax = tv?.stats.max || 1
      // weights → diverging blue/red (signed); importance/region → amber sequential (≥0)
      const cellBg = (v: number) => tv?.signed
        ? (v >= 0 ? `rgba(45,127,249,${Math.min(1, v / absmax)})` : `rgba(248,43,96,${Math.min(1, Math.abs(v) / absmax)})`)
        : ampColor(v, posMax)
      const cols = tv?.values[0]?.length ?? 0
      const srcLabel = src === 'weights' ? 'model weights θ' : src === 'importance' ? 'live |grad×weight| importance' : `region “${src}” importance`
      return (
        <VizCard title="Tensors" accent="var(--syn-num, #e0a85e)"
          subtitle="Pick a value source, a layer, then a tensor. weights = the model's parameters θ (signed). importance = the live |g×w| from the last spot/prompt. region = a saved spot's importance (0 outside the spot). Big tensors show a 48×48 window; stats are over the full tensor."
          right={<select value={src} onChange={(e) => pickSource(e.target.value)} title="which values to show at each parameter position"
            style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '2px 6px', maxWidth: 220 }}>
            <option value="weights">Original model · weights θ</option>
            <option value="importance">Live Parameter Importance</option>
            {(d.regions ?? []).length > 0 && <optgroup label="Saved regions">
              {(d.regions ?? []).map((r) => <option key={r.name} value={r.name}>{r.name}</option>)}
            </optgroup>}
          </select>}>
          <div style={{ ...hint, marginBottom: 8 }}>Showing: <span style={{ color: 'var(--text-1)' }}>{srcLabel}</span></div>
          <div style={{ ...hint, marginBottom: 4 }}>Layer</div>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 12 }}>
            {layers.map((l) => <span key={l} onClick={() => pickLayer(l)} className={`chip${sel.layer === l ? ' on' : ''}`} style={{ cursor: 'pointer' }}>{l === 'other' ? 'other' : `L${l}`}</span>)}
          </div>
          <div style={{ ...hint, marginBottom: 4 }}>Tensors in {sel.layer === 'other' ? 'other' : `layer ${sel.layer}`} · {inLayer.length}</div>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 14 }}>
            {inLayer.map((t) => (
              <span key={t.name} onClick={() => load(t.name)} className={`chip${sel.tensor === t.name ? ' on' : ''}`} title={`${t.name} · [${t.shape.join('×')}] ${t.dtype}`} style={{ cursor: 'pointer', display: 'inline-flex', gap: 6, alignItems: 'center' }}>
                <Icon name="tensor" size={11} /><span className="mono">{shortName(t.name)}</span><span style={hint}>[{t.shape.join('×')}]</span>
              </span>
            ))}
          </div>
          {!tv ? <span style={hint}>{loading ? 'loading values…' : 'select a tensor above to see its values'}</span>
            : <>
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'baseline', marginBottom: 8, fontSize: 11 }}>
                <span className="mono" style={{ color: 'var(--text-1)', fontWeight: 600 }}>{tv.name}</span>
                <span style={hint}>shape [{tv.shape.join('×')}] · {tv.dtype}{tv.flattened ? ' · flattened to 2D' : ''}</span>
                <span style={{ fontVariantNumeric: 'tabular-nums' }}><span style={hint}>min </span>{tv.stats.min.toExponential(2)} <span style={hint}>max </span>{tv.stats.max.toExponential(2)} <span style={hint}>mean </span>{tv.stats.mean.toExponential(2)} <span style={hint}>std </span>{tv.stats.std.toExponential(2)}</span>
              </div>
              {/* paging: windows into a tensor larger than 48×48 */}
              {(tv.rows_total > tv.values.length || tv.cols_total > cols) && (
                <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8, fontSize: 11 }}>
                  <span style={hint}>rows {tv.r0}–{tv.r0 + tv.values.length} / {tv.rows_total} · cols {tv.c0}–{tv.c0 + cols} / {tv.cols_total}</span>
                  <span style={{ display: 'flex', gap: 3 }}>
                    <Btn onClick={() => load(tv.name, Math.max(0, tv.r0 - 48), tv.c0)} disabled={tv.r0 <= 0} style={{ padding: '1px 7px' }}>↑</Btn>
                    <Btn onClick={() => load(tv.name, tv.r0 + 48, tv.c0)} disabled={tv.r0 + tv.values.length >= tv.rows_total} style={{ padding: '1px 7px' }}>↓</Btn>
                    <Btn onClick={() => load(tv.name, tv.r0, Math.max(0, tv.c0 - 48))} disabled={tv.c0 <= 0} style={{ padding: '1px 7px' }}>←</Btn>
                    <Btn onClick={() => load(tv.name, tv.r0, tv.c0 + 48)} disabled={tv.c0 + cols >= tv.cols_total} style={{ padding: '1px 7px' }}>→</Btn>
                  </span>
                </div>
              )}
              <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 6, fontSize: 11 }}>
                {tv.signed ? <>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}><span style={{ width: 12, height: 12, background: 'rgba(248,43,96,0.85)', borderRadius: 2 }} /><span style={hint}>negative</span></span>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}><span style={{ width: 12, height: 12, background: 'rgba(45,127,249,0.85)', borderRadius: 2 }} /><span style={hint}>positive</span></span>
                  <span style={hint}>intensity = |value| / {absmax.toExponential(2)} (abs max)</span>
                </> : <>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}><span style={{ width: 12, height: 12, background: ampColor(0, 1), borderRadius: 2 }} />→<span style={{ width: 12, height: 12, background: ampColor(1, 1), borderRadius: 2 }} /><span style={hint}>low → high</span></span>
                  <span style={hint}>{src === 'importance' ? '|grad×weight| importance' : 'region importance'} · dark = 0 (outside the spot / unimportant) · max {posMax.toExponential(2)}</span>
                </>}
              </div>
              {/* hover readout: value appears while over the map, clears when the mouse leaves. min-height keeps layout stable */}
              <div style={{ minHeight: 20, marginBottom: 6, fontSize: 12, fontVariantNumeric: 'tabular-nums' }}>
                {tvHover
                  ? <span><span style={hint}>[row {tv.r0 + tvHover.i}, col {tv.c0 + tvHover.j}] = </span><span style={{ color: tvHover.v >= 0 ? 'var(--accent)' : 'var(--danger)', fontWeight: 700 }}>{tvHover.v.toExponential(6)}</span></span>
                  : <span style={hint}>hover a cell to read its value</span>}
              </div>
              <div onMouseLeave={() => setTvHover(null)} style={{ overflow: 'auto', maxWidth: '100%', border: '1px solid var(--line)', borderRadius: 6, padding: 4 }}>
                <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 13px)`, gap: 1 }}>
                  {tv.values.flatMap((row, i) => row.map((v, j) => (
                    <div key={`${i}.${j}`} onMouseEnter={() => setTvHover({ i, j, v })}
                      style={{ width: 13, height: 13, background: cellBg(v), borderRadius: 2, outline: tvHover?.i === i && tvHover?.j === j ? '1px solid var(--text-1)' : 'none' }} />
                  )))}
                </div>
              </div>
            </>}
        </VizCard>
      )
    }
    if (view === 'control') {
      const regions = d.regions ?? []
      const sel = ctrlSel[mid] ?? ''
      const selReg = regions.find((r) => r.name === sel)
      const base = selReg?.base_topk
      const topk = ctrlTopk[mid] ?? (base != null ? Math.min(0.01, base) : 0.01)
      const applied = ctrlApplied[mid]
      const cp = ctrlPpl[mid] ?? {}
      const busy = pending.has(`ppl:${mid}`) || pending.has(`intervene:${mid}`)
      const fmt = (v: number | undefined) => v == null ? '—' : v >= 1e4 ? v.toExponential(1) : v.toFixed(1)
      const shortRegion = (name: string) => name.length > 28 ? `${name.slice(0, 25)}…` : name
      const pick = (name: string) => { setCtrlSel((s) => ({ ...s, [mid]: name })); setCtrlApplied((a) => ({ ...a, [mid]: null })); setCtrlPpl((c) => ({ ...c, [mid]: {} })) }
      const apply = () => { if (!sel) return; sendTo(mid, { type: 'intervene', region: { kind: 'named', name: sel, topk }, op: ctrlOp, alpha: ctrlAlpha, key: 'control' }); setCtrlApplied((a) => ({ ...a, [mid]: { name: sel, op: ctrlOp, alpha: ctrlAlpha, topk } })) }
      const clear = () => { sendTo(mid, { type: 'clear', key: 'control' }); setCtrlApplied((a) => ({ ...a, [mid]: null })) }
      // one-click A/B: measure clean → apply the adjustment → measure adjusted (kernel processes ws msgs in order)
      const compare = () => {
        if (!sel) return
        const code = dsExamples; const gen = GENERAL_SET.split('\n').filter(Boolean)
        setCtrlPpl((c) => ({ ...c, [mid]: {} }))
        sendTo(mid, { type: 'clear', key: 'control' })
        sendTo(mid, { type: 'ppl', examples: code, tag: 'ctrl:cleanCode' })
        sendTo(mid, { type: 'ppl', examples: gen, tag: 'ctrl:cleanGen' })
        sendTo(mid, { type: 'intervene', region: { kind: 'named', name: sel, topk }, op: ctrlOp, alpha: ctrlAlpha, key: 'control' })
        sendTo(mid, { type: 'ppl', examples: code, tag: 'ctrl:adjCode' })
        sendTo(mid, { type: 'ppl', examples: gen, tag: 'ctrl:adjGen' })
        setCtrlApplied((a) => ({ ...a, [mid]: { name: sel, op: ctrlOp, alpha: ctrlAlpha, topk } }))
      }
      // benchmark: HumanEval pass@1 on the CURRENT weights (clean if nothing applied, else the deleted state)
      const ce = ctrlEval[mid] ?? {}
      const evalBusy = pending.has(`eval_code:${mid}`)
      const evalMsg = (slot: 'clean' | 'deleted') => {
        ctrlEvalSlot.current[mid] = slot
        const extra = (ctrlEvalGpus[mid] ?? []).map((i) => `cuda:${i}`)
        setPendingKey(`eval_code:${mid}`, true)
        sendTo(mid, { type: 'eval_code', dataset: ctrlEvalDs, temperature: 0, max_tokens: 512, limit: ctrlEvalLimit, ...(extra.length ? { extra_devices: extra } : {}) })
      }
      const runBench = () => {
        if (!ctrlEvalDs) { toast('[bench] pick a HumanEval dataset first'); return }
        evalMsg(applied ? 'deleted' : 'clean')
      }
      // one-click: zero the region at the current top-k%, then benchmark THAT damaged model right away.
      // messages are processed in order on the connection, so the intervene lands before the eval reads weights.
      const deleteAndBench = () => {
        if (!ctrlEvalDs) { toast('[bench] pick a HumanEval dataset first'); return }
        apply()
        evalMsg('deleted')
      }
      const pctFmt = (v: number | undefined) => v == null ? '—' : `${(v * 100).toFixed(1)}%`
      return (
        <VizCard title="Region control" accent="var(--accent)"
          subtitle="Pick one of the regions you've found and adjust its weights — zero it (delete), scale by α, or randomize — then measure the effect on code vs general PPL. Reversible: Clear restores the original weights.">
          {regions.length === 0 ? <span style={hint}>no saved regions yet — compute a spot in the Spot view and Save it first.</span>
            : <>
              <div style={{ ...hint, marginBottom: 4 }}>Regions · {regions.length}</div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 14 }}>
                {regions.map((r) => (
                  <span key={r.name} onClick={() => pick(r.name)} className={`chip${sel === r.name ? ' on' : ''}`} style={{ cursor: 'pointer', display: 'inline-flex', gap: 6, alignItems: 'center' }}
                    title={`${r.count.toLocaleString()} weights${r.base_topk != null ? ` · saved at top ${pctLabel(r.base_topk)}%` : ''}`}>
                    <span className="mono">{r.name}</span><span style={hint}>{r.count.toLocaleString()}{r.base_topk != null ? ` · ${pctLabel(r.base_topk)}%` : ''}</span>
                  </span>
                ))}
              </div>
              {!sel ? <span style={hint}>select a region above to adjust it</span>
                : <>
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-end', marginBottom: 12 }}>
                    <label style={{ display: 'flex', flexDirection: 'column', gap: 3, fontSize: 11 }}><span style={hint}>operation</span>
                      <select value={ctrlOp} onChange={(e) => setCtrlOp(e.target.value)} style={{ fontSize: 12, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 8, padding: '3px 6px' }}>
                        {['zero', 'scale', 'mean', 'random'].map((o) => <option key={o} value={o}>{o === 'zero' ? 'zero (delete)' : o === 'scale' ? 'scale × α' : o}</option>)}
                      </select></label>
                    {ctrlOp === 'scale' && <label style={{ display: 'flex', flexDirection: 'column', gap: 3, fontSize: 11 }}><span style={hint}>α = {ctrlAlpha.toFixed(2)}</span>
                      <input type="range" min={0} max={2} step={0.05} value={Math.min(ctrlAlpha, 2)} onChange={(e) => setCtrlAlpha(Number(e.target.value))} style={{ width: 130 }} /></label>}
                    {(() => {
                      // log-scale slider over the paper range (0.0025%) up to 5% or the region's saved base, whichever is smaller
                      const lo = 0.000025, hi = Math.max(lo, base != null ? Math.min(0.05, base) : 0.05)
                      const Llo = Math.log10(lo), Lhi = Math.log10(hi), span = Lhi - Llo
                      const cur = Math.min(Math.max(topk, lo), hi)
                      const pos = span > 0 ? Math.round(((Math.log10(cur) - Llo) / span) * 1000) : 0
                      const fromPos = (pp: number) => span > 0 ? Math.pow(10, Llo + (pp / 1000) * span) : lo
                      const setTopk = (t: number) => { ctrlTopkRef.current = t; setCtrlTopk((tt) => ({ ...tt, [mid]: t })) }
                      const reapply = () => { if (applied) { const t = ctrlTopkRef.current; sendTo(mid, { type: 'intervene', region: { kind: 'named', name: sel, topk: t }, op: ctrlOp, alpha: ctrlAlpha, key: 'control' }); setCtrlApplied((a) => ({ ...a, [mid]: { name: sel, op: ctrlOp, alpha: ctrlAlpha, topk: t } })) } }
                      return <div style={{ display: 'flex', flexDirection: 'column', gap: 3, fontSize: 11 }}>
                        <span style={hint}>top-k %{base != null ? ` (≤ ${pctLabel(hi)})` : ''}{applied ? ' · change → re-apply' : ''}</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <input type="range" min={0} max={1000} value={pos}
                            onChange={(e) => setTopk(fromPos(Number(e.target.value)))} onPointerUp={reapply} style={{ width: 170 }} />
                          <input type="number" min={0.0001} max={+(hi * 100).toFixed(4)} step={0.001} value={+(topk * 100).toFixed(4)}
                            onChange={(e) => setTopk(Math.min(hi, Math.max(0.000001, (Number(e.target.value) || 0) / 100)))}
                            onBlur={reapply} onKeyDown={(e) => { if (e.key === 'Enter') reapply() }}
                            title="type an exact top-k% (e.g. 0.0025)" style={{ width: 72, fontSize: 12, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 8, padding: '3px 6px' }} />
                          <span style={hint}>%</span>
                        </div>
                      </div>
                    })()}
                    <div style={{ display: 'flex', gap: 6 }}>
                      <Btn onClick={apply} disabled={busy} color="var(--accent)" style={{ padding: '4px 12px' }}>Apply</Btn>
                      <Btn onClick={clear} disabled={busy} style={{ padding: '4px 12px' }}>Clear</Btn>
                      <Btn onClick={compare} disabled={busy} style={{ padding: '4px 12px' }} title="measure code+general PPL clean vs adjusted">{busy ? '⟳' : 'Compare PPL'}</Btn>
                    </div>
                  </div>
                  <div style={{ ...hint, fontSize: 11, marginBottom: 10 }}>
                    {applied ? <span style={{ color: 'var(--danger)' }}>● applied: {applied.op}{applied.op === 'scale' ? ` ×${applied.alpha.toFixed(2)}` : ''} on “{applied.name}” @ top {pctLabel(applied.topk)}% — weights are modified now</span>
                      : <span>● not applied — original weights</span>}
                  </div>
                  {(cp.cleanCode != null || cp.adjCode != null) && (
                    <div style={{ borderTop: '1px solid var(--line)', paddingTop: 10 }}>
                      <div style={{ color: 'var(--text-1)', marginBottom: 8, fontSize: 12, fontWeight: 600 }}>PPL · clean vs adjusted<span style={hint}> · code = {(dsExamples.length)} spot-editor examples · general = neutral text</span></div>
                      <div style={{ display: 'grid', gridTemplateColumns: '90px 1fr 1fr', gap: 1, maxWidth: 460, fontSize: 12 }}>
                        <div style={{ ...hint, padding: '3px 6px' }} />
                        <div style={{ padding: '3px 6px', fontWeight: 600, background: 'var(--bg-2)' }}>code PPL</div>
                        <div style={{ padding: '3px 6px', fontWeight: 600, background: 'var(--bg-2)' }}>general PPL</div>
                        <div style={{ ...hint, padding: '3px 6px' }}>clean</div>
                        <div className="mono" style={{ padding: '3px 6px', fontVariantNumeric: 'tabular-nums' }}>{fmt(cp.cleanCode)}</div>
                        <div className="mono" style={{ padding: '3px 6px', fontVariantNumeric: 'tabular-nums' }}>{fmt(cp.cleanGen)}</div>
                        <div style={{ ...hint, padding: '3px 6px' }}>adjusted</div>
                        <div className="mono" style={{ padding: '3px 6px', fontVariantNumeric: 'tabular-nums', color: (cp.adjCode != null && cp.cleanCode != null && cp.adjCode > cp.cleanCode * 3) ? 'var(--danger)' : 'var(--text-1)', fontWeight: 600 }}>{fmt(cp.adjCode)}</div>
                        <div className="mono" style={{ padding: '3px 6px', fontVariantNumeric: 'tabular-nums' }}>{fmt(cp.adjGen)}</div>
                      </div>
                      <div style={{ ...hint, fontSize: 10, marginTop: 6 }}>red = code PPL collapsed (&gt;3× clean). Damaging a real coding spot should spike code PPL while general stays put.</div>
                    </div>
                  )}
                  {/* Benchmark: HumanEval pass@1 on the current state (clean vs deleted) */}
                  <div style={{ borderTop: '1px solid var(--line)', paddingTop: 10, marginTop: 12 }}>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
                      <span style={{ color: 'var(--text-1)', fontSize: 12, fontWeight: 600 }}>Benchmark pass@1</span>
                      <select value={ctrlEvalDs} onChange={(e) => setCtrlEvalDs(e.target.value)} title="a HumanEvalPack dataset (rows with a `prompt`)"
                        style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 8, padding: '2px 6px', maxWidth: 260 }}>
                        <option value="">pick dataset…</option>
                        {datasets.filter((x) => x.name.endsWith('.jsonl')).map((x) => <option key={x.name} value={x.name}>{x.name}</option>)}
                      </select>
                      <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4, whiteSpace: 'nowrap', fontSize: 11 }} title="how many problems to evaluate (dataset has up to 164)">
                        <span style={hint}># problems</span>
                        <input type="number" min={1} value={ctrlEvalLimit} onChange={(e) => setCtrlEvalLimit(Math.max(1, Number(e.target.value) || 1))}
                          style={{ width: 56, fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 8, padding: '2px 6px' }} />
                      </label>
                      <Btn onClick={deleteAndBench} disabled={evalBusy} color="var(--accent)" style={{ padding: '4px 12px' }}
                        title={`zero “${sel}” at top ${pctLabel(topk)}% and benchmark that damaged model in one click`}>
                        {evalBusy ? `⟳ running ${shortRegion(sel)}` : `${ctrlOp === 'zero' ? 'Delete' : 'Apply'} ${shortRegion(sel)} @ ${pctLabel(topk)}% → benchmark`}</Btn>
                      <Btn onClick={runBench} disabled={evalBusy} style={{ padding: '4px 12px' }}
                        title="benchmark the CURRENT state as-is (clean if nothing applied, else the damaged model) — use for a clean baseline before deleting">
                        {applied ? 'Run on deleted' : 'Run on clean'}</Btn>
                    </div>
                    {gpus && gpus.count > 1 && (() => {
                      const home = gpus.devices.find((g) => g.models.includes(mid))?.index
                      const ex = ctrlEvalGpus[mid] ?? []
                      const toggle = (i: number) => { if (evalBusy) return; setCtrlEvalGpus((s) => ({ ...s, [mid]: ex.includes(i) ? ex.filter((x) => x !== i) : [...ex, i] })) }
                      return (
                        <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', fontSize: 11, marginBottom: 8, opacity: evalBusy ? 0.55 : 1 }}>
                          <span style={hint} title="split the benchmark problems across these GPUs too — faster, same pass@1">+ GPUs to help</span>
                          {gpus.devices.filter((g) => g.index !== home).map((g) => (
                            <span key={g.index} onClick={() => toggle(g.index)} title={evalBusy ? 'locked while running' : `${g.name} · ${Math.round(g.mem_used_mb / 1024)}/${Math.round(g.mem_total_mb / 1024)}G used`}
                              className={`chip${ex.includes(g.index) ? ' on' : ''}`} style={{ cursor: evalBusy ? 'default' : 'pointer' }}>
                              {ex.includes(g.index) ? '●' : '○'} GPU{g.index}
                            </span>
                          ))}
                          {ex.length > 0 && <span style={hint}>· runs on {1 + ex.length} GPUs in parallel{evalBusy ? ' (locked)' : ''}</span>}
                        </div>
                      )
                    })()}
                    {evalBusy && (() => {
                      const done = d.evalProg?.i ?? 0, tot = d.evalProg?.total ?? ctrlEvalLimit, pct = tot ? Math.round((done / tot) * 100) : 0
                      return <div style={{ marginBottom: 10 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
                          <span style={hint}>{done === 0 ? 'generating first solution…' : `solving · ${done}/${tot} · passed ${d.evalProg?.passed ?? 0}`}</span>
                          <span style={hint}>{pct}%</span>
                        </div>
                        <div style={{ background: 'var(--bg-2)', borderRadius: 6, height: 6, overflow: 'hidden' }}>
                          <div style={{ height: 6, width: `${pct}%`, background: 'var(--accent)', borderRadius: 6, transition: 'width var(--ease)' }} />
                        </div>
                      </div>
                    })()}
                    {(ce.clean || ce.deleted) && (
                      <div style={{ display: 'grid', gridTemplateColumns: '90px 1fr', gap: 1, maxWidth: 360, fontSize: 12 }}>
                        <div style={{ ...hint, padding: '3px 6px' }} /><div style={{ padding: '3px 6px', fontWeight: 600, background: 'var(--bg-2)' }}>pass@1</div>
                        <div style={{ ...hint, padding: '3px 6px' }}>clean</div>
                        <div className="mono" style={{ padding: '3px 6px', fontVariantNumeric: 'tabular-nums' }}>{ce.clean ? `${pctFmt(ce.clean.pass_at_1)} (${ce.clean.passed}/${ce.clean.total})` : '—'}</div>
                        <div style={{ ...hint, padding: '3px 6px' }}>deleted</div>
                        <div className="mono" style={{ padding: '3px 6px', fontVariantNumeric: 'tabular-nums', color: (ce.deleted && ce.clean && ce.deleted.pass_at_1 < ce.clean.pass_at_1) ? 'var(--danger)' : 'var(--text-1)', fontWeight: 600 }}>{ce.deleted ? `${pctFmt(ce.deleted.pass_at_1)} (${ce.deleted.passed}/${ce.deleted.total})` : '—'}</div>
                      </div>
                    )}
                    <div style={{ ...hint, fontSize: 10, marginTop: 6 }}>Clear → Run (clean baseline), then Apply delete → Run (deleted). A real coding spot should drop pass@1 sharply. limit caps # problems — full set is slower.</div>
                  </div>
                </>}
            </>}
        </VizCard>
      )
    }
    if (view === 'log') {
      const short = (id: string) => open.find((m) => m.id === id)?.label ?? id
      const toCSV = () => ['ts,model,kind,text,py', ...expLog.map((e) => [e.ts, short(e.model), e.kind, e.text, e.py].map((v) => `"${String(v).replace(/"/g, '""')}"`).join(','))].join('\n')
      const toScript = () => `# parametic-studio session replay — s = ModelSession.from_pretrained("<model>")\n# examples = your dataset lines\n` + expLog.filter((e) => e.py).map((e) => `${e.py}  # ${e.ts.slice(11, 19)} ${short(e.model)}`).join('\n')
      return (<>
        <div style={{ color: 'var(--text-1)', marginBottom: 6 }}>experiment log<span style={hint}> · {expLog.length} entries · session-only (not persisted)</span></div>
        <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
          <Btn onClick={() => setLogPy((v) => !v)} color={logPy ? 'var(--accent)' : 'var(--text-1)'} title="show each action as runnable ModelSession python" style={{ padding: '2px 10px' }}>{logPy ? '●' : '○'} py</Btn>
          <Btn onClick={() => download('experiment_log.json', JSON.stringify(expLog, null, 2), 'application/json')} style={{ padding: '2px 10px' }}>JSON</Btn>
          <Btn onClick={() => download('experiment_log.csv', toCSV(), 'text/csv')} style={{ padding: '2px 10px' }}>CSV</Btn>
          {logPy && <Btn onClick={() => download('replay.py', toScript(), 'text/x-python')} style={{ padding: '2px 10px' }}>replay.py</Btn>}
          <Btn onClick={() => setExpLog([])} color="var(--text-2)" style={{ padding: '2px 10px' }}>Clear</Btn>
        </div>
        {expLog.length === 0 && <span style={hint}>actions and results will be recorded here</span>}
        <div ref={logScrollRef} onScroll={(e) => { const el = e.currentTarget; logPinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24 }}
          style={{ maxHeight: '100%', overflow: 'auto' }}>
          {logPy
            ? <pre className="mono" style={{ fontSize: 11, whiteSpace: 'pre-wrap', background: 'var(--bg-2)', borderRadius: 10, padding: 8 }}>{expLog.filter((e) => e.py).map((e) => e.py).join('\n') || '# no actions yet'}</pre>
            : <div className="mono" style={{ display: 'grid', gridTemplateColumns: '58px 74px 1fr', gap: '2px 8px', fontSize: 11 }}>
                {expLog.map((e, i) => (<Fragment key={i}>
                  <span style={hint}>{e.ts.slice(11, 19)}</span>
                  <span style={{ color: 'var(--text-2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{short(e.model)}</span>
                  <span style={{ color: e.kind === 'result' ? 'var(--accent)' : 'var(--text-1)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={e.py || e.text}>{e.kind === 'result' ? '→ ' : ''}{e.text}</span>
                </Fragment>))}
              </div>}
        </div>
      </>)
    }
    if (view === 'eval') {
      const sel = { fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '2px 4px' } as const
      const damaged = d.knobs.length > 0
      const busy = pending.has(`eval_code:${mid}`)
      const canRun = !busy && codeEvalDsName !== ''
      const runEval = () => {
        const limit = Number(codeEvalLimit)
        sendTo(mid, {
          type: 'eval_code', dataset: codeEvalDsName, temperature: codeEvalTemp, max_tokens: codeEvalMaxTokens,
          ...(codeEvalLimit !== '' && limit > 0 ? { limit } : {}),
        })
      }
      const fmt = (r: { dataset: string; passed: number; total: number; pass_at_1: number; damaged: boolean; knobCount: number }) => (
        <div style={{ padding: 10, border: '1px solid var(--line-strong)', borderRadius: 12 }}>
          <div style={{ fontSize: 22, color: 'var(--text-0)' }}>pass@1 = {(r.pass_at_1 * 100).toFixed(2)}%</div>
          <div style={{ ...hint, fontSize: 11, marginTop: 2 }}>{r.passed}/{r.total} passed · {r.dataset}</div>
          <div style={{ fontSize: 11, marginTop: 4, color: r.damaged ? 'var(--danger)' : 'var(--accent)' }}>{r.damaged ? `damaged model (${r.knobCount} knobs)` : 'clean model'}</div>
        </div>
      )
      return (<>
        <div style={{ color: 'var(--text-1)', marginBottom: 6 }}>pass@k eval<span style={hint}> · HumanEvalPack — measures coding ability by test-pass rate</span></div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginBottom: 6 }}>
          <select value={codeEvalDsName} onChange={(e) => setCodeEvalDsName(e.target.value)} style={sel} title="load humanevalpack via + HF first">
            <option value="">dataset…</option>
            {datasets.map((x) => <option key={x.name} value={x.name}>{x.name} ({dsMeta[x.name]?.count ?? '…'})</option>)}
          </select>
          {datasets.length === 0 && <span style={hint}>load humanevalpack via + HF first</span>}
          <span style={hint}>temp</span><input type="number" min={0} step={0.1} value={codeEvalTemp} onChange={(e) => setCodeEvalTemp(Math.max(0, Number(e.target.value) || 0))} title="0 = greedy (pass@1)" style={{ ...sel, width: 48 }} />
          <span style={hint}>max_tokens</span><input type="number" min={1} value={codeEvalMaxTokens} onChange={(e) => setCodeEvalMaxTokens(Math.max(1, Math.round(Number(e.target.value)) || 1))} style={{ ...sel, width: 60 }} />
          <span style={hint}>limit</span><input type="number" min={1} value={codeEvalLimit} onChange={(e) => { const v = e.target.value; if (v === '' || Number(v) >= 1) setCodeEvalLimit(v) }} placeholder="all" title="cap the number of problems — for a quick test run" style={{ ...sel, width: 54 }} />
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 10 }}>
          <Btn onClick={runEval} disabled={!canRun} color={canRun ? 'var(--accent)' : 'var(--text-2)'}>{busy ? '⟳ ' : ''}Run</Btn>
          <span style={{ fontSize: 11, color: damaged ? 'var(--danger)' : 'var(--text-2)' }}>{damaged ? `evaluating DAMAGED model (${d.knobs.length} knobs)` : 'evaluating clean model'}</span>
        </div>
        {d.evalProg && (
          <div style={{ marginBottom: 10 }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}><span style={hint}>{d.evalProg.i}/{d.evalProg.total} · passed {d.evalProg.passed}</span></div>
            <div style={{ background: 'var(--bg-2)', borderRadius: 6, height: 4, marginTop: 3 }}><div style={{ height: 4, width: `${Math.round((d.evalProg.i / d.evalProg.total) * 100)}%`, background: 'var(--accent)', borderRadius: 6 }} /></div>
          </div>
        )}
        {d.evalResult && (
          <div style={{ display: 'grid', gridTemplateColumns: d.evalPrev ? '1fr 1fr' : '1fr', gap: 10 }}>
            <div><div style={{ ...hint, fontSize: 11, marginBottom: 3 }}>latest</div>{fmt(d.evalResult)}</div>
            {d.evalPrev && <div><div style={{ ...hint, fontSize: 11, marginBottom: 3 }}>previous · compare</div>{fmt(d.evalPrev)}</div>}
          </div>
        )}
        {!d.evalResult && !d.evalProg && <span style={hint}>pick a dataset and Run</span>}
      </>)
    }
    if (view === 'train') {
      const tr = d.train
      const needsRegion = trainMode.startsWith('spot')
      const dsLoading = trainDsName !== '' && dsMeta[trainDsName] == null   // selected but content not parsed yet
      const trainEx = trainDsName ? (dsMeta[trainDsName]?.examples ?? []).slice(0, trainLimit) : []
      const canTrain = !tr.running && !tr.trained && (!needsRegion || trainRegion !== '') && trainEx.length > 0
      const pickTrainDs = (name: string) => { setTrainDsName(name); const dset = datasets.find((x) => x.name === name); if (dset && dset.content == null) sendTo(mid, { type: 'read_dataset', name }) }
      const runTrain = () => {
        patch(mid, (dd) => ({ ...dd, train: { ...emptyTrain(), running: true }, knobs: [], kppl: { base: null, inter: null } }))  // kernel auto-clears knobs
        sendTo(mid, { type: 'ppl', examples: PRESETS.python.split('\n').filter(Boolean), tag: 'code' })     // pre-train evals → "before"
        sendTo(mid, { type: 'ppl', examples: GENERAL_SET.split('\n').filter(Boolean), tag: 'general' })
        sendTo(mid, {
          type: 'train', mode: trainMode, examples: trainEx,
          steps: Math.max(1, trainEpochs * trainEx.length), lr: trainLr, lora_dim: 8, ...(needsRegion ? { region: { kind: 'named', name: trainRegion, topk: trainRegionTopk } } : {}),
        })
      }
      const sel = { fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '2px 4px' } as const
      const L = tr.losses
      const MODES = [
        { key: 'full', label: 'Full', desc: 'every weight trains', region: false },
        { key: 'spot-freeze', label: 'Freeze region', desc: 'the region is frozen · everything else trains — protect the spot while adapting', region: true },
        { key: 'spot-only', label: 'Only region', desc: 'only the region trains · everything else (the complement) is frozen', region: true },
        { key: 'lora', label: 'LoRA', desc: 'base frozen · small adapters train', region: false },
      ]
      const modeInfo = MODES.find((m) => m.key === trainMode) ?? MODES[0]
      // which side is frozen in the current mode → the freeze diagram
      const froz = (part: 'region' | 'rest') => trainMode === 'lora' ? true : trainMode === 'full' ? false : trainMode === 'spot-freeze' ? part === 'region' : part === 'rest'
      const seg = (part: 'region' | 'rest', flex: number, txt: string) => (
        <div style={{ flex, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, height: 22, color: froz(part) ? 'var(--text-2)' : '#fff',
          background: froz(part) ? 'repeating-linear-gradient(45deg, var(--bg-2), var(--bg-2) 4px, var(--line) 4px, var(--line) 5px)' : 'var(--accent)', whiteSpace: 'nowrap', overflow: 'hidden' }}>{txt}</div>
      )
      const td = trainDelta[mid]
      const deltaBusy = pending.has(`training_delta:${mid}`)
      const showDelta = () => { if (!needsRegion || !trainRegion) { sendTo(mid, { type: 'training_delta' }); setPendingKey(`training_delta:${mid}`, true); return } setPendingKey(`training_delta:${mid}`, true); sendTo(mid, { type: 'training_delta', region: { kind: 'named', name: trainRegion, topk: trainRegionTopk } }) }
      const dPPL = (before: number | null | undefined, after: number | null | undefined) => {
        if (before == null || after == null) return null
        const delta = after - before
        return <span style={{ color: delta < 0 ? 'var(--accent)' : delta > 0 ? 'var(--danger)' : 'var(--text-2)', marginLeft: 4 }}>{delta < 0 ? '▼' : delta > 0 ? '▲' : ''}{Math.abs(delta).toPrecision(3)}</span>
      }
      return (<>
        <div style={{ color: 'var(--text-1)', marginBottom: 8 }}>region-aware fine-tuning<span style={hint}> · reversible via reset</span></div>
        {/* mode picker — segmented, human-labelled */}
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 4 }}>
          {MODES.map((m) => (
            <span key={m.key} onClick={() => { if (!tr.running) setTrainMode(m.key) }} className={`chip${trainMode === m.key ? ' on' : ''}`}
              style={{ cursor: tr.running ? 'default' : 'pointer', opacity: tr.running && trainMode !== m.key ? 0.5 : 1 }}>{m.label}</span>
          ))}
        </div>
        <div style={{ ...hint, fontSize: 11, marginBottom: 8 }}>{modeInfo.desc}</div>
        {/* region picker + freeze diagram */}
        {needsRegion && (() => {
          const rbase = d.regions.find((r) => r.name === trainRegion)?.base_topk
          return (
          <div style={{ marginBottom: 8 }}>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginBottom: 6 }}>
              <select value={trainRegion} onChange={(e) => setTrainRegion(e.target.value)} style={sel}>
                <option value="">pick region…</option>
                {d.regions.map((r) => <option key={r.name} value={r.name}>◈ {r.name}</option>)}
              </select>
              {trainRegion && <>
                <span style={hint}>top-k%</span>
                {SPOT_TOPK_OPTIONS.map((k) => {
                  const disabled = rbase != null && k > rbase  // can only re-threshold DOWN from the saved base
                  return <span key={k} onClick={() => { if (!disabled && !tr.running) setTrainRegionTopk(k) }} className={`chip${trainRegionTopk === k ? ' on' : ''}`}
                    title={disabled ? `region saved at top ${pctLabel(rbase!)}% — can't expand to ${pctLabel(k)}%` : `freeze/train the top ${pctLabel(k)}% of the region`}
                    style={{ cursor: disabled || tr.running ? 'default' : 'pointer', opacity: disabled ? 0.4 : 1 }}>{pctLabel(k)}%</span>
                })}
              </>}
            </div>
            {trainRegion && <>
              <div style={{ display: 'flex', borderRadius: 6, overflow: 'hidden', border: '1px solid var(--line-strong)' }}>
                {seg('region', 1, `◈ region ${pctLabel(trainRegionTopk)}% ${froz('region') ? '(frozen)' : '(trains)'}`)}
                {seg('rest', 5, `everything else ${froz('rest') ? '(frozen)' : '(trains)'}`)}
              </div>
              <div style={{ ...hint, fontSize: 10, marginTop: 3 }}>▨ = frozen (gradients blocked) · ▮ = trains · region = top {pctLabel(trainRegionTopk)}% by importance</div>
            </>}
          </div>
          )
        })()}
        {/* training data = a benchmark/dataset from the store (one record = one example) */}
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginBottom: 4 }}>
          <span style={hint}>benchmark</span>
          <select value={trainDsName} onChange={(e) => pickTrainDs(e.target.value)} style={sel} title="fine-tune on this dataset — each record is one training example">
            <option value="">pick dataset…</option>
            {datasets.map((x) => <option key={x.name} value={x.name}>{x.name} ({dsMeta[x.name]?.count ?? '…'})</option>)}
          </select>
          {trainDsName && <><span style={hint}>use</span><input type="number" min={1} value={trainLimit} onChange={(e) => setTrainLimit(Math.max(1, Math.round(Number(e.target.value)) || 1))} style={{ ...sel, width: 56 }} /><span style={hint}>examples</span></>}
        </div>
        {trainDsName && (dsLoading
          ? <div style={{ ...hint, fontSize: 11, marginBottom: 6 }}>loading {trainDsName}…</div>
          : <div style={{ ...hint, fontSize: 11, marginBottom: 6 }}>{trainEx.length} examples × {trainEpochs} epoch{trainEpochs > 1 ? 's' : ''} = {trainEpochs * trainEx.length} weight updates (each example seen {trainEpochs}×)</div>)}
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
          <span style={hint}>epochs</span><input type="number" min={1} value={trainEpochs} onChange={(e) => setTrainEpochs(Math.max(1, Math.round(Number(e.target.value)) || 1))} title="passes over the dataset — each example is seen this many times" style={{ ...sel, width: 48 }} />
          <span style={hint}>Learning Rate</span><input type="number" step={1e-5} value={trainLr} onChange={(e) => { const v = Number(e.target.value); if (v > 0) setTrainLr(v) }} style={{ ...sel, width: 72 }} />
        </div>
        <div style={{ display: 'flex', gap: 6, marginBottom: 8, alignItems: 'center' }}>
          <Btn onClick={runTrain} disabled={!canTrain} color={canTrain ? 'var(--accent)' : 'var(--text-2)'}>Train</Btn>
          {tr.running && <Btn onClick={() => sendTo(mid, { type: 'stop_train' })} color="var(--danger)">Stop</Btn>}
          {tr.trained && <Btn onClick={() => sendTo(mid, { type: 'reset_train' })} title="restore pre-training weights">Reset</Btn>}
          {tr.running && <span style={{ color: 'var(--live)', fontSize: 11 }}>● training {L.length}/{tr.total}</span>}
          {tr.trained && <span style={{ color: 'var(--accent)', fontSize: 11 }}>trained · weights modified</span>}
          {needsRegion && !trainRegion && !tr.running && <span style={{ ...hint, fontSize: 11 }}>pick a region first</span>}
        </div>
        {tr.error && <div style={{ color: 'var(--danger)', fontSize: 11, marginBottom: 6 }}>{tr.error}</div>}
        {L.length > 0 && (() => {
          const mx = Math.max(...L), mn = Math.min(...L)
          // per-step loss is on ONE (cycling) example → very noisy. Overlay an EMA so the trend is readable.
          const ema: number[] = []; let e = L[0]; for (const v of L) { e = 0.8 * e + 0.2 * v; ema.push(e) }
          const xy = (v: number, i: number) => `${(i / Math.max(1, tr.total - 1)) * 260},${36 - ((v - mn) / (mx - mn || 1)) * 30}`
          const raw = L.map(xy).join(' ')
          const smooth = ema.map(xy).join(' ')
          const diverging = L.length >= 6 && ema[ema.length - 1] > ema[Math.max(0, ema.length - 6)] * 1.15  // trend climbing
          return (<div style={{ marginBottom: 8 }}>
            <div style={hint}>loss <span className="mono">{L[L.length - 1].toFixed(3)}</span> · trend <span className="mono" style={{ color: diverging ? 'var(--danger)' : 'var(--accent)' }}>{ema[ema.length - 1].toFixed(3)}</span> · min <span className="mono">{mn.toFixed(3)}</span></div>
            <svg width="100%" height={40} viewBox="0 0 260 40" preserveAspectRatio="none" style={{ background: 'var(--bg-2)', borderRadius: 10 }}>
              <polyline points={raw} fill="none" stroke="var(--line-strong)" strokeWidth={0.8} opacity={0.7} />
              <polyline points={smooth} fill="none" stroke={diverging ? 'var(--danger)' : 'var(--accent)'} strokeWidth={1.6} />
            </svg>
            {diverging && <div style={{ color: 'var(--danger)', fontSize: 11, marginTop: 3 }}>⚠ loss is climbing — likely the learning rate is too high. Lower <b>Learning Rate</b> (try 1e-5) and retrain.</div>}
            <div style={{ ...hint, fontSize: 10, marginTop: 2 }}>faint line = per-step (one example, noisy) · bold = smoothed trend</div>
          </div>)
        })()}
        {/* performance before vs after (Δ) */}
        {tr.before && (
          <div style={{ fontSize: 11, marginBottom: 8 }}>
            <div style={{ ...hint, marginBottom: 3 }}>performance (PPL · ▼ lower = better)</div>
            <div style={{ display: 'grid', gridTemplateColumns: '64px 1fr 1fr', gap: 6, alignItems: 'baseline' }}>
              <span /><span style={hint}>code</span><span style={hint}>general</span>
              <span style={hint}>before</span>
              <span className="mono">{tr.before.code?.toPrecision(4) ?? '—'}</span><span className="mono">{tr.before.general?.toPrecision(4) ?? '—'}</span>
              <span style={hint}>after</span>
              <span className="mono" style={{ color: 'var(--text-0)' }}>{d.evals.code?.toPrecision(4) ?? '…'}{dPPL(tr.before.code, d.evals.code)}</span>
              <span className="mono" style={{ color: 'var(--text-0)' }}>{d.evals.general?.toPrecision(4) ?? '…'}{dPPL(tr.before.general, d.evals.general)}</span>
            </div>
          </div>
        )}
        {/* region change: how far weights moved (validates the freeze) */}
        {tr.trained && (
          <div style={{ borderTop: '1px solid var(--line)', paddingTop: 8 }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: td ? 6 : 0 }}>
              <span style={{ ...hint, fontSize: 11 }}>region change</span>
              <Btn onClick={showDelta} disabled={deltaBusy} style={{ padding: '2px 10px' }}>{deltaBusy ? '⟳' : 'What moved?'}</Btn>
            </div>
            {td && (
              <div style={{ fontSize: 11 }}>
                <div style={{ display: 'grid', gridTemplateColumns: '150px 1fr', gap: '3px 8px', alignItems: 'baseline' }}>
                  <span style={hint}>region weights (RMS Δ)</span>
                  <span className="mono" style={{ color: (td.region_rms ?? 0) < 1e-6 ? 'var(--text-2)' : 'var(--accent)' }}>{td.region_rms == null ? '—' : td.region_rms.toExponential(2)}{td.region_rms != null && td.region_rms < 1e-6 ? '  (frozen ✓)' : ''}</span>
                  {td.other_rms != null && <>
                    <span style={hint}>everything else (RMS Δ)</span>
                    <span className="mono" style={{ color: td.other_rms < 1e-6 ? 'var(--text-2)' : 'var(--accent)' }}>{td.other_rms.toExponential(2)}{td.other_rms < 1e-6 ? '  (frozen ✓)' : ''}</span>
                  </>}
                </div>
                <div style={{ ...hint, fontSize: 10, marginTop: 5 }}>RMS of (weight after − before). {trainMode === 'spot-freeze' ? 'region should read ~0 (frozen), rest > 0.' : trainMode === 'spot-only' ? 'only region moves; rest untouched.' : 'measures how far the run pushed the weights.'}</div>
              </div>
            )}
          </div>
        )}
        {/* benchmark pass@1 — base (pre-train) vs trained. eval fans out over GPUs (unlike training). */}
        {(() => {
          const te = trainEval[mid] ?? {}
          const evalBusy = pending.has(`eval_code:${mid}`)
          const pct = (v: number | undefined) => v == null ? '—' : `${(v * 100).toFixed(1)}%`
          const runBench = () => {
            if (!trainEvalDs) { toast('[bench] pick a HumanEval dataset first'); return }
            trainEvalSlot.current[mid] = tr.trained ? 'trained' : 'base'
            const extra = (trainEvalGpus[mid] ?? []).map((i) => `cuda:${i}`)
            setPendingKey(`eval_code:${mid}`, true)
            sendTo(mid, { type: 'eval_code', dataset: trainEvalDs, temperature: 0, max_tokens: 512, limit: trainEvalLimit, ...(extra.length ? { extra_devices: extra } : {}) })
          }
          const dp = (te.base && te.trained) ? te.trained.pass_at_1 - te.base.pass_at_1 : null
          return (
            <div style={{ borderTop: '1px solid var(--line)', marginTop: 8, paddingTop: 8 }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
                <span style={{ color: 'var(--text-1)', fontSize: 12, fontWeight: 600 }}>Benchmark pass@1</span>
                <select value={trainEvalDs} onChange={(e) => setTrainEvalDs(e.target.value)} style={sel} title="a HumanEvalPack dataset">
                  <option value="">pick dataset…</option>
                  {datasets.filter((x) => x.name.endsWith('.jsonl')).map((x) => <option key={x.name} value={x.name}>{x.name}</option>)}
                </select>
                <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4, whiteSpace: 'nowrap', fontSize: 11 }}><span style={hint}># problems</span>
                  <input type="number" min={1} value={trainEvalLimit} onChange={(e) => setTrainEvalLimit(Math.max(1, Number(e.target.value) || 1))} style={{ ...sel, width: 56 }} /></label>
                <Btn onClick={runBench} disabled={evalBusy} color="var(--accent)" style={{ padding: '3px 10px' }}
                  title={tr.trained ? 'benchmark the trained model' : 'benchmark the base model — do this before training for a baseline'}>{evalBusy ? '⟳ running' : tr.trained ? 'Run on trained' : 'Run on base'}</Btn>
              </div>
              {gpus && gpus.count > 1 && (() => {
                const home = gpus.devices.find((g) => g.models.includes(mid))?.index
                const ex = trainEvalGpus[mid] ?? []
                const toggle = (i: number) => { if (evalBusy) return; setTrainEvalGpus((s) => ({ ...s, [mid]: ex.includes(i) ? ex.filter((x) => x !== i) : [...ex, i] })) }
                return <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', fontSize: 11, marginBottom: 8, opacity: evalBusy ? 0.55 : 1 }}>
                  <span style={hint} title="split the benchmark problems across these GPUs too">+ GPUs to help</span>
                  {gpus.devices.filter((g) => g.index !== home).map((g) => <span key={g.index} onClick={() => toggle(g.index)} className={`chip${ex.includes(g.index) ? ' on' : ''}`} style={{ cursor: evalBusy ? 'default' : 'pointer' }}>{ex.includes(g.index) ? '●' : '○'} GPU{g.index}</span>)}
                  {ex.length > 0 && <span style={hint}>· {1 + ex.length} GPUs in parallel</span>}
                </div>
              })()}
              {evalBusy && (() => { const done = d.evalProg?.i ?? 0, tot = d.evalProg?.total ?? trainEvalLimit, passed = d.evalProg?.passed ?? 0, p = tot ? Math.round(done / tot * 100) : 0; return <div style={{ marginBottom: 8 }}><div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}><span style={hint}>{done === 0 ? 'generating first solution… (slow on the first problem)' : `solving · ${done}/${tot} · passed ${passed}`}</span><span style={hint}>{p}%</span></div><div style={{ background: 'var(--bg-2)', borderRadius: 6, height: 6, overflow: 'hidden' }}><div style={{ height: 6, width: `${p}%`, background: 'var(--accent)', borderRadius: 6 }} /></div></div> })()}
              {(te.base || te.trained) && (
                <div style={{ display: 'grid', gridTemplateColumns: '84px 1fr', gap: 1, maxWidth: 340, fontSize: 12 }}>
                  <div style={{ ...hint, padding: '3px 6px' }} /><div style={{ padding: '3px 6px', fontWeight: 600, background: 'var(--bg-2)' }}>pass@1</div>
                  <div style={{ ...hint, padding: '3px 6px' }}>base</div><div className="mono" style={{ padding: '3px 6px' }}>{te.base ? `${pct(te.base.pass_at_1)} (${te.base.passed}/${te.base.total})` : '—'}</div>
                  <div style={{ ...hint, padding: '3px 6px' }}>trained</div><div className="mono" style={{ padding: '3px 6px', fontWeight: 600, color: dp != null && dp !== 0 ? (dp > 0 ? 'var(--accent)' : 'var(--danger)') : undefined }}>{te.trained ? `${pct(te.trained.pass_at_1)} (${te.trained.passed}/${te.trained.total})` : '—'}{dp != null && dp !== 0 ? ` (${dp > 0 ? '+' : ''}${(dp * 100).toFixed(1)}%)` : ''}</div>
                </div>
              )}
              <div style={{ ...hint, fontSize: 10, marginTop: 6 }}>measure <b>base</b> first (before Train), then Train → measure <b>trained</b>. higher pass@1 = training improved coding.</div>
            </div>
          )
        })()}
      </>)
    }
    // if the editor still shows the untouched preview, recompute on the FULL cached set (parsing the
    // truncated preview would silently shrink the run); otherwise parse whatever the user typed.
    return (<>
      <div style={{ color: 'var(--text-1)', marginBottom: 6 }}>dataset → grad×param<span style={hint}> · pick a dataset · top cells = spot</span></div>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 10, flexWrap: 'wrap' }}>
        <select value={spotSrc} onChange={(e) => e.target.value && computeSpotFor(mid, e.target.value)} title="pick a dataset — samples and computes its spot" style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '2px 4px' }}>
          <option value="">▤ dataset…</option>
          {datasets.map((x) => <option key={x.name} value={x.name}>{x.name} ({dsMeta[x.name]?.count ?? '…'})</option>)}
        </select>
        <span style={hint}>sample</span>
        <input type="number" min={1} value={spotN} onChange={(e) => { const v = e.target.value; if (v === '' || Number(v) >= 1) setSpotN(v) }} placeholder="all" title="how many examples to use (empty = all)" style={{ width: 54, fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '2px 4px' }} />
        <select value={spotPick} onChange={(e) => setSpotPick(e.target.value as 'first' | 'random')} title="first-k or a random sample" style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '2px 4px' }}>
          <option value="first">first-k</option>
          <option value="random">random</option>
        </select>
        <span style={hint}>region top</span>
        {SPOT_TOPK_OPTIONS.map((k) => (
          <span key={k} onClick={() => {
            setSpotTopk(k)
            setNamedRegionTopk(k)
            patch(mid, (dd) => ({ ...dd, knobs: dd.knobs.map((row) => row.kind === 'spot' ? { ...row, topk: k } : row) }))
            const ex = spotExamples[mid]?.length ? spotExamples[mid] : dsExamples
            for (const row of d.knobs.filter((x) => x.kind === 'spot')) {
              sendTo(mid, { type: 'intervene', region: { kind: 'spot', examples: ex, topk: k }, op: row.op, alpha: row.alpha, key: row.key })
            }
          }} className={`chip${spotTopk === k ? ' on' : ''}`} title="parameter region size used for saved regions, spot knobs, and spot activation">
            {pctLabel(k)}%
          </span>
        ))}
        <Btn onClick={() => spotSrc && computeSpotFor(mid, spotSrc)} disabled={!spotSrc} color="var(--accent)" style={{ padding: '4px 12px' }}>Compute spot</Btn>
        {spotSrc && <span style={hint}>on {spotSrc}{spotExamples[mid]?.length ? ` · ${spotExamples[mid].length} ex` : ''}</span>}
      </div>
      {gpus && gpus.count > 1 && (() => {
        const home = gpus.devices.find((g) => g.models.includes(mid))?.index
        const extra = spotExtraGpus[mid] ?? []
        const computing = !!d.spotProg  // selection is locked in once compute starts — mid-run toggles wouldn't apply
        const toggle = (i: number) => { if (computing) return; setSpotExtraGpus((s) => ({ ...s, [mid]: extra.includes(i) ? extra.filter((x) => x !== i) : [...extra, i] })) }
        return (
          <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 10, flexWrap: 'wrap', fontSize: 11, opacity: computing ? 0.55 : 1 }}>
            <span style={hint} title="split the backward passes across these GPUs too — faster, no change to the result">+ GPUs to help</span>
            {gpus.devices.filter((g) => g.index !== home).map((g) => (
              <span key={g.index} onClick={() => toggle(g.index)} title={computing ? 'locked while computing' : `${g.name} · ${Math.round(g.mem_used_mb / 1024)}/${Math.round(g.mem_total_mb / 1024)}G used`}
                className={`chip${extra.includes(g.index) ? ' on' : ''}`} style={{ cursor: computing ? 'default' : 'pointer' }}>
                {extra.includes(g.index) ? '●' : '○'} GPU{g.index}
              </span>
            ))}
            {extra.length > 0 && <span style={hint}>· spot runs on {1 + extra.length} GPUs in parallel{computing ? ' (locked)' : ''}</span>}
          </div>
        )
      })()}
      {d.spotProg && <div style={{ marginBottom: 8 }}><div style={{ display: 'flex', gap: 8, alignItems: 'center' }}><span style={hint}>computing · {d.spotProg.i}/{d.spotProg.total}</span><Btn onClick={() => sendTo(mid, { type: 'stop_spot' })} color="var(--danger)" style={{ padding: '0 8px' }}>Stop</Btn></div><div style={{ background: 'var(--bg-2)', borderRadius: 6, height: 4, marginTop: 3 }}><div style={{ height: 4, width: `${Math.round((d.spotProg.i / d.spotProg.total) * 100)}%`, background: 'var(--accent)', borderRadius: 6 }} /></div></div>}
      {d.spot && (() => {
        const spot = d.spot
        const examples = spotExamples[mid]?.length ? spotExamples[mid] : dsExamples  // reuse the exact set THIS model's spot ran on (cache hit on save); fall back to the editor before any compute
        const evalExamples = evalDsName ? (dsMeta[evalDsName]?.examples ?? examples) : examples  // kppl measurement set — may differ from spot data
        const selected = new Set(d.knobs.map((k) => k.key))
        const measure = () => sendTo(mid, { type: 'ppl', examples: evalExamples, tag: 'inter' })
        const region = (k: KnobRow) => k.kind === 'spot' ? { kind: 'spot', examples, topk: k.topk ?? spotTopk }
          : k.kind === 'named' ? { kind: 'named', name: k.name, ...(k.topk != null ? { topk: k.topk } : {}) }
          : { kind: 'cell', layer: k.layer, module: k.module }
        const sendKnob = (k: KnobRow) => { sendTo(mid, { type: 'intervene', region: region(k), op: k.op, alpha: k.alpha, key: k.key }); measure() }
        const baselineOnce = () => { if (d.knobs.length === 0 && d.kppl.base == null) sendTo(mid, { type: 'ppl', examples: evalExamples, tag: 'base' }) }  // clean model baseline first
        const addKnob = (row: KnobRow) => { baselineOnce(); patch(mid, (dd) => ({ ...dd, knobs: [...dd.knobs, row] })); sendKnob(row) }
        const addCell = (l: number, module: string) => {
          const key = `${l}.${module}`
          if (selected.has(key)) return
          addKnob({ key, kind: 'cell', layer: l, module, op: 'scale', alpha: 0 })
        }
        const runAB = () => {
          abPhase.current[mid] = 'base'
          patch(mid, (dd) => ({ ...dd, ab: { base: null, inter: null }, output: '', busy: true }))
          sendTo(mid, { type: 'suspend' })            // baseline pass runs on original weights
          sendTo(mid, { type: 'generate', prompt: promptRef.current, max_tokens: genRef.current.maxTokens, temperature: genRef.current.temperature, probes: [] })
        }
        const adjust = (key: string, p: Partial<KnobRow>) => {
          patch(mid, (dd) => ({ ...dd, knobs: dd.knobs.map((k) => (k.key === key ? { ...k, ...p } : k)) }))
          const k = d.knobs.find((x) => x.key === key); if (k) sendKnob({ ...k, ...p })
        }
        const remove = (key: string) => { sendTo(mid, { type: 'clear', key }); patch(mid, (dd) => ({ ...dd, knobs: dd.knobs.filter((k) => k.key !== key) })); measure() }
        const clearAll = () => { sendTo(mid, { type: 'clear' }); patch(mid, (dd) => ({ ...dd, knobs: [], kppl: { base: null, inter: null } })) }
        const { base, inter } = d.kppl
        const shownSpotLayer = Math.min(Math.max(0, hoverLayer ?? layer), Math.max(0, spot.grid.length - 1))
        const spotScore = normalizedScoreGrid(spot.grid)
        return (<>
          <VizCard title="Coding spot" accent="var(--syn-num, #e0a85e)"
            right={<span className="badge">{spot.layers} × {spot.modules.length}</span>}
            subtitle="Per-parameter |gradient × weight| importance, reduced to layer × module, then normalized to 0–100 for comparison. Bright cells hold the coding spot. Hover a layer to inspect module scores; click a cell to add it as a knob.">
            <ScaleBar max={100} color={ampColor} label="score 0–100" />
            <div style={{ marginTop: 4 }}><SpotGrid grid={spotScore.grid} modules={spot.modules} onCell={(l, mod) => { setLayer(l); addCell(l, mod) }} selected={selected}
              onHover={(cell) => setHoverLayer(cell ? Number(cell.split('.')[0]) : null)}
              hovered={hoverLayer != null ? `${hoverLayer}.${spot.modules[0] ?? ''}` : null}
              cellTitle={(l, mod, v) => `L${l} · ${mod} · Parameter Importance Score=${v.toFixed(1)}/100 · raw |g×w|=${(spot.grid[l]?.[spot.modules.indexOf(mod)] ?? 0).toExponential(3)} · click → knob`} /></div>
            <div style={{ color: 'var(--text-1)', margin: '14px 0 8px' }}>layer {shownSpotLayer}{hoverLayer != null && hoverLayer !== layer ? <span style={hint}> · preview, pinned L{layer}</span> : <span style={hint}> · pinned</span>}<span style={hint}> · Parameter Importance Score 0–100 · raw max {spotScore.max.toExponential(3)} · {spot.modules.length} modules</span></div>
            <BarList items={spot.modules.map((m, c) => ({ label: moduleAbbrev(m), value: spotScore.grid[shownSpotLayer]?.[c] ?? 0, title: `${m} · raw |g×w|=${(spot.grid[shownSpotLayer]?.[c] ?? 0).toExponential(3)}` })).sort((a, b) => b.value - a.value)} color={ampColor} valueLabel={(v) => v.toFixed(1)} />
            <div style={{ ...hint, fontSize: 10, marginTop: 8 }}>cols: q k v o = attn · gate up down = mlp · ln1 ln2 = layernorms · +b = bias · left = layer index</div>
          </VizCard>
          <SpotSummary grid={spotScore.grid} modules={spot.modules} />
          {(() => {
            const conc = d.concentration
            const topPct = (frac: number) => { if (!conc) return null; let best = conc[0]; for (const p of conc) { if (p[0] <= frac) best = p; else break } return best[1] }
            const contrastTopk = d.knobs.find((k) => k.kind === 'spot')?.topk ?? spotTopk
            const runContrast = () => { patch(mid, (dd) => ({ ...dd, contrast: { topk: contrastTopk }, contrastProg: 'clean' })); sendTo(mid, { type: 'causal_contrast', topk: contrastTopk, code_examples: examples, general_examples: GENERAL_SET.split('\n').filter(Boolean) }) }
            return (
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 12 }}>
                <div style={{ flex: '1 1 320px', minWidth: 300 }}>
                  <VizCard title="Concentration" accent="var(--accent)"
                    subtitle="Cumulative importance vs fraction of weights (sorted by importance). The further the curve bows above the dashed equality line, the more the coding signal concentrates in a few weights.">
                    {conc ? (<>
                      <LineChart points={conc} />
                      <div style={{ ...hint, fontSize: 11, marginTop: 8 }}>
                        top 1% of weights → <span style={{ color: 'var(--accent)' }}>{((topPct(0.01) ?? 0) * 100).toFixed(1)}%</span> of importance ·
                        top 5% → <span style={{ color: 'var(--accent)' }}>{((topPct(0.05) ?? 0) * 100).toFixed(1)}%</span>
                      </div>
                    </>) : <span style={hint}>computing concentration…</span>}
                  </VizCard>
                </div>
                <div style={{ flex: '1 1 380px', minWidth: 320 }}>
                  <VizCard title="Causal contrast" accent="var(--danger)"
                    right={<Btn onClick={runContrast} disabled={pending.has(`causal_contrast:${mid}`)} color="var(--accent)" style={{ padding: '3px 10px' }}>{pending.has(`causal_contrast:${mid}`) ? '⟳ running' : 'Run'}</Btn>}
                    subtitle="Zero the spot vs matched random / bottom controls (same size), measure code + general PPL each. The proof: spot damage collapses code PPL while controls and general text barely move.">
                    {d.contrast ? <ContrastBars contrast={d.contrast} /> : <span style={hint}>Run to damage spot vs controls and compare PPL{gpus && gpus.count ? '' : ''}. Uses top-{(contrastTopk * 100).toFixed(1)}% (from the spot knob if set).</span>}
                    {d.contrastProg && <div style={{ ...hint, fontSize: 11, marginTop: 8 }}>measuring {d.contrastProg}…</div>}
                  </VizCard>
                </div>
              </div>
            )
          })()}
          {(() => {
            const parseKs = () => sweepKs.split(',').map((s) => parseFloat(s.trim())).filter((v) => !isNaN(v) && v > 0).map((v) => v / 100)
            const runSweep = () => {
              const topks = parseKs()
              if (!topks.length) { toast('[sweep] enter comma-separated % values, e.g. 0.0025, 0.01, 0.09, 0.25'); return }
              patch(mid, (dd) => ({ ...dd, sweep: { topks, rows: [] }, sweepProg: 'clean' }))
              sendTo(mid, { type: 'causal_sweep', topks, code_examples: examples, general_examples: GENERAL_SET.split('\n').filter(Boolean) })
            }
            const sw = d.sweep
            const cleanCode = sw?.clean?.code_ppl
            const fmt = (v: number | null | undefined) => v == null ? '—' : v >= 1e4 ? v.toExponential(1) : v.toFixed(1)
            const cell = (k: number, cond: string) => sw?.rows.find((r) => r.topk === k && r.cond === cond)
            const codeColor = (v: number | undefined) => (v != null && cleanCode != null && v > cleanCode * 3) ? 'var(--danger)' : 'var(--text-1)'
            const running = pending.has(`causal_sweep:${mid}`)
            const ks = sw?.topks?.length ? sw.topks : parseKs()
            const kPct = (k: number) => (k * 100).toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
            return (
              <div style={{ marginTop: 14 }}>
                <VizCard title="Control sweep · paper Table 1" accent="var(--danger)"
                  right={<span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    <input value={sweepKs} onChange={(e) => setSweepKs(e.target.value)} title="comma-separated top-k% values to sweep (e.g. the paper's 0.0025, 0.01, 0.09, 0.25)"
                      style={{ width: 190, fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 8, padding: '2px 6px' }} />
                    <span style={hint}>%</span>
                    <Btn onClick={runSweep} disabled={running} color="var(--accent)" style={{ padding: '3px 10px' }}>{running ? '⟳ running' : 'Run sweep'}</Btn>
                  </span>}
                  subtitle="Zero the spot vs matched random / bottom controls at each top-k%, measuring code + general PPL. Reproduces the paper's Table 1: only spot damage collapses code PPL; equal-size random / bottom controls (and general text) barely move.">
                  {!sw ? <span style={hint}>compute a spot first, then Run. Default k = the paper's 0.0025 – 0.25%.</span>
                    : <>
                      <div style={{ marginBottom: 8, fontSize: 12 }}><span style={hint}>clean (no damage) · </span><span style={{ fontWeight: 600 }}>code {fmt(sw.clean?.code_ppl)}</span><span style={hint}> · general {fmt(sw.clean?.general_ppl)}</span></div>
                      <div style={{ overflowX: 'auto' }}>
                        <div style={{ display: 'grid', gridTemplateColumns: '84px repeat(3, 1fr)', gap: 1, minWidth: 440, fontSize: 11 }}>
                          <div style={{ ...hint, padding: '3px 6px' }}>top-k%</div>
                          {['spot', 'random', 'bottom'].map((c) => <div key={c} style={{ padding: '3px 6px', fontWeight: 600, color: c === 'spot' ? 'var(--danger)' : 'var(--text-1)', background: 'var(--bg-2)' }}>{c}</div>)}
                          {ks.map((k) => <Fragment key={k}>
                            <div className="mono" style={{ padding: '3px 6px', color: 'var(--text-1)' }}>{kPct(k)}%</div>
                            {['spot', 'random', 'bottom'].map((c) => { const r = cell(k, c); return (
                              <div key={c} style={{ padding: '3px 6px', background: 'var(--bg-1)', fontVariantNumeric: 'tabular-nums' }}
                                title={r ? `code ${r.code_ppl} · general ${r.general_ppl ?? '—'}` : undefined}>
                                {r ? <><span style={{ color: codeColor(r.code_ppl), fontWeight: 600 }}>{fmt(r.code_ppl)}</span><span style={hint}> / {fmt(r.general_ppl)}</span></>
                                  : <span style={hint}>{running ? '…' : '—'}</span>}
                              </div>) })}
                          </Fragment>)}
                        </div>
                      </div>
                      <div style={{ ...hint, fontSize: 10, marginTop: 6 }}>each cell = code PPL / general PPL · red = code collapse (&gt;3× clean) · spot should collapse, random/bottom shouldn't</div>
                    </>}
                  {d.sweepProg && <div style={{ ...hint, fontSize: 11, marginTop: 8 }}>measuring {d.sweepProg}…</div>}
                </VizCard>
              </div>
            )
          })()}
          <div style={{ marginTop: 14, borderTop: '1px solid var(--line)', paddingTop: 12 }}>
            <div style={{ color: 'var(--text-1)', marginBottom: 6, display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
              knob board<span style={hint}> · {d.knobs.length} active · reversible</span>
              <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
                <span style={hint}>eval on</span>
                <select value={evalDsName} onChange={(e) => setEvalDsName(e.target.value)} title="dataset used to measure baseline/combined PPL (kppl) — separate from the spot data above" style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '2px 4px' }}>
                  <option value="">spot data ({examples.length})</option>
                  {datasets.map((x) => <option key={x.name} value={x.name}>{x.name} ({dsMeta[x.name]?.count ?? '…'})</option>)}
                </select>
              </span>
            </div>
            <div style={{ marginBottom: 6, display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
              {!selected.has('spot') && <Btn onClick={() => addKnob({ key: 'spot', kind: 'spot', topk: spotTopk, op: 'scale', alpha: 0 })} style={{ padding: '2px 10px' }}>+ Top-k% spot</Btn>}
              {d.knobs.length === 0 && <span style={hint}>or click a spot cell above ↑</span>}
              {locateProg[`${mid}:intervene`] && <span style={{ ...hint, fontSize: 11 }}>locating… {locateProg[`${mid}:intervene`].i}/{locateProg[`${mid}:intervene`].total}</span>}
              {d.regions.length > 0 && (() => {
                const cap = d.regions.find((r) => r.name === namedRegionPick)?.base_topk  // saved % = adjustable upper bound
                const capPct = cap != null ? cap * 100 : 100
                return (<>
                  <select value={namedRegionPick} onChange={(e) => setNamedRegionPick(e.target.value)} title="apply a saved region as a knob, re-thresholded to the % below" style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '2px 4px' }}>
                    <option value="">◈ from saved region…</option>
                    {d.regions.map((r) => <option key={r.name} value={r.name}>{r.name}</option>)}
                  </select>
                  {namedRegionPick && <>
                    <input type="number" min={0.001} max={capPct} step={0.005} value={+(namedRegionTopk * 100).toFixed(4)}
                      onChange={(e) => setNamedRegionTopk(Math.min(capPct, Math.max(0.001, Number(e.target.value) || 0.001)) / 100)}
                      title={cap != null ? `top-k% of the saved region · adjustable down from the saved ${capPct.toFixed(3)}%` : 'top-k% of the saved region'}
                      style={{ width: 56, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 8, fontSize: 11, padding: '0 2px' }} />
                    <span style={hint}>%{cap != null ? ` (≤ saved ${capPct.toFixed(3)}%)` : ''}</span>
                    <Btn onClick={() => { const name = namedRegionPick; addKnob({ key: `named:${name}`, kind: 'named', name, topk: namedRegionTopk, op: 'scale', alpha: 0 }); setNamedRegionPick('') }}
                      disabled={selected.has(`named:${namedRegionPick}`)} style={{ padding: '2px 10px' }}>+ Add</Btn>
                  </>}
                </>)
              })()}
            </div>
            {d.knobs.map((k) => (
              <div key={k.key} style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4 }}>
                {k.kind === 'spot'
                  ? <span style={{ width: 108, display: 'flex', alignItems: 'center', gap: 2, color: 'var(--accent)', fontSize: 11 }}>spot top<input type="number" min={0.001} max={100} step={0.005} value={+((k.topk ?? spotTopk) * 100).toFixed(4)} onChange={(e) => adjust(k.key, { topk: Math.min(100, Math.max(0.001, Number(e.target.value) || 0.001)) / 100 })} style={{ width: 52, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 8, fontSize: 11, padding: '0 2px' }} />%</span>
                  : k.kind === 'named' ? (() => {
                      const cap = d.regions.find((r) => r.name === k.name)?.base_topk
                      const capPct = cap != null ? cap * 100 : 100
                      return (
                        <span style={{ width: 148, display: 'flex', alignItems: 'center', gap: 2, color: 'var(--accent)', fontSize: 11 }} title={cap != null ? `adjustable down from the saved ${capPct.toFixed(3)}%` : k.name}>
                          <span className="mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 70 }}>◈ {k.name}</span>
                          <input type="number" min={0.001} max={capPct} step={0.005} value={+((k.topk ?? capPct / 100) * 100).toFixed(4)} onChange={(e) => adjust(k.key, { topk: Math.min(capPct, Math.max(0.001, Number(e.target.value) || 0.001)) / 100 })} style={{ width: 48, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 8, fontSize: 11, padding: '0 2px' }} />%
                        </span>
                      )
                    })()
                  : <span className="mono" style={{ width: 96, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-1)', fontSize: 11 }} title={`L${k.layer} · ${k.module}`}>L{k.layer}·{(k.module ?? '').replace('.weight', '').replace('_proj', '')}</span>}
                <select value={k.op} onChange={(e) => adjust(k.key, { op: e.target.value })} style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '1px 3px' }}>
                  {['scale', 'zero', 'mean', 'random'].map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
                {k.op === 'scale' && <>
                  <input type="range" min={0} max={2} step={0.05} value={Math.min(k.alpha, 2)} onChange={(e) => adjust(k.key, { alpha: Number(e.target.value) })} style={{ flex: 1, minWidth: 40 }} />
                  <input type="number" step={0.01} value={k.alpha} onChange={(e) => adjust(k.key, { alpha: Number(e.target.value) })} style={{ width: 52, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 8, fontSize: 11, padding: '1px 3px' }} title="precise α (can exceed 2 to amplify)" />
                </>}
                <button onClick={() => remove(k.key)} style={{ ...iconBtn, color: 'var(--danger)' }}>×</button>
              </div>
            ))}
            {d.knobs.length > 0 && <>
              <div style={{ display: 'flex', gap: 6, margin: '4px 0 8px' }}>
                <Btn onClick={clearAll} color="var(--text-2)" style={{ padding: '2px 10px' }}>Clear all</Btn>
                <Btn onClick={runAB} disabled={d.busy} color="var(--accent)" style={{ padding: '2px 10px' }} title="run the prompt twice: knobs off (baseline) then on (intervened)">A/B compare</Btn>
              </div>
              {base != null && (
                <div style={{ fontSize: 11 }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    <div><span style={hint}>baseline PPL</span><div className="mono" style={{ color: 'var(--text-0)' }}>{base.toPrecision(4)}</div></div>
                    <div><span style={hint}>combined PPL</span><div className="mono" style={{ color: inter != null && inter > base ? 'var(--danger)' : 'var(--text-0)' }}>{inter != null ? `${inter.toPrecision(4)}  (×${(inter / base).toPrecision(3)})` : '…'}</div></div>
                  </div>
                  <div style={{ ...hint, marginTop: 3 }}>measured on {evalDsName ? `${evalDsName} (${evalExamples.length})` : `spot data (${evalExamples.length})`}</div>
                </div>
              )}
              {(d.ab.base != null || d.ab.inter != null) && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8, fontSize: 11 }}>
                  <div><span style={hint}>baseline output</span><div className="mono" style={{ whiteSpace: 'pre-wrap', background: 'var(--bg-2)', borderRadius: 10, padding: 6, marginTop: 3, maxHeight: 180, overflow: 'auto' }}>{d.ab.base ?? <span style={hint}>generating…</span>}</div></div>
                  <div><span style={hint}>intervened output</span><div className="mono" style={{ whiteSpace: 'pre-wrap', background: 'var(--bg-2)', borderRadius: 10, padding: 6, marginTop: 3, maxHeight: 180, overflow: 'auto', border: '1px solid var(--line-strong)' }}>{d.ab.inter ?? (d.ab.base != null ? d.output || <span style={hint}>generating…</span> : <span style={hint}>…</span>)}</div></div>
                </div>
              )}
              <div style={hint}>re-run the prompt to see the combined output · A/B compare runs it twice</div>
            </>}
          </div>
          <div style={{ marginTop: 12, borderTop: '1px solid var(--line)', paddingTop: 10 }}>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 8 }}>
              {(() => {
                const rn = regionName[mid] ?? ''
                // save base defaults to top 1% (the paper's headline size), not the drifting global spotTopk;
                // an explicit spot-knob % still wins so a deliberate wider save is honored.
                const doSave = () => { const name = rn.trim(); if (!name) { toast('[save] enter a region name first'); return } sendTo(mid, { type: 'save_region', name, region: { kind: 'spot', examples, topk: d.knobs.find((k) => k.kind === 'spot')?.topk ?? 0.01 } }); setRegionName((s) => ({ ...s, [mid]: '' })) }
                return (<>
                  <input value={rn} onChange={(e) => setRegionName((s) => ({ ...s, [mid]: e.target.value }))} onKeyDown={(e) => { if (e.key === 'Enter') doSave() }} placeholder="region name"
                    style={{ width: 130, background: 'var(--bg-2)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '3px 8px', outline: 'none', fontSize: 11 }} />
                  <Btn onClick={doSave} disabled={pending.has(`save_region:${mid}`)} style={{ padding: '2px 10px' }} title="save the current top-k% spot mask to the workspace">{pending.has(`save_region:${mid}`) ? '⟳ saving…' : 'Save region'}</Btn>
                </>)
              })()}
              <Btn onClick={() => { sendTo(mid, { type: 'ppl', examples: PRESETS.python.split('\n').filter(Boolean), tag: 'code' }); sendTo(mid, { type: 'ppl', examples: GENERAL_SET.split('\n').filter(Boolean), tag: 'general' }) }} disabled={pending.has(`ppl:${mid}`)} color="var(--accent)" style={{ padding: '2px 10px' }} title="PPL on code vs general text — selective damage shows here">{pending.has(`ppl:${mid}`) ? '⟳ ' : ''}Eval code｜general</Btn>
            </div>
            <div style={{ ...hint, fontSize: 11, marginBottom: 4 }}>saved % is the upper bound — reload as a knob and dial it down anytime, never up</div>
            {locateProg[`${mid}:save_region`] && <div style={{ ...hint, fontSize: 11, marginTop: 4 }}>locating… {locateProg[`${mid}:save_region`].i}/{locateProg[`${mid}:save_region`].total}</div>}
            {(d.evals.code != null || d.evals.general != null) && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 11 }}>
                <div><span style={hint}>code PPL</span><div className="mono" style={{ color: 'var(--text-0)' }}>{d.evals.code?.toPrecision(4) ?? '…'}</div></div>
                <div><span style={hint}>general PPL</span><div className="mono" style={{ color: 'var(--text-0)' }}>{d.evals.general?.toPrecision(4) ?? '…'}</div></div>
              </div>
            )}
          </div>
        </>)
      })()}
    </>)
  }

  const closed = catalog.filter((c) => !open.some((o) => o.id === c.id))
  const multi = tileCount() > 1

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '9px 16px', borderBottom: '1px solid var(--line)', background: 'var(--bg-1)' }}>
        <button onClick={() => setExplorerOpen((v) => !v)} title="explorer" style={{ background: 'transparent', border: 'none', color: 'var(--text-1)', cursor: 'pointer', padding: 0, fontSize: 16, display: 'flex', alignItems: 'center' }}>≡</button>
        <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <img src="/logo.png" alt="" style={{ height: 18 }} onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none' }} />
          <strong style={{ fontSize: 14, fontWeight: 600, letterSpacing: '-0.01em' }}>Parametric Studio</strong>
        </span>
        <div style={{ display: 'flex', gap: 6 }}>
          {open.map((m) => {
            const loading = data[m.id]?.loading
            const dl = data[m.id]?.download
            const elapsed = loading && loadStart.current[m.id] ? Math.floor((Date.now() - loadStart.current[m.id]) / 1000) : 0
            const gb = (mb: number) => (mb / 1024).toFixed(1)
            const gpuIdx = gpus?.devices.find((d) => d.models.includes(m.id))?.index  // P16: which GPU this model landed on
            return (
              <span key={m.id} title={loading ? `downloading/loading… ${elapsed}s` : undefined} style={{ position: 'relative', overflow: 'hidden', fontSize: 12, fontWeight: 500, padding: '3px 10px', borderRadius: 999, background: m.id === focused() ? 'var(--accent-soft)' : 'var(--bg-2)', border: `1px solid ${m.id === focused() ? 'transparent' : 'var(--line)'}`, color: m.id === focused() ? 'var(--accent)' : 'var(--text-1)' }}>
                {loading ? <span style={{ color: 'var(--accent)' }}>⟳ </span> : data[m.id]?.busy ? <span style={{ color: 'var(--live)' }}>● </span> : ''}
                <span>{m.label}</span>
                {gpuIdx != null && <span className="mono" style={{ ...hint, marginLeft: 5, fontSize: 10 }}>GPU{gpuIdx}</span>}
                {dl ? <span className="mono" style={hint}> {Math.round(dl.pct)}% · {gb(dl.done_mb)}/{gb(dl.total_mb)}GB</span> : loading && <span className="mono" style={hint}> loading… {elapsed}s</span>}
                <span onClick={() => closeModel(m.id)} title="unload model" style={{ marginLeft: 6, cursor: 'pointer', color: 'var(--text-2)' }}>×</span>
                {loading && (dl
                  ? <span style={{ position: 'absolute', bottom: 0, left: 0, height: 2, width: `${Math.max(0, Math.min(100, dl.pct))}%`, background: 'var(--accent)' }} />
                  : <span style={{ position: 'absolute', bottom: 0, left: 0, height: 2, width: '40%', background: 'var(--accent)', animation: 'loading-slide 1.2s linear infinite' }} />)}
              </span>
            )
          })}
          <select value="" onChange={(e) => {
            if (e.target.value === '__hf') { setAsking('hf'); setAskValue(''); return }  // inline input below
            const c = catalog.find((x) => x.id === e.target.value)
            if (c) openModel(c.id, c.label, openDevice || undefined)
          }} onFocus={() => sendTo(focused(), { type: 'gpus' })} style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '2px 4px' }}>
            <option value="">+ model</option>
            {closed.map((c) => {
              const gb = c.size_mb != null ? (c.size_mb / 1024).toFixed(1) : null
              const label = c.installed ? `✓ ${c.label}` : gb ? `${c.label} (${gb}GB ⬇)` : c.label
              return <option key={c.id} value={c.id}>{label}</option>
            })}
            <option value="__hf">custom (HF id)…</option>
          </select>
          {gpus && gpus.count > 1 && (
            <select value={openDevice} onChange={(e) => setOpenDevice(e.target.value)} title="GPU to load the next model onto"
              style={{ fontSize: 11, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '2px 4px' }}>
              <option value="">Auto</option>
              {gpus.devices.map((d) => <option key={d.index} value={`cuda:${d.index}`}>GPU {d.index}</option>)}
            </select>
          )}
          {asking === 'hf' && (
            <input autoFocus value={askValue} onChange={(e) => setAskValue(e.target.value)} placeholder="org/model-id (Llama/Qwen-style) · Enter"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && askValue.trim()) { const id = askValue.trim(); openModel(id, id.split('/').pop() ?? id, openDevice || undefined); setAsking(null) }
                if (e.key === 'Escape') setAsking(null)
              }} onBlur={() => setAsking(null)}
              style={{ fontSize: 11, width: 240, background: 'var(--bg-2)', color: 'var(--text-0)', border: '1px solid var(--accent)', borderRadius: 10, padding: '2px 6px', outline: 'none' }} />
          )}
        </div>
        <Btn onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))} title="toggle light/dark" color="var(--text-2)" style={{ marginLeft: 'auto', padding: '2px 8px' }}>{theme === 'dark' ? '☾' : '☀'}</Btn>
        <Btn onClick={() => setSync((v) => !v)} title="broadcast input to all open models" color={sync ? 'var(--accent)' : 'var(--text-2)'} style={{ background: sync ? 'var(--bg-2)' : 'transparent', padding: '2px 8px' }}>{sync ? '●' : '○'} sync</Btn>
        <span style={{ color: anyBusy ? 'var(--live)' : 'var(--text-2)', fontSize: 11 }}>{anyBusy ? '● live' : 'idle'}</span>
      </div>

      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        {explorerOpen && (() => {
          const fd = data[focused()]
          const fmtDtype = (d: string) => d.replace('bfloat16', 'bf16').replace('float32', 'f32').replace('float16', 'f16')
          // VS Code-style folder tree: every dotted name segment is a folder, params are files.
          type TNode = { children: Map<string, TNode>; leaf?: { shape: number[]; dtype: string }; leaves: number }
          const buildTree = (mid: string): TNode | null => {
            const ts = data[mid]?.tensors
            if (!ts) return null
            const root: TNode = { children: new Map(), leaves: 0 }
            for (const t of ts) {
              let node = root
              const segs = t.name.split('.')
              segs.forEach((seg, i) => {
                node.leaves++
                if (!node.children.has(seg)) node.children.set(seg, { children: new Map(), leaves: 0 })
                node = node.children.get(seg)!
                if (i === segs.length - 1) { node.leaf = { shape: t.shape, dtype: t.dtype }; node.leaves++ }
              })
            }
            return root
          }
          const toggle = (p: string) => setExpPaths((s) => { const n = new Set(s); n.has(p) ? n.delete(p) : n.add(p); return n })
          // indent guides: one 1px vertical rule per level, then the row content
          const IndentGuides = ({ depth }: { depth: number }) => (<>{Array.from({ length: depth }, (_, i) => <span key={i} className="tree-indent" style={{ width: 10, flexShrink: 0, alignSelf: 'stretch' }} />)}</>)
          const renderTree = (node: TNode, path: string, depth: number): React.ReactNode =>
            [...node.children.entries()].map(([seg, child]) => {
              const p = `${path}.${seg}`
              if (child.leaf) return (
                <div key={p} className="tree-row mono" title={`${p.split('#.')[1] ?? p} · [${child.leaf.shape.join('×')}] ${child.leaf.dtype}`} style={{ fontSize: 10, height: 20 }}>
                  <IndentGuides depth={depth} />
                  <span style={{ width: 12, flexShrink: 0 }} />
                  <Icon name="tensor" size={12} />
                  <span className="tree-label" style={{ color: 'var(--text-1)' }}>{seg}</span>
                  <span style={{ ...hint, flexShrink: 0 }}>[{child.leaf.shape.join('×')}] {fmtDtype(child.leaf.dtype)}</span>
                </div>
              )
              return (
                <Fragment key={p}>
                  <div onClick={() => toggle(p)} className="tree-row mono" style={{ fontSize: 10, height: 20 }}>
                    <IndentGuides depth={depth} />
                    <Chevron open={expPaths.has(p)} />
                    <Icon name="folder" size={12} />
                    <span className="tree-label" style={{ color: 'var(--text-1)' }}>{seg}</span>
                    <span style={{ ...hint, flexShrink: 0 }}>({child.leaves})</span>
                  </div>
                  {expPaths.has(p) && renderTree(child, p, depth + 1)}
                </Fragment>
              )
            })
          const toggleModel = (mid: string) => {
            setExpModels((s) => { const n = new Set(s); n.has(mid) ? n.delete(mid) : n.add(mid); return n })
            if (!expModels.has(mid) && data[mid]?.tensors == null) sendTo(mid, { type: 'tensors' })  // lazy fetch per model
          }
          // dataset context actions — reuse the editor-tab code paths (sample→spot, →train data)
          const useForSpot = (name: string) => computeSpotFor(focused(), name)  // pick → sample → compute (fetches if needed)
          const useAsTrainData = (name: string) => {  // select this dataset as the Train-tab benchmark (fetch content if needed)
            const dset = datasets.find((x) => x.name === name); if (!dset) return
            setTrainDsName(name)
            if (dset.content == null) sendTo(focused(), { type: 'read_dataset', name })
          }
          // open a context menu at the event position, targeting `target` (items rebuilt each render)
          const openMenu = (e: React.MouseEvent, target: string) => { e.preventDefault(); e.stopPropagation(); setMenu({ x: e.clientX, y: e.clientY, target }) }
          // delete item: reuses the armed confirm. session presets delete outright (menu closes); server
          // datasets/regions arm on the 1st click (menu stays, label flips) and execute on the 2nd.
          const dsMenu = (dset: typeof datasets[number]): MenuItem[] => {
            const isArmed = armed === `data:${dset.name}`
            const label = !dset.server ? 'Remove from list' : dset.link ? (isArmed ? 'Unlink? — click again' : 'Unlink') : (isArmed ? 'Delete? — click again' : 'Delete')
            return [
              { label: 'Open in editor', onClick: () => { setMenu(null); openDataTab(dset.name) } },
              { label: 'Use for spot', onClick: () => { setMenu(null); useForSpot(dset.name) } },
              { label: 'Use as train data', onClick: () => { setMenu(null); useAsTrainData(dset.name) } },
              'sep',
              { label, danger: true, key: 'Del', onClick: () => { const willExecute = !dset.server || isArmed; deleteDataset(dset); if (willExecute) setMenu(null) } },
            ]
          }
          const regionMenu = (name: string): MenuItem[] => {
            const isArmed = armed === `region:${name}`
            return [
              { label: 'View', onClick: () => { setMenu(null); setSelected(`region:${name}`); openRegionTab(name) } },
              { label: 'Compare…', onClick: () => { setMenu(null); openTabId('compare') } },
              'sep',
              { label: isArmed ? 'Delete? — click again' : 'Delete', danger: true, key: 'Del', onClick: () => { deleteRegion(name); if (isArmed) setMenu(null) } },
            ]
          }
          const modelMenu = (mid: string): MenuItem[] => [
            { label: 'Unload', danger: true, key: 'Del', onClick: () => { setMenu(null); closeModel(mid); if (selected === `model:${mid}`) setSelected(null) } },
          ]
          // items for the currently open menu, keyed off its target (kind:name)
          const menuItems = (): MenuItem[] => {
            if (!menu) return []
            const [kind, ...rest] = menu.target.split(':'); const name = rest.join(':')
            if (kind === 'data') { const dset = datasets.find((x) => x.name === name); return dset ? dsMenu(dset) : [] }
            if (kind === 'region') return regionMenu(name)
            if (kind === 'model') return modelMenu(name)
            return []
          }
          const addFiles = (files: FileList | null) => {
            for (const f of Array.from(files ?? [])) {
              const name = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name
              setUploading((u) => u.includes(name) ? u : [...u, name])  // show 'uploading…' until dataset_saved/error clears it
              const reader = new FileReader()
              reader.onload = () => sendTo(focused(), { type: 'save_dataset', name, content: String(reader.result) })  // uploads persist to the kernel store
              reader.onerror = () => { setUploading((u) => u.filter((x) => x !== name)); toast(`[upload] ${name} — read failed`) }
              reader.readAsText(f)
            }
          }
          const addBtn = { ...hint, fontSize: 11, cursor: 'pointer', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '1px 8px', background: 'transparent' } as const
          return (
          <div style={{ width: explorerW, flexShrink: 0, padding: '14px 12px', overflow: 'auto', background: 'var(--bg-1)' }}>
            <div className="section-h" style={{ marginBottom: 10 }}>Models</div>
            {open.map((m) => {
              const tree = expModels.has(m.id) ? buildTree(m.id) : null
              const isSel = selected === `model:${m.id}`
              return (
                <Fragment key={m.id}>
                  <div className={`tree-row${isSel ? ' sel' : ''}`} onClick={() => { setSelected(`model:${m.id}`); setFocusModel(m.id) }} onContextMenu={(e) => { setSelected(`model:${m.id}`); openMenu(e, `model:${m.id}`) }} title={m.id}>
                    <span onClick={(e) => { e.stopPropagation(); toggleModel(m.id) }}><Chevron open={expModels.has(m.id)} /></span>
                    <Icon name="cube" />
                    <span className="tree-label" style={{ color: m.id === focused() && !isSel ? 'var(--accent)' : 'var(--text-1)' }}>{m.label}</span>
                    <span className="tree-x" title="unload model" onClick={(e) => { e.stopPropagation(); closeModel(m.id); if (isSel) setSelected(null) }}>×</span>
                  </div>
                  {expModels.has(m.id) && (tree == null ? <div style={{ ...hint, fontSize: 10, paddingLeft: 22 }}>loading…</div> : renderTree(tree, `${m.id}#`, 1))}
                </Fragment>
              )
            })}

            <div className="section-h" style={{ margin: '20px 0 8px' }}>Data</div>
            {datasets.map((dset) => {
              const isSel = selected === `data:${dset.name}`
              const isArmed = armed === `data:${dset.name}`
              return (
              <div key={dset.name} className={`tree-row${isSel ? ' sel' : ''}`} onClick={() => { setSelected(`data:${dset.name}`); openDataTab(dset.name) }} onContextMenu={(e) => { setSelected(`data:${dset.name}`); openMenu(e, `data:${dset.name}`) }}
                title={dset.link ? `linked from outside the store (${dset.link}) · open in an editor tab` : dset.server ? 'on disk · open in an editor tab' : 'session-only · open in an editor tab'} style={{ fontSize: 11 }}>
                <Icon name={dset.link ? 'link' : dset.server ? 'database' : 'file'} />
                <span className="tree-label" style={{ color: 'var(--text-1)' }}>{dset.name} <span style={hint}>({dsMeta[dset.name]?.count ?? `${((dset.size ?? 0) / 1024).toFixed(1)}k`})</span></span>
                <span className={`tree-x${isArmed ? ' armed' : ''}`} onClick={(e) => { e.stopPropagation(); deleteDataset(dset) }}
                  title={dset.link ? `unlink '${dset.link}' — originals kept · click twice` : dset.server ? 'delete from store (cannot be undone) · click twice' : 'remove from list'}>{isArmed ? (dset.link ? 'unlink?' : 'sure?') : '×'}</span>
              </div>
              )
            })}
            <div style={{ display: 'flex', gap: 5, marginTop: 4, flexWrap: 'wrap' }}>
              <label style={addBtn}>+ File<input type="file" multiple style={{ display: 'none' }} onChange={(e) => { addFiles(e.target.files); e.target.value = '' }} /></label>
              <label style={addBtn}>+ Folder<input type="file" multiple style={{ display: 'none' }} {...({ webkitdirectory: '', directory: '' } as object)} onChange={(e) => { addFiles(e.target.files); e.target.value = '' }} /></label>
              <button onClick={() => { setAsking('path'); setAskValue('') }} title="symlink an external path into ~/.parametic_studio/datasets" style={addBtn}>+ Path</button>
              <button onClick={() => { setAsking('editor'); setAskValue('') }} title="save the spot-view editor content to the dataset store" style={addBtn}>+ Editor</button>
              <button onClick={() => { setAsking('hf-dataset'); setAskValue(''); setAskSplit(''); setAskConfig(''); setAskFilter('') }} title="load a dataset from the Hugging Face Hub by repo id" style={addBtn}>+ HF</button>
            </div>
            {(asking === 'path' || asking === 'editor') && (
              <input autoFocus value={askValue} onChange={(e) => setAskValue(e.target.value)}
                placeholder={asking === 'path' ? '~/data/my.jsonl or folder · Enter' : 'dataset name · Enter'}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && askValue.trim()) {
                    if (asking === 'path') sendTo(focused(), { type: 'link_path', path: askValue.trim() })
                    else if (ds.trim()) sendTo(focused(), { type: 'save_dataset', name: askValue.trim(), content: ds })
                    setAsking(null)
                  }
                  if (e.key === 'Escape') setAsking(null)
                }} onBlur={() => setAsking(null)}
                style={{ fontSize: 11, width: '100%', marginTop: 4, background: 'var(--bg-2)', color: 'var(--text-0)', border: '1px solid var(--accent)', borderRadius: 10, padding: '2px 6px', outline: 'none' }} />
            )}
            {asking === 'hf-dataset' && (() => {
              const submitHfDataset = () => {
                if (!askValue.trim()) return
                const [filterCol, ...rest] = askFilter.split('=')
                const filterVal = rest.join('=').trim()
                sendTo(focused(), {
                  type: 'load_hf_dataset', repo: askValue.trim(),
                  ...(askSplit.trim() ? { split: askSplit.trim() } : {}),
                  ...(askConfig.trim() ? { config: askConfig.trim() } : {}),
                  ...(filterCol.trim() && filterVal ? { filter_column: filterCol.trim(), filter_value: filterVal } : {}),
                })
                setAsking(null)
              }
              const onKey = (e: React.KeyboardEvent) => {
                if (e.key === 'Enter') submitHfDataset()
                if (e.key === 'Escape') setAsking(null)
              }
              const onBlur = (e: React.FocusEvent) => { if (!e.relatedTarget) setAsking(null) }
              const fieldStyle = { fontSize: 11, minWidth: 0, background: 'var(--bg-2)', color: 'var(--text-0)', border: '1px solid var(--accent)', borderRadius: 10, padding: '2px 6px', outline: 'none' } as const
              return (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
                  <input autoFocus value={askValue} onChange={(e) => setAskValue(e.target.value)} placeholder="openai/gsm8k · Enter"
                    onKeyDown={onKey} onBlur={onBlur} style={{ ...fieldStyle, flex: 2 }} />
                  <input value={askSplit} onChange={(e) => setAskSplit(e.target.value)} placeholder="train"
                    onKeyDown={onKey} onBlur={onBlur} style={{ ...fieldStyle, flex: 1 }} />
                  <input value={askConfig} onChange={(e) => setAskConfig(e.target.value)} placeholder="python  (humanevalpack)"
                    onKeyDown={onKey} onBlur={onBlur} style={{ ...fieldStyle, flex: 1 }} title="dataset config (e.g. humanevalpack language)" />
                  <input value={askFilter} onChange={(e) => setAskFilter(e.target.value)} placeholder="programming_language=Python  (tiny-codes)"
                    onKeyDown={onKey} onBlur={onBlur} style={{ ...fieldStyle, flex: 2 }} title="row filter: col=value, applied before capping rows" />
                  <button onMouseDown={(e) => e.preventDefault()} onClick={submitHfDataset} disabled={!askValue.trim()}
                    title="download this dataset from the Hugging Face Hub"
                    style={{ ...addBtn, flexShrink: 0, borderColor: 'var(--accent)', color: 'var(--accent)', opacity: askValue.trim() ? 1 : 0.5 }}>Load</button>
                </div>
              )
            })()}
            {hfLoading && <div style={{ ...hint, fontSize: 11, marginTop: 4 }}>loading {hfLoading}…</div>}
            {uploading.length > 0 && <div style={{ ...hint, fontSize: 11, marginTop: 4 }}>⟳ uploading {uploading.length === 1 ? uploading[0] : `${uploading.length} files`}…</div>}

            <div style={{ display: 'flex', alignItems: 'center', margin: '20px 0 8px' }}>
              <span className="section-h">Regions</span>
              {(fd?.regions ?? []).length >= 2 && <button onClick={() => openTabId('compare')} title="compare saved regions (per-dataset spots)" style={{ ...iconBtn, marginLeft: 'auto', color: 'var(--accent)', padding: 0 }}>Compare</button>}
            </div>
            {(fd?.regions ?? []).length === 0 && <div style={{ ...hint, fontSize: 11 }}>save one in the spot view</div>}
            <button onClick={() => { setAsking('import-region'); setAskValue(''); setAskRegionPath('') }} disabled={pending.has(`import_region:${focused()}`)}
              title="import a research-pipeline mask directory (one bool .pt per parameter, filename = param name) as a region — no recompute" style={{ ...addBtn, marginBottom: 4 }}>{pending.has(`import_region:${focused()}`) ? '⟳ ' : ''}+ Import</button>
            {asking === 'import-region' && (() => {
              const submitImport = () => {
                if (!askValue.trim() || !askRegionPath.trim()) return
                sendTo(focused(), { type: 'import_region', name: askValue.trim(), path: askRegionPath.trim() })
                setAsking(null)
              }
              const onKey = (e: React.KeyboardEvent) => { if (e.key === 'Enter') submitImport(); if (e.key === 'Escape') setAsking(null) }
              const onBlur = (e: React.FocusEvent) => { if (!e.relatedTarget) setAsking(null) }
              const fieldStyle = { fontSize: 11, minWidth: 0, background: 'var(--bg-2)', color: 'var(--text-0)', border: '1px solid var(--accent)', borderRadius: 10, padding: '2px 6px', outline: 'none' } as const
              return (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 6 }}>
                  <input autoFocus value={askValue} onChange={(e) => setAskValue(e.target.value)} placeholder="region name (e.g. java)"
                    onKeyDown={onKey} onBlur={onBlur} style={{ ...fieldStyle, flex: 1 }} />
                  <input value={askRegionPath} onChange={(e) => setAskRegionPath(e.target.value)} placeholder="folder of <param>.pt masks"
                    onKeyDown={onKey} onBlur={onBlur} style={{ ...fieldStyle, flex: 2 }} />
                  <button onMouseDown={(e) => e.preventDefault()} onClick={submitImport} disabled={!askValue.trim() || !askRegionPath.trim()}
                    title="combine the folder's per-param masks into one region" style={{ ...addBtn, flexShrink: 0, borderColor: 'var(--accent)', color: 'var(--accent)', opacity: askValue.trim() && askRegionPath.trim() ? 1 : 0.5 }}>Import</button>
                </div>
              )
            })()}
            {(fd?.regions ?? []).map((r) => {
              const isSel = selected === `region:${r.name}`
              const isArmed = armed === `region:${r.name}`
              return (
              <div key={r.name} className={`tree-row${isSel ? ' sel' : ''}`} onClick={() => { setSelected(`region:${r.name}`); openRegionTab(r.name) }} onContextMenu={(e) => { setSelected(`region:${r.name}`); openMenu(e, `region:${r.name}`) }}
                title={`${r.count.toLocaleString()} weights · open viewer`} style={{ fontSize: 11 }}>
                <Icon name="diamond" />
                <span className="tree-label" style={{ color: 'var(--text-1)' }}>{r.name} <span style={hint}>({r.count.toLocaleString()})</span></span>
                <span className={`tree-x${isArmed ? ' armed' : ''}`} onClick={(e) => { e.stopPropagation(); deleteRegion(r.name) }} title="delete region from disk — click twice to confirm">{isArmed ? 'sure?' : '×'}</span>
              </div>
              )
            })}
            {menu && <ContextMenu x={menu.x} y={menu.y} items={menuItems()} onClose={() => setMenu(null)} />}
          </div>
          )
        })()}
        {explorerOpen && <div onMouseDown={(e) => dragSidebar(e, 'left')} style={{ width: 5, flexShrink: 0, cursor: 'col-resize', background: 'var(--line)' }} />}
        {open.length === 0 ? (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, color: 'var(--text-2)' }}>
          <div>No model loaded</div>
          {catalog.length > 0 ? <div style={hint}>use + Model above to load one</div> : <div style={hint}>kernel offline — start it on :8000</div>}
        </div>
        ) : (
        <div ref={rowRef} style={{ flex: 1, display: 'flex', minWidth: 0 }}>
        {cols.map((col, ci) => (
          <Fragment key={col.id}>
            <div style={{ flexGrow: maxTile != null ? (col.tiles.some((t) => t.id === maxTile) ? 1 : 0) : col.w, flexBasis: 0, minWidth: 0, display: maxTile != null && !col.tiles.some((t) => t.id === maxTile) ? 'none' : 'flex', flexDirection: 'column' }}>
              {col.tiles.map((tile, ti) => (
                <Fragment key={tile.id}>
                  <div onMouseDown={() => setFocusModel(tile.model)} onDragOver={(e) => { e.preventDefault(); setOverTile(tile.id) }} onDrop={() => moveTab(tile.id)}
                    style={{ position: 'relative', flexGrow: maxTile != null ? (tile.id === maxTile ? 1 : 0) : tile.h, flexBasis: 0, minHeight: 0, display: maxTile != null && tile.id !== maxTile ? 'none' : 'flex', flexDirection: 'column', borderTop: ti > 0 ? '1px solid var(--line)' : 'none' }}>
                    <div style={{ display: 'flex', alignItems: 'center', borderBottom: '1px solid var(--line)', background: 'var(--bg-1)', overflow: 'hidden' }}>
                      <select value={tile.model} onChange={(e) => setTileModel(tile.id, e.target.value)} style={{ fontSize: 11, fontWeight: 500, background: 'transparent', color: 'var(--accent)', border: 'none', borderRight: '1px solid var(--line)', padding: '6px 6px', maxWidth: 110 }}>
                        {open.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
                      </select>
                      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
                        {tile.tabs.map((v, idx) => (
                          <span key={v} draggable onDragStart={() => { drag.current = { tid: tile.id, idx }; setDragging(true) }} onDragEnd={() => { drag.current = null; setDragging(false); setOverTile(null) }}
                            onDragOver={(e) => { e.preventDefault(); e.stopPropagation() }} onDrop={(e) => { e.stopPropagation(); moveTab(tile.id, idx) }} onClick={() => setActive(tile.id, idx)}
                            style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '7px 12px', cursor: 'grab', fontSize: 12, fontWeight: idx === tile.active ? 600 : 500, whiteSpace: 'nowrap', color: idx === tile.active ? 'var(--text-0)' : 'var(--text-2)', background: 'transparent', boxShadow: idx === tile.active ? 'inset 0 -2px 0 var(--accent)' : 'none', transition: 'color var(--ease)' }}>
                            {v.startsWith('data:') ? `▤ ${v.slice(5)}` : v.startsWith('region:') ? `◈ ${v.slice(7)}` : viewLabel(v)}<span onClick={(e) => { e.stopPropagation(); closeTab(tile.id, idx) }} style={hint}>×</span>
                          </span>
                        ))}
                        {tile.tabs.filter((t) => (VIEWS as readonly string[]).includes(t)).length < VIEWS.length && (
                          <select value="" onChange={(e) => { if (e.target.value) addTab(tile.id, e.target.value as View) }} title="add view" style={{ ...iconBtn, appearance: 'none', background: 'transparent' }}>
                            <option value="">+</option>
                            {VIEWS.filter((v) => !tile.tabs.includes(v)).map((v) => <option key={v} value={v}>{viewLabel(v)}</option>)}
                          </select>
                        )}
                      </div>
                      <div style={{ display: 'flex', flexShrink: 0 }}>
                        <button onClick={() => setMaxTile((m) => m === tile.id ? null : tile.id)} title={maxTile === tile.id ? 'restore size' : 'maximize this pane'} style={{ ...iconBtn, color: maxTile === tile.id ? 'var(--accent)' : undefined }}>{maxTile === tile.id ? '⤡' : '⤢'}</button>
                        <button onClick={() => splitRight(tile.id)} title="split right" style={iconBtn}>⊟</button>
                        <button onClick={() => splitDown(tile.id)} title="split down" style={iconBtn}>⊞</button>
                        {multi && <button onClick={() => pruneClose(tile.id)} title="close pane" style={iconBtn}>×</button>}
                      </div>
                    </div>
                    <div style={{ flex: 1, overflow: 'auto', padding: '12px 14px' }}>{viewBody(tile.model, tile.tabs[tile.active])}</div>
                    {dragging && overTile === tile.id && <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,122,255,0.16)', border: '1px solid var(--accent)', pointerEvents: 'none' }} />}
                  </div>
                  {maxTile == null && ti < col.tiles.length - 1 && <div onMouseDown={(e) => resizeTiles(ci, ti, e)} style={{ height: 5, flexShrink: 0, cursor: 'row-resize', background: 'var(--line)' }} />}
                </Fragment>
              ))}
            </div>
            {maxTile == null && ci < cols.length - 1 && <div onMouseDown={(e) => resizeCols(ci, e)} style={{ width: 5, flexShrink: 0, cursor: 'col-resize', background: 'var(--line)' }} />}
          </Fragment>
        ))}
        </div>
        )}
        {/* chat dock — bound to the focused model. flush with the window edge (no floating/shadow). hidden while a pane is maximized. */}
        {maxTile == null && <div onMouseDown={(e) => dragSidebar(e, 'right')} style={{ width: 5, flexShrink: 0, cursor: 'col-resize', background: 'var(--line)' }} />}
        <div style={{ width: chatW, flexShrink: 0, display: maxTile != null ? 'none' : 'flex', flexDirection: 'column', background: 'var(--bg-1)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '10px 14px', color: 'var(--text-1)', fontWeight: 500, borderBottom: '1px solid var(--line)' }}>
            <span style={{ width: 20, height: 20, borderRadius: '50%', background: 'var(--accent-soft)', color: 'var(--accent)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, flexShrink: 0 }}>◆</span>
            <span style={{ fontSize: 13 }}>Chat{sync && <span style={{ ...hint, fontSize: 12 }}> · broadcast</span>}</span>
            <span className="mono" style={{ fontSize: 11, color: 'var(--text-1)', background: 'var(--bg-2)', border: '1px solid var(--line-strong)', borderRadius: 999, padding: '2px 8px' }}>{open.find((m) => m.id === focused())?.label ?? ''}</span>
            <button onClick={() => setGenOpen((v) => !v)} title="generation options" style={{ ...iconBtn, marginLeft: 'auto', color: genOpen ? 'var(--accent)' : 'var(--text-2)' }}>⚙</button>
          </div>
          {genOpen && (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', padding: '6px 12px', borderBottom: '1px solid var(--line)', fontSize: 11 }}>
              <span style={hint}>max_tokens</span>
              <input type="number" min={1} value={maxTokens} onChange={(e) => setMaxTokens(Math.max(1, Number(e.target.value)))} style={{ width: 60, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '1px 4px', fontSize: 11 }} />
              <span style={hint} title="0 = greedy (deterministic) · >0 = sampling">temp</span>
              <input type="number" min={0} step={0.1} value={temperature} onChange={(e) => setTemperature(Math.max(0, Number(e.target.value)))} style={{ width: 48, background: 'var(--bg-2)', color: 'var(--text-1)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '1px 4px', fontSize: 11 }} />
              <span style={hint} title="fewer probes = faster generation">probes</span>
              {(['attention', 'activation', 'logitlens', 'spot_activation'] as const).map((p) => (
                <span key={p} onClick={() => setProbesOn((o) => ({ ...o, [p]: !o[p] }))} className={`chip${probesOn[p] ? ' on' : ''}`}>{probesOn[p] ? '●' : '○'} {p === 'spot_activation' ? 'spot-act' : p.slice(0, 4)}</span>
              ))}
            </div>
          )}
          <div className="mono" style={{ flex: 1, overflow: 'auto', padding: '10px 12px', whiteSpace: 'pre-wrap' }}>
            {(data[focused()]?.output) || <span style={hint} className="mono">ask below</span>}
            {data[focused()]?.busy && <span style={{ color: 'var(--accent)' }}>▌</span>}
          </div>
          <div style={{ padding: 12, borderTop: '1px solid var(--line)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, background: 'var(--bg-2)', border: '1px solid var(--line-strong)', borderRadius: 16, padding: '6px 6px 6px 14px', transition: 'border-color var(--ease)' }}>
              <input value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && send()} placeholder={sync ? 'ask all open models…' : 'ask focused…'}
                style={{ flex: 1, background: 'transparent', border: 'none', outline: 'none', padding: '6px 0', fontSize: 13, color: 'var(--text-0)' }} />
              <button onClick={anyBusy ? stop : send} title={anyBusy ? 'stop generation' : 'send'}
                style={{ flexShrink: 0, width: 30, height: 30, borderRadius: '50%', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13, color: '#fff', background: anyBusy ? 'var(--danger)' : 'var(--accent)', transition: 'opacity var(--ease)' }}
                onMouseEnter={(e) => { e.currentTarget.style.opacity = '0.85' }} onMouseLeave={(e) => { e.currentTarget.style.opacity = '1' }}>{anyBusy ? '■' : '↑'}</button>
            </div>
          </div>
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 12px', borderTop: '1px solid var(--line)', background: 'var(--bg-1)', color: 'var(--text-2)', fontSize: 11 }}>
        <span>{kernelUp === false
          ? <span style={{ color: 'var(--danger)' }}>● kernel offline · reconnecting…</span>
          : kernelUp === null
            ? <span style={hint}>○ connecting to kernel :8000…</span>
            : <>kernel {isRemoteConnected() ? `${sshHost || 'remote'} (ssh)` : (() => { try { return new URL(WS_URL.replace(/^ws/, 'http')).host } catch { return WS_URL } })()} · mps · bf16 · {open.length} model{open.length > 1 ? 's' : ''}{kernelStats && ` · rss ${(kernelStats.rss_mb / 1024).toFixed(1)}G`}
              {gpus && gpus.count > 0 && (
                <span className="mono"> · {gpus.count}× {gpus.devices[0]?.name || 'GPU'}{gpus.count > 1 && ' · ' + gpus.devices.map((d) => `GPU${d.index} ${Math.round(d.mem_used_mb / 1024)}/${Math.round(d.mem_total_mb / 1024)}G`).join(' · ')}</span>
              )}</>}</span>
        <span style={{ display: 'flex', gap: 10 }}>
          {!inTauri() && <button onClick={() => setSettingsOpen(true)} style={{ ...iconBtn, padding: 0 }}>Settings</button>}
          <span>{sync ? 'sync: broadcast' : 'sync: focused'} · {tileCount()} pane{tileCount() > 1 ? 's' : ''}</span>
        </span>
      </div>
      {settingsOpen && (() => {
        const inp = { width: '100%', background: 'var(--bg-2)', color: 'var(--text-0)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: '3px 8px', outline: 'none', fontSize: 12 } as const
        const setC = (k: string, v: string) => setConfig((c) => ({ ...c, [k]: v }))
        const gb = (mb: number | null) => (mb != null ? (mb / 1024).toFixed(1) : '?')
        return (
          <div style={{ position: 'fixed', top: 44, right: 12, width: 360, maxHeight: 'calc(100vh - 60px)', overflow: 'auto', background: 'var(--bg-1)', border: '1px solid var(--line-strong)', borderRadius: 14, boxShadow: '0 8px 32px rgba(0,0,0,0.45)', padding: 14, zIndex: 50 }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
              <strong style={{ color: 'var(--text-0)' }}>Settings</strong>
              <button onClick={() => setSettingsOpen(false)} style={{ ...iconBtn, marginLeft: 'auto', fontSize: 14 }}>×</button>
            </div>

            <div className="section-h" style={{ marginBottom: 6 }}>Kernel connection</div>
            <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
              {(['local', 'remote'] as const).map((mode) => (
                <button key={mode} onClick={() => { if (mode === 'remote' && !inTauri()) return; setKernelMode(mode) }}
                  disabled={mode === 'remote' && !inTauri()}
                  title={mode === 'remote' && !inTauri() ? 'Remote (SSH) is only available in the desktop app' : undefined}
                  style={{ flex: 1, padding: '4px 8px', fontSize: 11, borderRadius: 10, cursor: mode === 'remote' && !inTauri() ? 'default' : 'pointer',
                    border: `1px solid ${kernelMode === mode ? 'var(--accent)' : 'var(--line-strong)'}`,
                    color: mode === 'remote' && !inTauri() ? 'var(--text-2)' : kernelMode === mode ? 'var(--accent)' : 'var(--text-1)',
                    background: 'var(--bg-2)', opacity: mode === 'remote' && !inTauri() ? 0.5 : 1 }}>
                  {mode === 'local' ? 'Local' : 'Remote (SSH)'}
                </button>
              ))}
            </div>
            {!inTauri() && kernelMode === 'remote' && (
              <div style={{ ...hint, fontSize: 11, marginBottom: 10 }}>Remote (SSH) is only available in the desktop app.</div>
            )}

            {kernelMode === 'local' ? (
              <>
                <div style={{ display: 'grid', gap: 6, marginBottom: 4 }}>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>url</span>
                    <input value={kernelUrlInput} onChange={(e) => setKernelUrlInput(e.target.value)} placeholder={DEFAULT_WS} spellCheck={false} style={inp} />
                  </label>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>token</span>
                    <input type="password" value={kernelTokenInput} onChange={(e) => setKernelTokenInput(e.target.value)} spellCheck={false} style={inp} />
                  </label>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '6px 0 4px' }}>
                  <Btn onClick={() => {
                    const url = kernelUrlInput.trim()
                    url && url !== DEFAULT_WS ? localStorage.setItem('ps_kernel_url', url) : localStorage.removeItem('ps_kernel_url')
                    const token = kernelTokenInput.trim()
                    token ? localStorage.setItem('ps_kernel_token', token) : localStorage.removeItem('ps_kernel_token')
                    window.location.reload()
                  }} color="var(--accent)">Connect</Btn>
                  <span style={{ ...hint, fontSize: 11 }}>reloads the app</span>
                </div>
                <div style={{ ...hint, fontSize: 11, marginBottom: 14 }}>remote kernel: regions/datasets are stored on that machine, not here — e.g. ws(s)://host:port/ws</div>
              </>
            ) : (
              <>
                <div style={{ display: 'grid', gap: 6, marginBottom: 4 }}>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>host</span>
                    <input value={sshHost} onChange={(e) => setSshHost(e.target.value)} placeholder="gpu.lab.edu or 1.2.3.4" spellCheck={false} disabled={isTunnelLive()} style={inp} />
                  </label>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>port</span>
                    <input value={sshPort} onChange={(e) => setSshPort(e.target.value)} placeholder="22" spellCheck={false} disabled={isTunnelLive()} style={inp} />
                  </label>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>username</span>
                    <input value={sshUser} onChange={(e) => setSshUser(e.target.value)} spellCheck={false} disabled={isTunnelLive()} style={inp} />
                  </label>
                  <div style={{ display: 'flex', gap: 6, margin: '2px 0' }}>
                    {(['password', 'key'] as const).map((auth) => (
                      <button key={auth} onClick={() => setSshAuth(auth)} disabled={isTunnelLive()}
                        style={{ flex: 1, padding: '4px 8px', fontSize: 11, borderRadius: 10, cursor: isTunnelLive() ? 'default' : 'pointer',
                          border: `1px solid ${sshAuth === auth ? 'var(--accent)' : 'var(--line-strong)'}`,
                          color: sshAuth === auth ? 'var(--accent)' : 'var(--text-1)',
                          background: 'var(--bg-2)', opacity: isTunnelLive() ? 0.5 : 1 }}>
                        {auth === 'password' ? 'Password' : 'Key (.pem)'}
                      </button>
                    ))}
                  </div>
                  {sshAuth === 'password' ? (
                    <label style={{ display: 'grid', gap: 2 }}>
                      <span style={{ ...hint, fontSize: 11 }}>password</span>
                      <input type="password" value={sshPassword} onChange={(e) => setSshPassword(e.target.value)} spellCheck={false} disabled={isTunnelLive()} style={inp} />
                    </label>
                  ) : (
                    <>
                      <label style={{ display: 'grid', gap: 2 }}>
                        <span style={{ ...hint, fontSize: 11 }}>key path</span>
                        <div style={{ display: 'flex', gap: 6 }}>
                          <input value={sshKeyPath} onChange={(e) => setSshKeyPath(e.target.value)} placeholder="~/.ssh/gpu.pem" spellCheck={false} disabled={isTunnelLive()} style={{ ...inp, flex: 1 }} />
                          <Btn onClick={browseSshKeyPath} disabled={isTunnelLive() || !inTauri()}
                            title={inTauri() ? undefined : 'file picker is only available in the desktop app'}
                            style={{ flexShrink: 0 }}>Browse…</Btn>
                        </div>
                      </label>
                      <label style={{ display: 'grid', gap: 2 }}>
                        <span style={{ ...hint, fontSize: 11 }}>key passphrase (optional)</span>
                        <input type="password" value={sshKeyPassphrase} onChange={(e) => setSshKeyPassphrase(e.target.value)} placeholder="passphrase (encrypted keys only)" spellCheck={false} disabled={isTunnelLive()} style={inp} />
                      </label>
                    </>
                  )}
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>remote repo dir</span>
                    <input value={sshRepoDir} onChange={(e) => setSshRepoDir(e.target.value)} placeholder="~/parametic-report" spellCheck={false} disabled={isTunnelLive()} style={inp} />
                  </label>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>python path (optional)</span>
                    <input value={sshPythonPath} onChange={(e) => setSshPythonPath(e.target.value)} placeholder="/opt/conda/bin/python (where torch lives)" spellCheck={false} disabled={isTunnelLive()} style={inp} />
                  </label>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>HF cache dir (optional)</span>
                    <input value={sshHfHome} onChange={(e) => setSshHfHome(e.target.value)} placeholder="/shared/you/hf_cache — roomy volume for big models" spellCheck={false} disabled={isTunnelLive()} style={inp} />
                  </label>
                  <label style={{ display: 'grid', gap: 2 }}>
                    <span style={{ ...hint, fontSize: 11 }}>model (optional)</span>
                    <input value={sshModel} onChange={(e) => setSshModel(e.target.value)} placeholder="Qwen/Qwen2.5-1.5B-Instruct" spellCheck={false} disabled={isTunnelLive()} style={inp} />
                  </label>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '6px 0 4px' }}>
                  {isTunnelLive() ? (
                    <Btn onClick={sshDisconnect} color="var(--danger)">Disconnect</Btn>
                  ) : (
                    <Btn onClick={sshConnect} disabled={sshConnecting} color="var(--accent)">{sshConnecting ? 'Connecting…' : 'Connect'}</Btn>
                  )}
                  {sshStatus && (
                    <span style={{ ...hint, fontSize: 11 }}>
                      {sshStatus.state === 'connecting' && 'authenticating…'}
                      {sshStatus.state === 'starting-kernel' && 'starting remote kernel…'}
                      {sshStatus.state === 'forwarding' && 'forwarding…'}
                      {sshStatus.state === 'connected' && 'connected'}
                      {sshStatus.state === 'disconnected' && 'disconnected'}
                      {sshStatus.state === 'error' && `error: ${sshStatus.detail ?? 'unknown'}`}
                    </span>
                  )}
                </div>
                <div style={{ ...hint, fontSize: 11, marginBottom: 14 }}>runs the studio kernel over SSH on a remote GPU box; the app tunnels to it at localhost:8422.</div>
              </>
            )}

            <div className="section-h" style={{ marginBottom: 6 }}>Kernel</div>
            <div style={{ display: 'grid', gap: 6, marginBottom: 4 }}>
              {(['python_path', 'kernel_dir', 'model'] as const).map((k) => (
                <label key={k} style={{ display: 'grid', gap: 2 }}>
                  <span style={{ ...hint, fontSize: 11 }}>{k}</span>
                  <input value={config[k] ?? ''} onChange={(e) => setC(k, e.target.value)} spellCheck={false} style={inp} />
                </label>
              ))}
              <label style={{ display: 'grid', gap: 2 }}>
                <span style={{ ...hint, fontSize: 11 }}>datasets_dir</span>
                <input value={config.datasets_dir ?? ''} onChange={(e) => setC('datasets_dir', e.target.value)} spellCheck={false}
                  placeholder="~/.parametic_studio/datasets (default)" style={inp} />
              </label>
              <label style={{ display: 'grid', gap: 2 }}>
                <span style={{ ...hint, fontSize: 11 }}>hf_token</span>
                <input type="password" value={config.hf_token ?? ''} onChange={(e) => setC('hf_token', e.target.value)} spellCheck={false}
                  autoComplete="off" placeholder="for gated datasets" style={inp} />
              </label>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '6px 0 14px' }}>
              <Btn onClick={() => sendTo(focused(), { type: 'set_config', config })} color="var(--accent)">Save</Btn>
              <span style={{ ...hint, fontSize: 11 }}>applies on next app launch</span>
            </div>

            <div className="section-h" style={{ marginBottom: 6 }}>Models · cache</div>
            {installed == null && <div style={{ ...hint, fontSize: 11, marginBottom: 10 }}>loading…</div>}
            {installed?.length === 0 && <div style={{ ...hint, fontSize: 11, marginBottom: 10 }}>no cached models</div>}
            {installed?.map((it) => (
              <div key={it.id} style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 11, marginBottom: 3 }}>
                <span className="mono" style={{ color: 'var(--text-1)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }} title={it.id}>{it.id}</span>
                <span className="mono" style={hint}>{gb(it.size_mb)}GB</span>
                <span onClick={() => confirmClick(`cache:${it.id}`, () => sendTo(focused(), { type: 'delete_cached', model: it.id }))}
                  title="delete from HF cache — click twice" style={{ cursor: 'pointer', color: armed === `cache:${it.id}` ? 'var(--danger)' : 'var(--text-2)', whiteSpace: 'nowrap' }}>{armed === `cache:${it.id}` ? 'sure?' : '×'}</span>
              </div>
            ))}

            <div className="section-h" style={{ margin: '14px 0 6px' }}>Appearance</div>
            <Btn onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}>{theme === 'dark' ? '☾' : '☀'} theme: {theme}</Btn>
          </div>
        )
      })()}
      {setupOpen && (() => {
        const sel = pythons?.find((p) => p.path === selectedPy) ?? null
        return (
          <div style={{ position: 'fixed', inset: 0, zIndex: 60, background: 'rgba(0,0,0,0.55)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ width: 520, maxWidth: 'calc(100vw - 40px)', maxHeight: 'calc(100vh - 60px)', overflow: 'auto', background: 'var(--bg-1)', border: '1px solid var(--line-strong)', borderRadius: 14, boxShadow: '0 8px 32px rgba(0,0,0,0.45)', padding: 18 }}>
              <strong style={{ color: 'var(--text-0)', fontSize: 15 }}>Kernel setup</strong>
              <div style={{ ...hint, fontSize: 12, margin: '4px 0 14px' }}>
                local kernel isn't starting — Python dependencies are missing
                {envStatus && envStatus.missing.length > 0 && <><br />missing: {envStatus.missing.join(', ')}</>}
              </div>

              {pythons === null ? (
                <div style={{ ...hint, fontSize: 12 }}>scanning…</div>
              ) : pythons.length === 0 ? (
                <div>
                  <div style={{ ...hint, fontSize: 12, marginBottom: 8 }}>no Python found — install Python 3.10+ (e.g. brew install python) then rescan</div>
                  <Btn onClick={rescan}>rescan</Btn>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {pythons.map((p) => {
                    const on = p.path === selectedPy
                    return (
                      <div key={p.path} onClick={() => !installing && setSelectedPy(p.path)}
                        style={{ display: 'flex', alignItems: 'baseline', gap: 8, padding: '6px 8px', borderRadius: 10, cursor: installing ? 'default' : 'pointer',
                          border: `1px solid ${on ? 'var(--accent)' : 'var(--line-strong)'}`, background: 'var(--bg-2)' }}>
                        <span style={{ color: on ? 'var(--accent)' : 'var(--text-2)' }}>{on ? '●' : '○'}</span>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0, flex: 1 }}>
                          <span className="mono" style={{ color: 'var(--text-1)', fontSize: 11.5, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.path}</span>
                          <span style={{ ...hint, fontSize: 11 }}>
                            {p.version} · {p.source}
                            {p.ready
                              ? <span style={{ color: 'var(--accent)' }}> · ready ✓</span>
                              : <span style={{ color: 'var(--text-2)' }}> · missing: {(p.pip ? p.missing : ['pip', ...p.missing]).join(', ')}</span>}
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}

              {pythons !== null && pythons.length > 0 && (
                <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
                  <Btn onClick={browsePython} disabled={installing}>Browse…</Btn>
                  <Btn onClick={rescan} disabled={installing}>rescan</Btn>
                </div>
              )}

              {sel && (
                <div style={{ marginTop: 14 }}>
                  {sel.ready ? (
                    <Btn onClick={useThisPython} color="var(--accent)" disabled={!selectedPy || installing}>Use this Python</Btn>
                  ) : (
                    <>
                      <Btn onClick={startInstall} color="var(--accent)" disabled={!selectedPy || installing}>{installing ? 'Installing…' : 'Install dependencies'}</Btn>
                      <div style={{ ...hint, fontSize: 11, marginTop: 6 }}>~2GB download (torch) — may take several minutes</div>
                    </>
                  )}
                </div>
              )}

              {(installLog.length > 0 || installing) && (
                <div ref={installLogRef} className="mono" style={{ marginTop: 12, height: 180, overflow: 'auto', background: 'var(--bg-0)', border: '1px solid var(--line-strong)', borderRadius: 10, padding: 8, fontSize: 11, whiteSpace: 'pre-wrap', color: 'var(--text-1)' }}>
                  {installLog.join('\n')}
                </div>
              )}

              {installExit && !installExit.ok && (
                <div style={{ marginTop: 12, padding: 10, border: '1px solid var(--line-strong)', borderRadius: 10, background: 'var(--bg-2)' }}>
                  <div style={{ color: 'var(--text-0)', fontSize: 12 }}>pip failed (exit {installExit.code}) — see log above</div>
                  <div style={{ ...hint, fontSize: 11, marginTop: 4 }}>externally-managed or offline? pick a different Python from the list</div>
                </div>
              )}

              {!installing && installExit?.ok && (
                <div style={{ ...hint, fontSize: 12, marginTop: 12 }}>starting kernel…</div>
              )}

              <div style={{ display: 'flex', gap: 14, marginTop: 16, ...hint, fontSize: 11 }}>
                <span onClick={() => { tauriInvokeValue<EnvStatus>('kernel_env_status').then(setEnvStatus).catch(() => {}); rescan() }} style={{ cursor: 'pointer' }}>retry</span>
                <span onClick={() => setSetupOpen(false)} style={{ cursor: 'pointer' }}>dismiss</span>
                {localStorage.getItem('ps_kernel_url') && <span onClick={useLocalKernel} style={{ cursor: 'pointer' }}>use local kernel</span>}
              </div>
            </div>
          </div>
        )
      })()}
      {remoteStale && kernelUp === false && (
        <div style={{ position: 'fixed', bottom: 40, left: '50%', transform: 'translateX(-50%)', zIndex: 55, background: 'var(--bg-1)', border: '1px solid var(--line-strong)', borderRadius: 10, boxShadow: '0 4px 16px rgba(0,0,0,0.4)', padding: '8px 12px', fontSize: 12, color: 'var(--text-1)', display: 'flex', gap: 12, alignItems: 'center' }}>
          <span>remote kernel unreachable</span>
          <span onClick={() => { resumeReconnect(); setRemoteStale(false) }} style={{ cursor: 'pointer', color: 'var(--accent)' }}>retry</span>
          <span onClick={useLocalKernel} style={{ cursor: 'pointer', color: 'var(--accent)' }}>use local kernel</span>
        </div>
      )}
      {toasts.length > 0 && (
        <div className="toast-stack">
          {toasts.map((t) => <div key={t.id} className="toast" onClick={() => dismissToast(t.id)} title="dismiss">{t.text}</div>)}
        </div>
      )}
    </div>
  )
}
