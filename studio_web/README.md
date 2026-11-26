# studio_web — Parametric Studio desktop app

Tauri v2 shell (Rust, `src-tauri/`) + React/TypeScript frontend (`src/`). See the
[repository README](../README.md) for what the app does and how it works.

## Develop

```bash
npm ci
npm run tauri dev     # runs the Vite dev server + the Tauri shell
```

`tauri dev` spawns the Python kernel from `../parametic_studio` (two levels up from
`src-tauri`). Install its dependencies first: `pip install -r ../requirements-studio.txt`.

## Build

```bash
npm run tauri build
```

`scripts/bundle-kernel.mjs` runs first (via `beforeBuildCommand`) and copies `parametic_studio/`
and `requirements-studio.txt` into `src-tauri/resources/` so the packaged app is self-contained.
Bundles land in `src-tauri/target/release/bundle/`.
