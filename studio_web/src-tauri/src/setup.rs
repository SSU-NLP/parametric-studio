//! First-run kernel setup: discover local pythons, probe their deps, one-click
//! `pip install -r requirements-studio.txt`, and respawn the app-owned kernel.
//! All the GPU-less onboarding surface the frontend setup overlay drives.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tauri::{AppHandle, Emitter, Manager};

use crate::{kernel_running, load_config, spawn_kernel, studio_home};

/// Kernel deps we check for (subset of requirements-studio.txt that gates the kernel booting).
const KERNEL_DEPS: [&str; 6] = ["torch", "transformers", "fastapi", "uvicorn", "websockets", "datasets"];

/// One-shot importability probe — `importlib.util.find_spec` is fast (no real imports).
/// Prints a single JSON line: {"version":"3.11.9","pip":true,"missing":["torch",...]}.
const PROBE_SCRIPT: &str = "import json,sys,importlib.util as u;m=[\"torch\",\"transformers\",\"fastapi\",\"uvicorn\",\"websockets\",\"datasets\"];print(json.dumps({\"version\":\".\".join(map(str,sys.version_info[:3])),\"pip\":u.find_spec(\"pip\") is not None,\"missing\":[x for x in m if u.find_spec(x) is None]}))";

#[derive(serde::Serialize, Clone)]
pub struct PyInfo {
    path: String,         // realpath (dedup key)
    version: String,      // "3.11.9"
    source: String,       // "config" | "conda (active)" | "conda" | "pyenv" | "pyenv shim" | "homebrew" | "python.org" | "system" | "custom"
    pip: bool,
    missing: Vec<String>, // subset of KERNEL_DEPS not importable
    ready: bool,          // pip && missing.is_empty()
}

#[derive(serde::Serialize)]
pub struct EnvStatus {
    python_path: Option<String>, // None = no usable python found at all
    deps_ok: bool,
    missing: Vec<String>,
}

/// Only one pip install may run at a time — the overlay's [Install] button + a respawn racing it
/// would otherwise fight over the same requirements file. `swap(true)` claims it; every exit clears.
static INSTALLING: AtomicBool = AtomicBool::new(false);

// ---- pure helpers (unit-tested) --------------------------------------------------------------

/// Parse the probe's stdout into a PyInfo. Tolerates trailing noise (deprecation warnings, etc.)
/// by scanning from the last line for the first that parses as the expected JSON object.
fn parse_probe(path: &str, source: &str, stdout: &str) -> Option<PyInfo> {
    for line in stdout.lines().rev() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(version) = v["version"].as_str() {
                let pip = v["pip"].as_bool().unwrap_or(false);
                let missing: Vec<String> = v["missing"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                let ready = pip && missing.is_empty();
                return Some(PyInfo {
                    path: path.to_string(),
                    version: version.to_string(),
                    source: source.to_string(),
                    pip,
                    missing,
                    ready,
                });
            }
        }
    }
    None
}

/// The locked pip arg vector — mirrors installer-hooks.nsh; requirements file is the single dep source.
fn pip_install_args(req: &Path) -> Vec<String> {
    vec![
        "-m".into(),
        "pip".into(),
        "install".into(),
        "--disable-pip-version-check".into(),
        "-r".into(),
        req.to_string_lossy().into_owned(),
    ]
}

/// Ordered candidate python locations (existence unchecked — probing skips missing ones).
/// Order = priority: the first probe of a given realpath wins, so "config" labels win a dedup.
fn candidate_paths(configured: Option<String>, home: &Path) -> Vec<(PathBuf, &'static str)> {
    let mut v: Vec<(PathBuf, &'static str)> = Vec::new();
    // config python_path is cross-platform — common to both.
    if let Some(c) = configured {
        v.push((PathBuf::from(c), "config"));
    }
    #[cfg(unix)]
    {
        if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
            if !prefix.is_empty() {
                v.push((PathBuf::from(prefix).join("bin/python"), "conda (active)"));
            }
        }
        v.push((home.join("miniconda3/bin/python"), "conda"));
        v.push((home.join("anaconda3/bin/python"), "conda"));
        // pyenv versions — plain read_dir, no glob crate.
        if let Ok(rd) = std::fs::read_dir(home.join(".pyenv/versions")) {
            for e in rd.flatten() {
                v.push((e.path().join("bin/python"), "pyenv"));
            }
        }
        v.push((home.join(".pyenv/shims/python3"), "pyenv shim"));
        v.push((PathBuf::from("/opt/homebrew/bin/python3"), "homebrew"));
        v.push((PathBuf::from("/usr/local/bin/python3"), "homebrew"));
        v.push((PathBuf::from("/usr/bin/python3"), "system"));
    }
    #[cfg(windows)]
    {
        // Windows conda has no bin/ — python.exe sits at the prefix root.
        if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
            if !prefix.is_empty() {
                v.push((PathBuf::from(prefix).join("python.exe"), "conda (active)"));
            }
        }
        v.push((home.join("miniconda3").join("python.exe"), "conda"));
        v.push((home.join("anaconda3").join("python.exe"), "conda"));
        // pyenv-win versions — plain read_dir, no glob crate.
        if let Ok(rd) = std::fs::read_dir(home.join(".pyenv").join("pyenv-win").join("versions")) {
            for e in rd.flatten() {
                v.push((e.path().join("python.exe"), "pyenv"));
            }
        }
        // python.org / Store installs under %LOCALAPPDATA%\Programs\Python\Python3XX.
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            if !local.is_empty() {
                let local = PathBuf::from(local);
                if let Ok(rd) = std::fs::read_dir(local.join("Programs").join("Python")) {
                    for e in rd.flatten() {
                        v.push((e.path().join("python.exe"), "python.org"));
                    }
                }
                // Microsoft Store shim.
                v.push((
                    local
                        .join("Microsoft")
                        .join("WindowsApps")
                        .join("python.exe"),
                    "system",
                ));
            }
        }
    }
    v
}

fn home_dir() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_default()
}

// ---- probing ---------------------------------------------------------------------------------

/// Run the probe against `program` (a path or a bare name resolved via PATH). No existence check —
/// a failed spawn just yields None. 10s timeout guards a hung interpreter.
async fn probe_program(program: &Path, source: &str) -> Option<PyInfo> {
    let out = tokio::time::timeout(
        Duration::from_secs(10),
        tokio::process::Command::new(program)
            .arg("-c")
            .arg(PROBE_SCRIPT)
            .output(),
    )
    .await
    .ok()? // timeout
    .ok()?; // spawn/io error
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let canonical = std::fs::canonicalize(program).unwrap_or_else(|_| program.to_path_buf());
    parse_probe(&canonical.to_string_lossy(), source, &stdout)
}

/// Discover candidate: skip cheaply if the path doesn't exist, else probe.
async fn probe_one(path: PathBuf, source: &str) -> Option<PyInfo> {
    if !path.exists() {
        return None;
    }
    probe_program(&path, source).await
}

// ---- commands --------------------------------------------------------------------------------

#[tauri::command]
pub async fn discover_pythons() -> Vec<PyInfo> {
    let configured = load_config().0;
    let home = home_dir();
    let mut out: Vec<PyInfo> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (path, source) in candidate_paths(configured, &home) {
        if let Some(info) = probe_one(path, source).await {
            // dedup by realpath, keeping the first (highest-priority) occurrence.
            if seen.insert(info.path.clone()) {
                out.push(info);
            }
        }
    }
    log::info!("discover_pythons: {} usable python(s)", out.len());
    out
}

#[tauri::command]
pub async fn probe_python(path: String) -> Option<PyInfo> {
    probe_one(PathBuf::from(path), "custom").await
}

#[tauri::command]
pub async fn kernel_env_status() -> EnvStatus {
    // configured python_path if set, else bare `python3`/`python` (Command::new resolves via PATH).
    let candidates: Vec<(PathBuf, &str)> = match load_config().0 {
        Some(c) => vec![(PathBuf::from(c), "config")],
        None => vec![
            (PathBuf::from("python3"), "system"),
            (PathBuf::from("python"), "system"),
        ],
    };
    for (path, source) in candidates {
        if let Some(info) = probe_program(&path, source).await {
            return EnvStatus {
                python_path: Some(info.path),
                deps_ok: info.ready,
                missing: info.missing,
            };
        }
    }
    EnvStatus {
        python_path: None,
        deps_ok: false,
        missing: KERNEL_DEPS.iter().map(|s| s.to_string()).collect(),
    }
}

/// Resolve requirements-studio.txt: bundled resource (packaged app) → dev fallback (cwd two levels
/// up, same dance as spawn_kernel's kernel-dir fallback → repo root when running `tauri dev`).
fn requirements_path(app: &AppHandle) -> Option<PathBuf> {
    if let Ok(res) = app.path().resource_dir() {
        let p = res.join("requirements-studio.txt");
        if p.exists() {
            return Some(p);
        }
    }
    let dev = std::env::current_dir()
        .ok()?
        .parent()?
        .parent()?
        .join("requirements-studio.txt");
    if dev.exists() {
        Some(dev)
    } else {
        None
    }
}

/// Read config.json as raw JSON, set only python_path, preserve every other key (kernel_dir/model/…).
/// NOT via load_config() — that tuple drops unknown keys and would clobber them on rewrite.
fn write_config_python(python_path: &str) {
    let home = studio_home();
    let path = home.join("config.json");
    let mut v: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    if !v.is_object() {
        v = serde_json::json!({});
    }
    v["python_path"] = serde_json::Value::String(python_path.to_string());
    if let Err(e) = std::fs::create_dir_all(&home) {
        log::warn!("write_config_python: create_dir_all failed: {e}");
        return;
    }
    match serde_json::to_string_pretty(&v) {
        Ok(s) => {
            if let Err(e) = std::fs::write(&path, s) {
                log::warn!("write_config_python: write {path:?} failed: {e}");
            }
        }
        Err(e) => log::warn!("write_config_python: serialize failed: {e}"),
    }
}

/// Stream a child pipe line-by-line as `deps-progress` events (mirrors ssh.rs's emit_status).
fn stream_lines<R: std::io::Read>(app: &AppHandle, pipe: R) {
    let reader = BufReader::new(pipe);
    for line in reader.lines() {
        match line {
            Ok(l) => {
                let _ = app.emit("deps-progress", l);
            }
            Err(_) => break,
        }
    }
}

#[tauri::command]
pub fn install_kernel_deps(app: AppHandle, python_path: String) -> Result<(), String> {
    let req = requirements_path(&app).ok_or_else(|| "requirements-studio.txt not found".to_string())?;
    // claim the single-install slot; any early return below must clear it.
    if INSTALLING.swap(true, Ordering::SeqCst) {
        return Err("install already running".into());
    }
    let args = pip_install_args(&req);
    log::info!("install_kernel_deps: {python_path} {args:?}");
    let mut cmd = Command::new(&python_path);
    cmd.args(&args)
        .env("PYTHONUNBUFFERED", "1") // so pip lines stream instead of buffering to the end
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW — no console popup
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            INSTALLING.store(false, Ordering::SeqCst);
            return Err(format!("pip spawn failed: {e}"));
        }
    };

    // reader threads for each pipe; a joining thread waits and reports completion.
    let out_app = app.clone();
    let h_out = child
        .stdout
        .take()
        .map(|s| std::thread::spawn(move || stream_lines(&out_app, s)));
    let err_app = app.clone();
    let h_err = child
        .stderr
        .take()
        .map(|s| std::thread::spawn(move || stream_lines(&err_app, s)));

    let done_app = app.clone();
    let py = python_path.clone();
    std::thread::spawn(move || {
        if let Some(h) = h_out {
            let _ = h.join();
        }
        if let Some(h) = h_err {
            let _ = h.join();
        }
        let code = match child.wait() {
            Ok(status) => status.code().unwrap_or(-1),
            Err(e) => {
                log::warn!("install_kernel_deps: wait failed: {e}");
                -1
            }
        };
        let ok = code == 0;
        if ok {
            write_config_python(&py); // persist python_path only on success
            log::info!("install_kernel_deps: done ok, wrote python_path");
        } else {
            log::warn!("install_kernel_deps: pip exited {code}");
        }
        INSTALLING.store(false, Ordering::SeqCst);
        let _ = done_app.emit("deps-done", format!(r#"{{"ok":{ok},"code":{code}}}"#));
    });

    Ok(()) // return immediately — progress + completion arrive as events
}

#[tauri::command]
pub fn respawn_kernel(app: AppHandle, python_path: Option<String>) -> Result<(), String> {
    if let Some(p) = python_path.as_deref() {
        if !p.is_empty() {
            write_config_python(p);
        }
    }
    let state = app.state::<crate::Kernel>();
    let mut lock = state
        .0
        .lock()
        .map_err(|e| format!("kernel state lock poisoned: {e}"))?;
    if let Some(mut child) = lock.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    // A just-killed kernel still answers the :8000 TCP probe for a moment; poll until the port
    // frees (≤3s) so spawn_kernel doesn't "attach" to the corpse instead of starting a new one.
    for _ in 0..30 {
        if !kernel_running() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    *lock = spawn_kernel(&app);
    // None + port up = a legit external kernel we attach to; None + port down = spawn really failed.
    if lock.is_none() && !kernel_running() {
        return Err("kernel spawn failed — check python path".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_probe_happy_missing_some() {
        let stdout = r#"{"version":"3.11.9","pip":true,"missing":["torch","datasets"]}"#;
        let info = parse_probe("/x/python", "custom", stdout).expect("should parse");
        assert_eq!(info.version, "3.11.9");
        assert!(info.pip);
        assert_eq!(info.missing, vec!["torch".to_string(), "datasets".to_string()]);
        assert!(!info.ready); // deps missing → not ready
        assert_eq!(info.path, "/x/python");
        assert_eq!(info.source, "custom");
    }

    #[test]
    fn parse_probe_ready_when_pip_and_no_missing() {
        let stdout = r#"{"version":"3.12.1","pip":true,"missing":[]}"#;
        let info = parse_probe("/x/python", "config", stdout).expect("should parse");
        assert!(info.ready);
        assert!(info.missing.is_empty());
    }

    #[test]
    fn parse_probe_tolerates_trailing_noise() {
        // a warning line before the JSON — scan-from-last still finds the JSON.
        let stdout = "DeprecationWarning: whatever\n{\"version\":\"3.10.0\",\"pip\":false,\"missing\":[\"torch\"]}";
        let info = parse_probe("/x", "system", stdout).expect("should parse");
        assert_eq!(info.version, "3.10.0");
        assert!(!info.pip);
        assert!(!info.ready);
    }

    #[test]
    fn parse_probe_garbage_returns_none() {
        assert!(parse_probe("/x", "custom", "not json at all").is_none());
        assert!(parse_probe("/x", "custom", "").is_none());
    }

    #[test]
    fn pip_install_args_exact_vector() {
        let args = pip_install_args(Path::new("/x/requirements-studio.txt"));
        assert_eq!(
            args,
            vec![
                "-m".to_string(),
                "pip".to_string(),
                "install".to_string(),
                "--disable-pip-version-check".to_string(),
                "-r".to_string(),
                "/x/requirements-studio.txt".to_string(),
            ]
        );
    }
}
