use std::io::Read;
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem, Submenu};
use tauri::{Emitter, Manager};

mod setup;
mod ssh;
use ssh::SshState;

// The app owns the kernel: spawn on launch (unless one is already serving :8000 — dev/attach
// mode), kill on exit. The kernel's lifecycle is coupled to the app window's.
// `pub(crate)` so setup.rs (respawn_kernel) can reach the managed child.
pub(crate) struct Kernel(pub(crate) Mutex<Option<Child>>);

pub(crate) fn kernel_running() -> bool {
    TcpStream::connect_timeout(
        &"127.0.0.1:8000".parse().unwrap(),
        Duration::from_millis(300),
    )
    .is_ok()
}

pub(crate) fn studio_home() -> PathBuf {
    std::env::var("PARAMETIC_STUDIO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE")) // windows
                .unwrap_or_default();
            PathBuf::from(home).join(".parametic_studio")
        })
}

/// ~/.parametic_studio/config.json: {"python_path": "...", "kernel_dir": "...", "model": "..."}
/// (system-python packaging: the kernel source lives in the repo checkout, python is the user's)
/// python_path is optional — absent → try `python3`/`python` candidates at spawn time.
pub(crate) fn load_config() -> (Option<String>, Option<PathBuf>, Option<String>) {
    let p = studio_home().join("config.json");
    if let Ok(mut f) = std::fs::File::open(&p) {
        let mut s = String::new();
        if f.read_to_string(&mut s).is_ok() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                return (
                    v["python_path"].as_str().map(String::from),
                    v["kernel_dir"].as_str().map(PathBuf::from),
                    v["model"].as_str().map(String::from),
                );
            }
        }
    }
    (None, None, None)
}

/// Configured python_path, else platform-ordered candidates (windows: `python` first).
fn python_candidates(configured: Option<String>) -> Vec<String> {
    if let Some(p) = configured {
        return vec![p];
    }
    if cfg!(windows) {
        vec!["python".into(), "python3".into()]
    } else {
        vec!["python3".into(), "python".into()]
    }
}

/// The kernel source (`parametic_studio/`) is bundled into the app as a resource, so the packaged app
/// finds it with no config — `kernel_dir` becomes automatic. Returns the resource dir if it holds the
/// package. (See tauri.conf.json `bundle.resources` + studio_web/scripts/bundle-kernel.mjs.)
fn bundled_kernel_dir(app: &tauri::AppHandle) -> Option<PathBuf> {
    let res = app.path().resource_dir().ok()?;
    if res.join("parametic_studio").is_dir() {
        Some(res)
    } else {
        None
    }
}

pub(crate) fn spawn_kernel(app: &tauri::AppHandle) -> Option<Child> {
    if kernel_running() {
        log::info!("kernel already on :8000 — attach mode, not spawning");
        return None;
    }
    let (python_path, kernel_dir, model) = load_config();
    // where the `parametic_studio` package lives: explicit config wins → the bundled copy (packaged app,
    // no config needed) → dev fallback (cwd two levels up when running `tauri dev`).
    let dir = kernel_dir
        .or_else(|| bundled_kernel_dir(app))
        .or_else(|| {
            std::env::current_dir()
                .ok()
                .and_then(|d| d.parent().and_then(|p| p.parent()).map(|p| p.to_path_buf()))
        })?;
    if !dir.join("parametic_studio").is_dir() {
        log::error!("no kernel source at {dir:?} — the bundled copy is missing and no kernel_dir is set");
        return None;
    }
    let candidates = python_candidates(python_path);
    for python in &candidates {
        let mut cmd = Command::new(python);
        cmd.args(["-m", "parametic_studio.api"])
            .current_dir(&dir)
            .env("PARAMETIC_STUDIO_PARENT_WATCH", "1") // kernel self-exits if the app dies uncleanly
            .env("PYTHONDONTWRITEBYTECODE", "1"); // resource dir is read-only (Program Files) — don't write .pyc
        if let Some(m) = &model {
            cmd.env("PARAMETIC_STUDIO_MODEL", m);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW — no console popup
        }
        match cmd.spawn() {
            Ok(child) => {
                log::info!("kernel spawned (pid {}) via {} in {:?}", child.id(), python, dir);
                return Some(child);
            }
            Err(e) => log::warn!("kernel spawn via {python} failed: {e}"),
        }
    }
    // frontend shows "kernel offline · reconnecting…" — fix config.json (python_path) and relaunch
    log::error!("kernel spawn failed for all candidates {candidates:?} in {dir:?}");
    None
}

/// Splash → main handoff: show + focus the main window, close the splash. Idempotent.
fn do_close_splash(app: &tauri::AppHandle) {
    if let Some(main) = app.get_webview_window("main") {
        let _ = main.show();
        let _ = main.set_focus();
    }
    if let Some(splash) = app.get_webview_window("splash") {
        let _ = splash.close();
    }
}

/// Frontend calls this once the kernel is reachable. The 8s fallback lives in Rust (setup),
/// NOT the frontend: the main window is hidden at boot and WKWebView suspends timers in
/// hidden webviews, so a JS setTimeout never fires when the kernel is down → stuck splash.
#[tauri::command]
fn close_splash(app: tauri::AppHandle) {
    do_close_splash(&app);
}

/// Native menubar. Custom items emit `menu` events to the webview so App.tsx drives the UI.
fn build_menu(app: &tauri::AppHandle) -> tauri::Result<Menu<tauri::Wry>> {
    let app_menu = Submenu::with_items(
        app,
        "Parametric Studio",
        true,
        &[
            &PredefinedMenuItem::about(app, None, None)?,
            &PredefinedMenuItem::separator(app)?,
            &MenuItem::with_id(app, "settings", "Settings…", true, Some("Cmd+,"))?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::quit(app, None)?,
        ],
    )?;
    // Edit menu — restores standard Cmd+C/V/X/A in the webview. Setting a custom menu replaces
    // the OS default (which included Edit), so these predefined items must be re-added by hand;
    // each carries the standard key equivalent + responder action that revives copy/paste.
    let edit_menu = Submenu::with_items(
        app,
        "Edit",
        true,
        &[
            &PredefinedMenuItem::undo(app, None)?,
            &PredefinedMenuItem::redo(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::cut(app, None)?,
            &PredefinedMenuItem::copy(app, None)?,
            &PredefinedMenuItem::paste(app, None)?,
            &PredefinedMenuItem::select_all(app, None)?,
        ],
    )?;
    let model_menu = Submenu::with_items(
        app,
        "Model",
        true,
        &[
            &MenuItem::with_id(app, "load-model", "Load Model…", true, Some("Cmd+L"))?,
            &MenuItem::with_id(app, "unload-all", "Unload All", true, None::<&str>)?,
        ],
    )?;
    let view_menu = Submenu::with_items(
        app,
        "View",
        true,
        &[
            &MenuItem::with_id(app, "toggle-theme", "Toggle Theme", true, Some("Cmd+Shift+T"))?,
            &MenuItem::with_id(app, "toggle-explorer", "Toggle Explorer", true, Some("Cmd+B"))?,
        ],
    )?;
    Menu::with_items(app, &[&app_menu, &edit_menu, &model_menu, &view_menu])
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            close_splash,
            ssh::ssh_connect,
            ssh::ssh_disconnect,
            setup::discover_pythons,
            setup::probe_python,
            setup::kernel_env_status,
            setup::install_kernel_deps,
            setup::respawn_kernel
        ])
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            app.manage(SshState::default());
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            let menu = build_menu(app.handle())?;
            app.set_menu(menu)?;
            app.on_menu_event(|app, event| {
                let _ = app.emit("menu", event.id().0.as_str());
            });
            app.manage(Kernel(Mutex::new(spawn_kernel(app.handle()))));
            // hard 8s splash fallback — must be Rust-side (hidden webview timers are suspended)
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(8));
                do_close_splash(&handle);
                // Window is now visible. While it was hidden, WKWebView/WebView2 suspended JS timers,
                // so the frontend's own 6s first-run-setup trigger (a setTimeout) never fired — the exact
                // case the onboarding is FOR (kernel down ⇒ window stays hidden the full 8s). Nudge the
                // frontend to run the setup check now that JS is live; event delivery isn't timer-gated.
                if !kernel_running() {
                    let _ = handle.emit("check-setup", "");
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(k) = app.try_state::<Kernel>() {
                    if let Some(mut child) = k.0.lock().unwrap().take() {
                        let _ = child.kill(); // window closed → kernel goes with it
                        let _ = child.wait();
                    }
                }
                // tear down the SSH tunnel (listener + session); the remote kernel is left running
                if let Some(s) = app.try_state::<SshState>() {
                    s.shutdown();
                }
            }
        });
}
