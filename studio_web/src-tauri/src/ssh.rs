//! SSH tunnel to a remote GPU kernel.
//!
//! `ssh_connect` authenticates to the user's remote host, starts (or reuses) the studio kernel
//! bound to remote 127.0.0.1:8000, then opens a local port-forward: 127.0.0.1:8422 → (over the SSH
//! session, `direct-tcpip`) → remote 127.0.0.1:8000. The frontend then connects to
//! `ws://localhost:8422/ws`. The tunnel is owned by the Rust backend, not the webview, so a webview
//! reload never drops it. Security is the SSH channel itself — the remote kernel is bound to
//! loopback and never exposed on the remote network, so it needs no token.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use russh::client::{self, Handle};
use russh::keys::{load_secret_key, ssh_key, PrivateKeyWithHashAlg};
use russh::{ChannelMsg, Disconnect};
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

const LOCAL_FORWARD_PORT: u16 = 8422;
const REMOTE_KERNEL_PORT: u32 = 8000;
const KERNEL_READY_TIMEOUT_SECS: u64 = 60;

/// A live tunnel: the SSH session handle plus the accept-loop task. Dropping/aborting the accept
/// loop closes the local listener; disconnecting the handle closes the SSH session and every
/// forwarded channel with it.
struct Conn {
    session: Arc<Handle<Client>>,
    accept_loop: JoinHandle<()>,
}

/// Managed Tauri state. `Mutex<Option<..>>` because at most one tunnel exists at a time; a fresh
/// `ssh_connect` tears down any existing one first.
#[derive(Default)]
pub struct SshState(Mutex<Option<Conn>>);

impl SshState {
    /// Abort the accept loop and disconnect the SSH session. Idempotent. The remote kernel is left
    /// running on purpose — the user owns it and other clients may still be attached.
    pub fn shutdown(&self) {
        if let Some(conn) = self.0.lock().unwrap().take() {
            conn.accept_loop.abort();
            let session = conn.session.clone();
            // best-effort graceful disconnect; the drop of `session` closes the socket regardless.
            // tauri::async_runtime::spawn (not tokio::spawn) so this is safe from the main-thread
            // RunEvent::Exit handler, which has no ambient Tokio runtime — tokio::spawn panics there.
            tauri::async_runtime::spawn(async move {
                let _ = session
                    .disconnect(Disconnect::ByApplication, "client disconnect", "")
                    .await;
            });
        }
    }
}

/// russh client handler. Accepts any server key: this is a user-initiated tunnel to a host the user
/// typed themselves (like `ssh -o StrictHostKeyChecking=no`); the SSH transport still encrypts.
struct Client;

impl client::Handler for Client {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &ssh_key::PublicKey,
    ) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

fn emit_status(app: &AppHandle, payload: &str) {
    let _ = app.emit("ssh-status", payload);
}

/// Single-quote a value for safe embedding in a remote `sh -c` command line.
fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Expand a leading `~` in a local filesystem path to the user's home directory (HOME on unix,
/// USERPROFILE on windows). Only a leading `~` / `~/` is expanded; other `~user` forms are left
/// as-is (they'd fail to open, which surfaces a clear "read key" error).
fn expand_home(path: &str) -> String {
    if path == "~" || path.starts_with("~/") {
        if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            return format!("{}{}", home, &path[1..]);
        }
    }
    path.to_string()
}

/// Build the remote command that reuses the kernel if :8000 is already listening, else launches it
/// with `nohup`. `python` may carry candidates (`python3 || python`) chosen by the caller.
fn kernel_launch_command(
    repo_dir: &str,
    python: &str,
    model: &Option<String>,
    hf_home: &Option<String>,
) -> String {
    let model_env = match model {
        Some(m) => format!("PARAMETIC_STUDIO_MODEL={} ", sh_quote(m)),
        None => String::new(),
    };
    // HF cache location — cloud GPU boxes often have a tiny home disk; point HF_HOME at a roomy
    // volume (e.g. /shared) so multi-GB model downloads don't fill the root partition.
    let hf_env = match hf_home {
        Some(h) if !h.trim().is_empty() => {
            let q = sh_quote(h);
            format!("HF_HOME={q} HF_HUB_CACHE={q}/hub ")
        }
        _ => String::new(),
    };
    // -s (not -sf): the kernel returns 404 on `/` (only /ws is a route), and `curl -f` treats any
    // 4xx as failure — so any HTTP response at all means the kernel is up. ss is the fallback.
    let check = format!(
        "curl -s -o /dev/null http://127.0.0.1:{p}/ >/dev/null 2>&1 || (command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ':{p} ')",
        p = REMOTE_KERNEL_PORT
    );
    let launch = format!(
        "cd {dir} && PARAMETIC_STUDIO_HOST=127.0.0.1 PARAMETIC_STUDIO_PORT={port} {hf}{model}nohup {py} -m parametic_studio.api > /tmp/studio-kernel.log 2>&1 &",
        dir = sh_quote(repo_dir),
        port = REMOTE_KERNEL_PORT,
        hf = hf_env,
        model = model_env,
        py = python,
    );
    // if already listening, do nothing; otherwise launch in the background.
    format!("if {check}; then echo already-running; else {launch} echo started; fi")
}

/// Run one remote command over a fresh session channel; return its combined stdout as a String.
async fn remote_exec(session: &Handle<Client>, command: &str) -> Result<String, String> {
    let mut channel = session
        .channel_open_session()
        .await
        .map_err(|e| format!("open session channel: {e}"))?;
    channel
        .exec(true, command.as_bytes())
        .await
        .map_err(|e| format!("exec: {e}"))?;
    let mut out = Vec::new();
    while let Some(msg) = channel.wait().await {
        match msg {
            ChannelMsg::Data { data } => out.extend_from_slice(&data),
            ChannelMsg::ExtendedData { data, .. } => out.extend_from_slice(&data),
            ChannelMsg::Eof | ChannelMsg::Close | ChannelMsg::ExitStatus { .. } => {}
            _ => {}
        }
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// Poll the remote :8000 over SSH exec until it answers, or time out.
async fn wait_kernel_ready(session: &Handle<Client>) -> Result<(), String> {
    // -s not -sf: a 404 on `/` still means the kernel is answering (see kernel_launch_command).
    let probe = format!(
        "curl -s -o /dev/null http://127.0.0.1:{}/ >/dev/null 2>&1 && echo ok",
        REMOTE_KERNEL_PORT
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(KERNEL_READY_TIMEOUT_SECS);
    loop {
        if let Ok(out) = remote_exec(session, &probe).await {
            if out.contains("ok") {
                return Ok(());
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(format!(
                "remote kernel did not become ready on :{} within {}s (see /tmp/studio-kernel.log on the host)",
                REMOTE_KERNEL_PORT, KERNEL_READY_TIMEOUT_SECS
            ));
        }
        tokio::time::sleep(Duration::from_millis(1500)).await;
    }
}

/// Accept loop for the local forward. Each inbound TCP connection gets its own `direct-tcpip`
/// channel to remote 127.0.0.1:8000, and raw bytes are copied both ways — no HTTP parsing, so the
/// WebSocket upgrade passes through untouched.
async fn run_forward(listener: TcpListener, session: Arc<Handle<Client>>) {
    loop {
        let (mut inbound, peer) = match listener.accept().await {
            Ok(pair) => pair,
            Err(e) => {
                log::warn!("ssh forward accept failed: {e}");
                continue;
            }
        };
        let session = session.clone();
        tokio::spawn(async move {
            let channel = match session
                .channel_open_direct_tcpip(
                    "127.0.0.1",
                    REMOTE_KERNEL_PORT,
                    &peer.ip().to_string(),
                    peer.port() as u32,
                )
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    log::warn!("ssh direct-tcpip open failed: {e}");
                    let _ = inbound.shutdown().await;
                    return;
                }
            };
            let mut stream = channel.into_stream();
            // raw bidirectional pipe; carries the WebSocket upgrade + frames verbatim.
            if let Err(e) = tokio::io::copy_bidirectional(&mut inbound, &mut stream).await {
                log::debug!("ssh forward copy ended: {e}");
            }
        });
    }
}

#[tauri::command]
pub async fn ssh_connect(
    app: AppHandle,
    host: String,
    port: u16,
    username: String,
    password: String,
    key_path: Option<String>,
    key_passphrase: Option<String>,
    repo_dir: String,
    python_path: Option<String>,
    model: Option<String>,
    hf_home: Option<String>,
) -> Result<(), String> {
    // tear down any existing tunnel first so a reconnect is clean.
    if let Some(state) = app.try_state::<SshState>() {
        state.shutdown();
    }

    emit_status(&app, r#"{"state":"connecting","detail":"authenticating"}"#);

    let config = Arc::new(client::Config {
        keepalive_interval: Some(Duration::from_secs(20)),
        ..Default::default()
    });
    let mut session = client::connect(config, (host.as_str(), port), Client)
        .await
        .map_err(|e| {
            let msg = format!("connect failed: {e}");
            emit_status(&app, &error_json(&msg));
            msg
        })?;

    // Branch on auth method: a non-empty key_path means .pem/public-key auth (cloud GPU hosts use
    // keys, not passwords); otherwise fall back to password auth exactly as before.
    let use_key = key_path.as_deref().map(|p| !p.is_empty()).unwrap_or(false);
    if use_key {
        let raw_path = key_path.unwrap();
        let path = expand_home(&raw_path);
        // load_secret_key is blocking file I/O; run it off the async reactor. The passphrase is
        // moved into the closure and dropped there — never logged, never stored.
        let passphrase = key_passphrase.clone();
        let load_path = path.clone();
        let key = tokio::task::spawn_blocking(move || {
            load_secret_key(&load_path, passphrase.as_deref())
        })
        .await
        .map_err(|e| {
            let msg = format!("read key {path}: {e}");
            emit_status(&app, &error_json(&msg));
            msg
        })?
        .map_err(|e| {
            // Distinguish "encrypted key, no/blank passphrase given" so the user knows to supply one.
            let msg = if matches!(e, russh::keys::Error::KeyIsEncrypted) {
                format!("read key {path}: key is encrypted — passphrase required")
            } else {
                format!("read key {path}: {e}")
            };
            emit_status(&app, &error_json(&msg));
            msg
        })?;
        // For RSA keys, negotiate the strongest hash the server supports (ssh-rsa vs rsa-sha2-*);
        // non-RSA keys ignore the hash. Fall back to None if negotiation is inconclusive.
        let hash_alg = session
            .best_supported_rsa_hash()
            .await
            .ok()
            .flatten()
            .flatten();
        let auth = session
            .authenticate_publickey(&username, PrivateKeyWithHashAlg::new(Arc::new(key), hash_alg))
            .await
            .map_err(|e| {
                let msg = format!("auth failed: {e}");
                emit_status(&app, &error_json(&msg));
                msg
            })?;
        if !auth.success() {
            let msg = "auth failed: key rejected".to_string();
            emit_status(&app, &error_json(&msg));
            return Err(msg);
        }
    } else {
        let auth = session
            .authenticate_password(&username, password)
            .await
            .map_err(|e| {
                let msg = format!("auth failed: {e}");
                emit_status(&app, &error_json(&msg));
                msg
            })?;
        // password is consumed by authenticate_password (moved above) — never logged, never stored.
        if !auth.success() {
            let msg = "auth failed: password rejected".to_string();
            emit_status(&app, &error_json(&msg));
            return Err(msg);
        }
    }

    let session = Arc::new(session);

    // start (or reuse) the remote kernel.
    emit_status(&app, r#"{"state":"starting-kernel"}"#);
    // caller-provided python, else try python3 then python via shell `||`. Treat an empty string
    // like None — the frontend sends "" when the field is blank, and Some("") would otherwise make
    // the launch `nohup  -m …` (empty python → nohup eats `-m`). GPU boxes often need an explicit
    // interpreter (e.g. /opt/conda/bin/python where torch lives), so the field matters.
    let python = python_path
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "$(command -v python3 || command -v python)".to_string());
    let launch = kernel_launch_command(&repo_dir, &python, &model, &hf_home);
    remote_exec(&session, &launch).await.map_err(|e| {
        let msg = format!("start kernel: {e}");
        emit_status(&app, &error_json(&msg));
        msg
    })?;
    wait_kernel_ready(&session).await.map_err(|e| {
        emit_status(&app, &error_json(&e));
        e
    })?;

    // open the local forward.
    emit_status(&app, r#"{"state":"forwarding"}"#);
    let listener = TcpListener::bind(("127.0.0.1", LOCAL_FORWARD_PORT))
        .await
        .map_err(|e| {
            let msg = format!("bind local :{LOCAL_FORWARD_PORT} failed: {e}");
            emit_status(&app, &error_json(&msg));
            msg
        })?;

    let accept_loop = tokio::spawn(run_forward(listener, session.clone()));

    if let Some(state) = app.try_state::<SshState>() {
        *state.0.lock().unwrap() = Some(Conn {
            session,
            accept_loop,
        });
    }

    emit_status(&app, r#"{"state":"connected"}"#);
    Ok(())
}

#[tauri::command]
pub async fn ssh_disconnect(app: AppHandle) -> Result<(), String> {
    if let Some(state) = app.try_state::<SshState>() {
        state.shutdown();
    }
    emit_status(&app, r#"{"state":"disconnected"}"#);
    Ok(())
}

fn error_json(detail: &str) -> String {
    serde_json::json!({ "state": "error", "detail": detail }).to_string()
}
