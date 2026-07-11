; Parametric Studio — NSIS installer hooks.
; After the app files are installed, install the Python kernel dependencies with the user's `python`
; on PATH, so the desktop app works with no manual `pip install`. Requires Python already installed
; (e.g. from python.org "Add to PATH" or conda). ~2GB download (torch) — install may take a few minutes.
; Only affects the NSIS setup .exe (not the .msi).

!macro NSIS_HOOK_POSTINSTALL
  DetailPrint "Setting up Parametric Studio kernel dependencies (torch, transformers, ...)."
  DetailPrint "This downloads ~2GB and may take several minutes — please wait."
  ; run pip via the python found on PATH; ExecToLog surfaces output in the installer detail view
  nsExec::ExecToLog 'cmd /c python -m pip install --disable-pip-version-check "torch>=2.3" "transformers>=4.45" datasets fastapi uvicorn websockets'
  Pop $0
  StrCmp $0 "0" deps_ok deps_fail
  deps_fail:
    DetailPrint "Dependency setup did not complete (exit $0)."
    DetailPrint "If Python is missing or not on PATH, install Python, then open PowerShell and run:"
    DetailPrint '  pip install "torch>=2.3" "transformers>=4.45" datasets fastapi uvicorn websockets'
    Goto deps_done
  deps_ok:
    DetailPrint "Kernel dependencies installed. Parametric Studio is ready."
  deps_done:
!macroend
