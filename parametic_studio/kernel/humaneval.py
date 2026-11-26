"""HumanEvalPack pass@k harness — generate a completion, run its unit tests in a
separate python process (timeout-guarded), report pass/fail.

bigcode-evaluation-harness humanevalpack protocol + OpenAI human-eval execution.py
check_correctness pattern. Python only for now; a `language` field is left as the
branch point for Java/other langs later.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

# bigcode humaneval stop sequences: cut the completion at the first thing that
# ends the target function (a new top-level def/class, a comment, a main guard, a print).
STOP_SEQUENCES = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("]


def truncate_completion(completion, stops=STOP_SEQUENCES):
    """Cut a raw model completion at the first stop sequence (bigcode humaneval)."""
    idx = len(completion)
    for s in stops:
        pos = completion.find(s)
        if pos != -1:
            idx = min(idx, pos)
    return completion[:idx]


def build_program(row, completion):
    """Assemble the runnable program (bigcode HumanEvalPack, Python):
    imports + prompt (signature+docstring) + completion (body) + test_setup + test + check(entry_point).
    `completion` is the function body that continues `prompt`."""
    imports = row.get("import", "") or ""
    setup = row.get("test_setup", "") or ""
    return (
        imports + "\n"
        + row["prompt"] + completion + "\n"
        + setup + "\n"
        + row["test"] + "\n"
        + f"check({row['entry_point']})\n"
    )


def check_correctness(program, timeout=10.0):
    """Run `program` in a fresh python subprocess with a timeout (OpenAI human-eval
    check_correctness pattern). Returns (passed, detail). exit 0 → pass; any exception,
    AssertionError, or timeout → fail with a short detail string.

    Separate process + timeout is required (never exec in the kernel process). This is
    research code inside the workspace container — no seccomp sandbox, just isolation."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "candidate.py"
        path.write_text(program)
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout}s"
        if proc.returncode == 0:
            return True, "passed"
        detail = (proc.stderr or proc.stdout or "").strip()
        return False, detail[-500:] if detail else f"exit {proc.returncode}"


def run_program(program, timeout=10.0):
    """Run `program` in a fresh python subprocess, capturing stdout/stderr/exit — for the interactive
    Code tab (show what happened, not a pass/fail verdict). Same process isolation + timeout as
    check_correctness. Returns {exit, stdout, stderr, timed_out, duration_ms}."""
    import time
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "prog.py"
        path.write_text(program)
        t0 = time.time()
        try:
            proc = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            return {"exit": None, "stdout": e.stdout or "", "stderr": e.stderr or "",
                    "timed_out": True, "duration_ms": int((time.time() - t0) * 1000)}
        return {"exit": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
                "timed_out": False, "duration_ms": int((time.time() - t0) * 1000)}


def estimate_pass_at_k(n, c, k):
    """Standard unbiased pass@k estimator: 1 - C(n-c, k)/C(n, k).
    n samples, c correct → probability at least one of k drawn passes.
    n=1,c=1,k=1 → 1.0; n=5,c=0 → 0.0."""
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod
