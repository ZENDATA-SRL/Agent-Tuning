"""Remote-execution runner: rsync the project to a GPU host, run training there, fetch the artifacts back.

Usage:
    python -m train --remote --dataset data/traces.jsonl --output-dir outputs/run-1

All connection parameters come from `RemoteConfig.from_env()` (`.env` file).
This module shells out to the system `ssh` and `rsync` binaries; it does not
add a Python SSH dependency.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from train.config import RemoteConfig


# Files / dirs that should never be synced to the remote.
# Kept conservative: project sources, requirements, scripts; no caches, no
# previous outputs, no virtualenvs.
_RSYNC_EXCLUDES: tuple[str, ...] = (
    ".git/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".DS_Store",
    "outputs/",
    "checkpoints/",
    "wandb/",
    ".env",            # secrets stay local
    "data/",           # synced separately and selectively
)


def _ssh_base_args(remote: RemoteConfig) -> list[str]:
    """Common SSH arguments shared by `ssh` invocations and `rsync -e`."""
    args = ["-p", str(remote.port)]
    if remote.key_path:
        args += ["-i", remote.key_path]
    # Keep connections alive across slow rsync transfers.
    args += [
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=10",
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    if remote.password:
        # With password auth we MUST allow the password prompt that sshpass
        # will satisfy; `BatchMode=yes` would suppress it and break login.
        # We also disable public-key auth to avoid the client trying every
        # key in the agent first (which can lock the account on strict
        # servers after a few "wrong" attempts).
        args += [
            "-o", "BatchMode=no",
            "-o", "PubkeyAuthentication=no",
            "-o", "PreferredAuthentications=password",
        ]
    else:
        # Key-based / agent-based auth: never prompt, fail fast instead.
        args += ["-o", "BatchMode=yes"]
    return args


def _sshpass_prefix(remote: RemoteConfig) -> list[str]:
    """Return the `sshpass -e` prefix when password auth is configured.

    `-e` makes sshpass read the password from the `SSHPASS` env var, which
    is the safest of its options:
      - not visible in `ps` (unlike `-p PASSWORD`)
      - not on disk (unlike `-f FILE`)
    Returns an empty list when no password is set, so callers can prepend
    unconditionally.
    """
    if not remote.password:
        return []
    if shutil.which("sshpass") is None:
        raise RuntimeError(
            "SSH_REMOTE_PASSWORD is set but `sshpass` is not installed.\n"
            "Install it with one of:\n"
            "  macOS  : brew install hudochenkov/sshpass/sshpass\n"
            "  Debian : sudo apt-get install sshpass\n"
            "  Fedora : sudo dnf install sshpass\n"
            "Or switch to SSH key authentication (recommended) by removing "
            "SSH_REMOTE_PASSWORD from .env."
        )
    return ["sshpass", "-e"]


def _build_env(remote: RemoteConfig) -> dict[str, str] | None:
    """Build the env dict for subprocess, injecting SSHPASS when needed."""
    if not remote.password:
        return None
    env = os.environ.copy()
    env["SSHPASS"] = remote.password
    return env


_SECRET_ENV_PREFIXES: tuple[str, ...] = (
    "SSHPASS=",
    "WANDB_API_KEY=",
)


def _redact(cmd: list[str]) -> str:
    """Pretty-print a command for logging without leaking secrets.

    Two layers of scrubbing:
      1. Argv elements that ARE a secret env binding (e.g. `SSHPASS=...`)
         are replaced wholesale with `KEY=***`.
      2. Embedded secret bindings inside a longer argv element (e.g. the
         full remote shell command containing `WANDB_API_KEY='...'` as part
         of the env prefix from `_env_prefix_for_remote`) are surgically
         redacted in-place via regex so the surrounding command structure
         stays readable.
    """
    import re
    pattern = re.compile(
        r"(" + "|".join(re.escape(p) for p in _SECRET_ENV_PREFIXES) + r")"
        r"(?:'[^']*'|\"[^\"]*\"|\S+)"
    )
    out: list[str] = []
    for c in cmd:
        for prefix in _SECRET_ENV_PREFIXES:
            if c.startswith(prefix):
                out.append(f"{prefix}***")
                break
        else:
            out.append(shlex.quote(pattern.sub(lambda m: f"{m.group(1)}***", c)))
    return " ".join(out)


def _rsync_ssh_string(remote: RemoteConfig) -> str:
    """The string passed to `rsync -e` to use the same SSH options.

    When password auth is in use we wrap the ssh call with `sshpass -e` so
    that rsync's underlying ssh process inherits the same authentication
    path as our explicit ssh invocations.
    """
    parts = _sshpass_prefix(remote) + ["ssh"] + _ssh_base_args(remote)
    return " ".join(shlex.quote(p) for p in parts)


def _expand_remote_path(remote: RemoteConfig, path: str) -> str:
    """Expand a leading `~` for use in `rsync` destinations.

    rsync handles `~` correctly only when the destination is quoted on the
    remote shell; we keep it simple by leaving `~` to be expanded by the
    remote shell itself when used in `ssh` commands, but for rsync paths
    we keep `~/...` as-is — modern rsync forwards the expansion.
    """
    return path


def _run(
    cmd: list[str],
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> int:
    """Stream stdout/stderr live, return exit code."""
    print(f"[train.remote] $ {_redact(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Remote command failed (exit {proc.returncode})")
    return proc.returncode


def _rsync_push(
    remote: RemoteConfig,
    local: str | Path,
    remote_path: str,
    *,
    excludes: tuple[str, ...] = (),
) -> None:
    """rsync local → remote. `local` may be a directory or a single file."""
    local = str(local)
    target = f"{remote.ssh_target()}:{_expand_remote_path(remote, remote_path)}"

    cmd: list[str] = _sshpass_prefix(remote) + [
        "rsync", "-az", "--progress",
        "-e", _rsync_ssh_string(remote),
    ]
    for ex in excludes:
        cmd += ["--exclude", ex]
    cmd += [local, target]

    _run(cmd, env=_build_env(remote))


def _rsync_pull(
    remote: RemoteConfig,
    remote_path: str,
    local: str | Path,
) -> None:
    """rsync remote → local."""
    local_str = str(local)
    Path(local_str).parent.mkdir(parents=True, exist_ok=True)
    src = f"{remote.ssh_target()}:{_expand_remote_path(remote, remote_path)}"

    cmd = _sshpass_prefix(remote) + [
        "rsync", "-az", "--progress",
        "-e", _rsync_ssh_string(remote),
        src, local_str,
    ]
    _run(cmd, env=_build_env(remote))


def _ssh_exec(remote: RemoteConfig, command: str) -> None:
    """Execute a shell command on the remote, streaming output."""
    cmd = (
        _sshpass_prefix(remote)
        + ["ssh"]
        + _ssh_base_args(remote)
        + [remote.ssh_target(), command]
    )
    _run(cmd, env=_build_env(remote))


def _remote_quote(path: str) -> str:
    """Shell-quote a remote path while preserving leading `~/` expansion.

    `shlex.quote('~/foo')` → `'~/foo'` (single-quoted), which means the
    remote shell sees the tilde as a literal character and does NOT
    expand it to $HOME. To keep `~/...` semantics we split off the
    leading `~/` (or a bare `~`), quote only the remainder, and
    concatenate without an intervening space.

    Examples:
        _remote_quote('~/agent-tuning/requirements.txt')
            -> "~/'agent-tuning/requirements.txt'"
        _remote_quote('/usr/bin/python3.12')
            -> "'/usr/bin/python3.12'"
        _remote_quote('python3.12')
            -> "'python3.12'"
    """
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def _ssh_mkdir(remote: RemoteConfig, path: str) -> None:
    """Ensure a directory exists on the remote (no-op if already present)."""
    _ssh_exec(remote, f"mkdir -p {_remote_quote(path)}")


def _venv_dir(remote: RemoteConfig) -> str:
    """Infer the venv root from `python_bin` (`.../<venv>/bin/python` → `.../<venv>`).

    If `python_bin` does not look like a venv interpreter we still derive
    something — e.g. plain `python` becomes `~/agent-tuning/.venv` — but
    in that case the user is expected to manage their env manually and
    `_bootstrap_remote_env` is a no-op (we don't touch a non-venv path).
    """
    p = remote.python_bin
    # Common case: ".../bin/python", ".../bin/python3", ".../bin/python3.12"
    if "/bin/" in p:
        return p.rsplit("/bin/", 1)[0]
    # Fallback: assume a sibling `.venv/` of the project on the remote.
    return f"{remote.remote_workdir.rstrip('/')}/.venv"


def _bootstrap_remote_env(remote: RemoteConfig) -> None:
    """Create the remote venv (if missing) and (re)install requirements.

    Idempotent:
      - If the venv interpreter already exists, only `pip install -r` runs
        (pip itself short-circuits on already-satisfied requirements).
      - If the venv is missing, we create it with `bootstrap_python` —
        failing fast with an actionable message when that interpreter is
        not in the remote PATH.

    We deliberately avoid `pip install --upgrade` to keep behaviour
    predictable on a frozen requirements set.
    """
    venv = _venv_dir(remote)
    venv_python = remote.python_bin
    bootstrap = remote.bootstrap_python
    project = remote.remote_workdir

    # `command -v` is POSIX; `which` is not on minimal images. The `set -e`
    # makes the whole command exit non-zero if any step fails, which our
    # `_run` translates into a clean RuntimeError with the exit code.
    script = (
        f"set -e; "
        # Step 1 — make sure bootstrap python exists, or fail with a
        # helpful message before doing anything else.
        f'if [ ! -x {_remote_quote(venv_python)} ]; then '
        f'  if ! command -v {shlex.quote(bootstrap)} >/dev/null 2>&1; then '
        f'    echo "[bootstrap] ERROR: {bootstrap} not found in remote PATH." >&2; '
        f'    echo "[bootstrap] Install it (apt/dnf/brew) or set SSH_REMOTE_BOOTSTRAP_PYTHON in .env." >&2; '
        f'    exit 127; '
        f'  fi; '
        f'  echo "[bootstrap] Creating venv at {venv} using $({shlex.quote(bootstrap)} -V)"; '
        f'  {shlex.quote(bootstrap)} -m venv {_remote_quote(venv)}; '
        f'fi; '
        # Step 2 — upgrade pip itself (small, safe, fixes a lot of
        # head-scratchers on freshly-bootstrapped venvs with old pip).
        f'{_remote_quote(venv_python)} -m pip install --upgrade pip --quiet; '
        # Step 3 — install requirements. We print a marker so users can
        # grep through long transcripts.
        # We force `--upgrade-strategy eager` only on the first install
        # path. For idempotent subsequent runs we rely on pip's default
        # `only-if-needed`: a package is upgraded ONLY when its installed
        # version no longer satisfies the constraint (e.g. when we bump
        # transformers>=5.5.2 from a previous <=5.2.0 pin). This avoids
        # silently pulling in major regressions on every run.
        f'echo "[bootstrap] Installing requirements (this may take a while on first run)..."; '
        f'{_remote_quote(venv_python)} -m pip install '
        f'--upgrade-strategy only-if-needed '
        f'-r {_remote_quote(project + "/requirements.txt")}; '
        f'echo "[bootstrap] Done."'
    )
    _ssh_exec(remote, script)


# Environment variables forwarded from the local shell to the remote
# `python -m train` process. We forward only the variables the training
# pipeline actually consumes — never wildcard `env` to avoid leaking
# unrelated secrets onto the wire and into remote shell history.
#
# Note we deliberately do NOT forward SSH_REMOTE_PASSWORD or other SSH_*
# credentials: those are local-only and have no business on the remote.
_FORWARDED_ENV_VARS: tuple[str, ...] = (
    # Model selection (consumed by train/__main__.py default for --model-name)
    "TRAIN_MODEL_NAME",
    # HuggingFace token — required for gated models (Gemma, Llama, etc.)
    # when the remote host has no ~/.cache/huggingface/token configured.
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",  # legacy alias still accepted by `huggingface_hub`
    # W&B credentials + run identity (consumed by train/wandb_setup.py and
    # the underlying `wandb` SDK). WANDB_API_KEY in particular is required
    # for `wandb.init` to succeed in non-interactive environments like a
    # remote SSH session.
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_TAGS",
    "WANDB_BASE_URL",  # for self-hosted W&B servers
    "WANDB_MODE",      # "online" | "offline" | "disabled"
)


def _env_prefix_for_remote() -> str:
    """Return a `KEY=value KEY=value ...` prefix for the remote command.

    Only forwards variables that are actually set in the local environment,
    so the remote shell sees a minimal, predictable environment. Values are
    `shlex.quote`d so passwords / API keys with shell metacharacters are
    safe.
    """
    parts: list[str] = []
    for key in _FORWARDED_ENV_VARS:
        val = os.environ.get(key)
        if val:
            parts.append(f"{key}={shlex.quote(val)}")
    return " ".join(parts)


def _build_remote_command(
    remote: RemoteConfig,
    project_dir: str,
    train_args: list[str],
) -> str:
    """Compose the shell command executed on the remote host.

    We `cd` into the project, optionally run `pre_command` (e.g. activate a
    venv / conda env), then exec `python -m train` with the same arguments
    the user passed locally — minus `--remote` (which would loop) and with
    paths translated to be remote-relative.

    A small allow-list of env vars (see `_FORWARDED_ENV_VARS`) is prefixed
    inline to the python invocation so e.g. `WANDB_API_KEY` from the local
    `.env` is available to the remote training process WITHOUT having to
    redundantly maintain a copy on the GPU host.
    """
    quoted = " ".join(shlex.quote(a) for a in train_args)
    parts = [f"cd {_remote_quote(project_dir)}"]
    if remote.pre_command:
        parts.append(remote.pre_command)
    env_prefix = _env_prefix_for_remote()
    train_invocation = f"{_remote_quote(remote.python_bin)} -m train {quoted}"
    if env_prefix:
        train_invocation = f"{env_prefix} {train_invocation}"
    parts.append(train_invocation)
    return " && ".join(parts)


def _filter_remote_flags(argv: list[str]) -> list[str]:
    """Drop `--remote` from argv before forwarding to the remote process.

    The remote invocation must execute the local code path
    (`run_sft`), not recurse into another remote dispatch.
    """
    return [a for a in argv if a != "--remote"]


def run_remote(
    remote: RemoteConfig,
    *,
    dataset_path: str,
    output_dir: str,
    forwarded_argv: list[str],
    project_root: str | Path | None = None,
) -> None:
    """Execute the SFT training on a remote host via SSH + rsync.

    Args:
        remote:           Connection profile (typically `RemoteConfig.from_env()`).
        dataset_path:     Local dataset path. Will be rsynced to
                          `<remote_workdir>/<dataset_path>` if `sync_dataset`.
        output_dir:       Local output dir. After training, the remote dir
                          `<remote_workdir>/<output_dir>` is rsynced back here
                          if `fetch_output`.
        forwarded_argv:   The `sys.argv[1:]` of the parent CLI; passed verbatim
                          to `python -m train` on the remote (after removing
                          `--remote`).
        project_root:     Local directory to rsync as the project. Defaults
                          to the repo root inferred from this file's location.
    """
    project_root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    if not project_root.exists():
        raise FileNotFoundError(f"Project root not found: {project_root}")

    print(f"[train.remote] Project root:   {project_root}")
    print(f"[train.remote] Remote target:  {remote.ssh_target()}:{remote.remote_workdir}")
    print(f"[train.remote] Python (remote): {remote.python_bin}")
    if remote.password:
        print("[train.remote] Auth:           password (sshpass)")
    elif remote.key_path:
        print(f"[train.remote] Auth:           key ({remote.key_path})")
    else:
        print("[train.remote] Auth:           ssh-agent / ~/.ssh/config")
    if remote.pre_command:
        print(f"[train.remote] Pre-command:    {remote.pre_command}")

    # 1. Make sure the remote workdir exists.
    _ssh_mkdir(remote, remote.remote_workdir)

    # 2. Push project tree (rsync trailing slash to copy *contents*).
    print("[train.remote] Syncing project sources →")
    _rsync_push(
        remote,
        local=str(project_root) + "/",
        remote_path=remote.remote_workdir + "/",
        excludes=_RSYNC_EXCLUDES,
    )

    # 3. Push dataset (kept separate so we can disable on shared filesystems).
    if remote.sync_dataset:
        local_ds = Path(dataset_path)
        if not local_ds.exists():
            raise FileNotFoundError(f"Dataset not found locally: {local_ds}")
        # Mirror the same relative path on the remote (e.g. data/foo.jsonl).
        remote_ds = f"{remote.remote_workdir}/{dataset_path}"
        _ssh_mkdir(remote, str(Path(remote_ds).parent))
        print(f"[train.remote] Syncing dataset → {remote_ds}")
        _rsync_push(remote, local=local_ds, remote_path=remote_ds)

    # 4. Bootstrap the remote venv if needed and (re)install requirements.
    print("[train.remote] Ensuring remote venv + dependencies")
    _bootstrap_remote_env(remote)

    # 5. Run training remotely.
    forwarded = _filter_remote_flags(forwarded_argv)
    remote_cmd = _build_remote_command(remote, remote.remote_workdir, forwarded)
    print("[train.remote] Launching remote training")
    _ssh_exec(remote, remote_cmd)

    # 6. Pull back the LoRA adapter / tokenizer.
    if remote.fetch_output:
        remote_out = f"{remote.remote_workdir}/{output_dir}/"
        local_out = Path(output_dir)
        local_out.mkdir(parents=True, exist_ok=True)
        print(f"[train.remote] Fetching outputs ← {remote_out}")
        _rsync_pull(remote, remote_path=remote_out, local=str(local_out) + "/")

    print("[train.remote] Done.")
