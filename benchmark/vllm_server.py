"""Start a local vLLM OpenAI server for a benchmark run."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


DEFAULT_PORT = 8000
DEFAULT_GPU_MEMORY_UTILIZATION = 0.75
DEFAULT_MAX_MODEL_LEN = 16834
_READY_TIMEOUT_S = 900


def resolve_hf_overrides(
    model_id: str,
    *,
    recipe_hf_overrides: dict[str, Any] | Callable[[str], dict[str, Any]] | None = None,
    hf_overrides: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return explicit overrides, else resolve those declared on the recipe.

    Recipe overrides may be a static dict or a ``(model_id) -> dict`` builder
    defined on the model config (see ``VLLMSpec.hf_overrides``).
    """
    if hf_overrides is not None:
        return hf_overrides
    if recipe_hf_overrides is None:
        return None
    if callable(recipe_hf_overrides):
        return recipe_hf_overrides(model_id)
    return recipe_hf_overrides


def _try_load_recipe(model_id: str):
    """Load a shared ModelRecipe when available; otherwise ``None``."""
    try:
        from shared.configs import load_model

        return load_model(model_id)
    except Exception:
        return None


@contextmanager
def vllm_server(
    base_model: str,
    *,
    tool_call_parser: str,
    port: int = DEFAULT_PORT,
    lora_name: str | None = None,
    lora_path: str | Path | None = None,
    hf_overrides: dict | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    chat_template_kwargs: dict | None = None,
    log_path: str | Path = "outputs/vllm-serve.log",
) -> Iterator[str]:
    """Start ``vllm serve`` and yield its OpenAI base URL.

    The server process group is terminated when the context exits, including
    when the benchmark raises. Logs go to ``log_path`` so they do not mix
    with the expected/predicted lines.

    Args:
        base_model: Hugging Face id passed to ``vllm serve``.
        tool_call_parser: vLLM ``--tool-call-parser`` value.
        port: Local port. GPU processes are stopped first so the port and
            the VRAM are free.
        lora_name: Adapter name the benchmark requests. Requires ``lora_path``.
        lora_path: Directory containing ``adapter_config.json``.
        hf_overrides: Optional JSON object for ``--hf-overrides``.
        gpu_memory_utilization: Fraction of GPU memory vLLM may reserve.
        max_model_len: Context length passed to vLLM.
        chat_template_kwargs: Passed as ``--default-chat-template-kwargs``.
        log_path: File that receives the server stdout and stderr.

    Yields:
        Base URL ending in ``/v1``.

    Raises:
        RuntimeError: The port stays taken after the cleanup, or the server
            exits before it is ready.
        TimeoutError: The server does not become ready in time.
    """
    if (lora_name is None) != (lora_path is None):
        raise ValueError("lora_name and lora_path must be set together")
    _free_vram()
    _require_free_port(port)

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = _serve_command(
        base_model,
        tool_call_parser=tool_call_parser,
        port=port,
        lora_name=lora_name,
        lora_path=None if lora_path is None else Path(lora_path),
        hf_overrides=hf_overrides,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        chat_template_kwargs=chat_template_kwargs or {"enable_thinking": False},
    )
    print(f"Starting vLLM on port {port}. Server log: {log_file.resolve()}")
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=_vllm_env(),
        )
        try:
            _wait_until_ready(process, port, log_file)
            print(f"vLLM ready at http://127.0.0.1:{port}/v1")
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            _stop_process_group(process)


def _vllm_env() -> dict[str, str]:
    """Environment for the vLLM child, with the active venv ``bin`` on ``PATH``.

    FlashInfer shells out to ``ninja`` during JIT warmup. Launching
    ``.venv/bin/vllm`` by absolute path does not put that directory on
    ``PATH``, so a venv-installed ``ninja`` would be invisible without this.
    """
    env = os.environ.copy()
    # Do not resolve(): uv venvs symlink python to a managed install, and
    # resolve() would leave the venv ``bin`` (where ``ninja`` lives) off PATH.
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    return env


def _serve_command(
    base_model: str,
    *,
    tool_call_parser: str,
    port: int,
    lora_name: str | None,
    lora_path: Path | None,
    hf_overrides: dict | None,
    gpu_memory_utilization: float,
    max_model_len: int,
    chat_template_kwargs: dict,
) -> list[str]:
    vllm_bin = Path(sys.executable).with_name("vllm")
    command = [
        str(vllm_bin),
        "serve",
        base_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        tool_call_parser,
        "--default-chat-template-kwargs",
        json.dumps(chat_template_kwargs),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
    ]
    if lora_name is not None and lora_path is not None:
        command.extend([
            "--enable-lora",
            "--lora-modules",
            f"{lora_name}={lora_path.resolve()}",
        ])
    if hf_overrides is not None:
        command.extend(["--hf-overrides", json.dumps(hf_overrides)])
    return command


def _free_vram() -> None:
    """Stop other GPU compute processes so vLLM can allocate memory.

    A leftover ``vllm serve`` keeps the weights resident after its port is
    closed. Those processes are terminated here, including the parent server
    when the process on the GPU is only the engine child. The benchmark
    process and its parents are left alone.
    """
    protected = _ancestor_pids(os.getpid())
    apps = [app for app in _gpu_compute_apps() if app[0] not in protected]
    if not apps:
        return

    print("Freeing VRAM:")
    roots: set[int] = set()
    for pid, name, memory in apps:
        print(f"  {name} pid {pid}, {memory}")
        roots.add(_server_root(pid, protected))

    victims: set[int] = set()
    for root in roots:
        victims.add(root)
        victims.update(_descendants(root))
    victims -= protected
    _terminate(victims)
    _wait_until_gpu_clear(protected)


def _gpu_compute_apps() -> list[tuple[int, str, str]]:
    """Return ``(pid, process_name, used_memory)`` for processes on the GPU."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    apps: list[tuple[int, str, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        apps.append((int(parts[0]), parts[1], parts[2]))
    return apps


def _server_root(pid: int, protected: set[int]) -> int:
    """Walk up to the ``vllm`` parent, and stop before the shell or IDE."""
    root = pid
    current = pid
    seen: set[int] = set()
    while current > 1 and current not in seen:
        seen.add(current)
        parent = _ppid(current)
        if parent <= 1 or parent in protected or not _is_vllm_process(parent):
            break
        root = parent
        current = parent
    return root


def _is_vllm_process(pid: int) -> bool:
    command = _cmdline(pid).lower()
    comm = _comm(pid).lower()
    return "vllm" in command or comm.startswith("vllm")


def _ancestor_pids(pid: int) -> set[int]:
    ancestors = {pid}
    current = pid
    while current > 1:
        current = _ppid(current)
        if current <= 1:
            break
        ancestors.add(current)
    return ancestors


def _descendants(root: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        parent = _ppid(pid)
        children.setdefault(parent, []).append(pid)
    found: set[int] = set()
    stack = list(children.get(root, []))
    while stack:
        pid = stack.pop()
        if pid in found:
            continue
        found.add(pid)
        stack.extend(children.get(pid, []))
    return found


def _terminate(pids: set[int]) -> None:
    living = [pid for pid in pids if _pid_exists(pid)]
    for pid in living:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if not any(_pid_exists(pid) for pid in living):
            return
        time.sleep(0.5)
    for pid in living:
        if not _pid_exists(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _wait_until_gpu_clear(protected: set[int]) -> None:
    deadline = time.monotonic() + 20
    remaining: list[tuple[int, str, str]] = []
    while time.monotonic() < deadline:
        remaining = [app for app in _gpu_compute_apps() if app[0] not in protected]
        if not remaining:
            print("VRAM free.")
            return
        time.sleep(0.5)
    still = ", ".join(f"{name} pid {pid}" for pid, name, _memory in remaining)
    raise RuntimeError(f"GPU memory is still held by: {still}")


def _ppid(pid: int) -> int:
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return 0
    for line in status.splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    return 0


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")


def _comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _pid_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def _require_free_port(port: int) -> None:
    """Wait briefly for a just-killed server to drop the socket, then fail."""
    deadline = time.monotonic() + 5
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            busy = sock.connect_ex(("127.0.0.1", port)) == 0
        if not busy:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Port {port} is still in use after freeing the GPU."
            )
        time.sleep(0.25)


def _wait_until_ready(process: subprocess.Popen, port: int, log_path: Path) -> None:
    deadline = time.monotonic() + _READY_TIMEOUT_S
    health_url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited with code {process.returncode}.\n{_log_tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2)
    raise TimeoutError(
        f"vLLM did not become ready within {_READY_TIMEOUT_S}s.\n{_log_tail(log_path)}"
    )


def _stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def run_served_benchmark(
    *,
    base_model: str,
    dataset_path: str,
    tool_call_parser: str | None = None,
    lora_name: str | None = None,
    lora_path: str | Path | None = None,
    hf_overrides: dict | None = None,
    max_traces: int | None = None,
    support_multi_tool_calls: bool | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    decoding_mode: str = "no_thinking",
) -> dict:
    """Start vLLM, replay ``dataset_path`` against it, then stop the server.

    Serving knobs default from the matching ``shared.configs`` recipe when
    present (``tool_call_parser``, ``hf_overrides``, max model length, …).

    Args:
        base_model: Hugging Face id passed to ``vllm serve``.
        dataset_path: JSONL trace file.
        tool_call_parser: vLLM ``--tool-call-parser`` value. ``None`` uses the
            recipe when available.
        lora_name: Adapter name requested by the benchmark client.
        lora_path: LoRA checkpoint directory.
        hf_overrides: Optional ``--hf-overrides`` object. When omitted, the
            recipe ``VLLMSpec.hf_overrides`` is applied if set.
        max_traces: Limit the number of traces. ``None`` runs the full file.
        support_multi_tool_calls: When False, parallel reference tool_calls are
            split into sequential single-call turns. ``None`` resolves from the
            matching recipe when available, else defaults to True.
        gpu_memory_utilization: Optional override for vLLM memory fraction.
        max_model_len: Optional override for vLLM context length.
        decoding_mode: ``"thinking"`` or ``"no_thinking"``. Selects the recipe's
            chat-template kwargs. Unknown modes raise ``KeyError``.

    Returns:
        The benchmark report dict.
    """
    from benchmark.config import InferenceProfile, resolve_support_multi_tool_calls
    from benchmark.runner import run_benchmark

    recipe = _try_load_recipe(base_model)
    parser = tool_call_parser
    if parser is None:
        if recipe is None:
            raise ValueError(
                "tool_call_parser is required when no shared recipe matches "
                f"{base_model!r}."
            )
        parser = recipe.vllm.tool_call_parser

    recipe_overrides = None if recipe is None else recipe.vllm.hf_overrides
    overrides = resolve_hf_overrides(
        base_model,
        recipe_hf_overrides=recipe_overrides,
        hf_overrides=hf_overrides,
    )

    mem = gpu_memory_utilization
    if mem is None and recipe is not None and recipe.vllm.gpu_memory_utilization is not None:
        mem = recipe.vllm.gpu_memory_utilization
    if mem is None:
        mem = DEFAULT_GPU_MEMORY_UTILIZATION

    ctx = max_model_len
    if ctx is None and recipe is not None and recipe.vllm.max_model_len is not None:
        ctx = recipe.vllm.max_model_len
    if ctx is None:
        ctx = DEFAULT_MAX_MODEL_LEN

    chat_kwargs = {"enable_thinking": False}
    if recipe is not None:
        template_kwargs = recipe.decoding_setup(decoding_mode).template_kwargs
        if template_kwargs:
            chat_kwargs = dict(template_kwargs)

    multi = resolve_support_multi_tool_calls(base_model, support_multi_tool_calls)
    served_model = lora_name or base_model
    with vllm_server(
        base_model,
        tool_call_parser=parser,
        lora_name=lora_name,
        lora_path=lora_path,
        hf_overrides=overrides,
        gpu_memory_utilization=mem,
        max_model_len=ctx,
        chat_template_kwargs=chat_kwargs,
    ) as base_url:
        profile = InferenceProfile(
            backend="vllm",
            model=served_model,
            base_url=base_url,
            temperature=0.0,
            max_tokens=1024,
            support_multi_tool_calls=multi,
            extra={
                "extra_body": {
                    "chat_template_kwargs": chat_kwargs,
                },
            },
        )
        return run_benchmark(
            dataset_path=dataset_path,
            profile=profile,
            max_traces=max_traces,
        )


def _log_tail(log_path: Path, lines: int = 40) -> str:
    if not log_path.exists():
        return "(no server log)"
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])
