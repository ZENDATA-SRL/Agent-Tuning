from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
BASE_MODEL_ID = "base"
BASE_MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"


def list_adapters(outputs_dir: Path | None = None) -> list[dict]:
    adapters = [{
        "id": BASE_MODEL_ID,
        "path": BASE_MODEL_NAME,
        "label": f"originale (no LoRA) · {BASE_MODEL_NAME}",
    }]
    root = outputs_dir or OUTPUTS_DIR
    if not root.exists():
        return adapters
    for cfg in sorted(root.rglob("adapter_config.json")):
        path = cfg.parent
        rel = path.relative_to(root).as_posix()
        adapters.append({
            "id": rel,
            "path": str(path),
            "label": rel,
        })
    return adapters


def resolve_adapter(adapter_id: str, outputs_dir: Path | None = None) -> Path | str:
    if adapter_id == BASE_MODEL_ID:
        return BASE_MODEL_NAME
    root = outputs_dir or OUTPUTS_DIR
    path = (root / adapter_id).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Invalid adapter id: {adapter_id}")
    if not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_id}")
    return path
