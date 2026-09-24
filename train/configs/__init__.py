"""Registry of supported model recipes."""
from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path

from train.config import GRPOConfig, ModelRecipe, SFTConfig

__all__ = [
    "GRPOConfig",
    "ModelRecipe",
    "SFTConfig",
    "available_models",
    "load_model",
]


def _module_name(path: Path, package_dir: Path) -> str:
    relative = path.relative_to(package_dir).with_suffix("")
    return "train.configs." + ".".join(relative.parts)


@lru_cache(maxsize=1)
def _registry() -> dict[str, ModelRecipe]:
    package_dir = Path(__file__).resolve().parent
    found: dict[str, ModelRecipe] = {}
    for path in sorted(package_dir.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        module = importlib.import_module(_module_name(path, package_dir))
        recipe = getattr(module, "RECIPE", None)
        if recipe is None:
            continue
        if not isinstance(recipe, ModelRecipe):
            raise TypeError(
                f"{module.__name__}.RECIPE deve essere un ModelRecipe, "
                f"non {type(recipe).__name__}."
            )
        if recipe.id in found:
            raise RuntimeError(
                f"Id ricetta duplicato {recipe.id!r} "
                f"({found[recipe.id]!r} e {module.__name__})."
            )
        found[recipe.id] = recipe
    return found


def available_models() -> tuple[str, ...]:
    """Ids of every discovered model recipe, sorted."""
    return tuple(sorted(_registry()))


def load_model(model_id: str) -> ModelRecipe:
    """Load a supported model recipe by id."""
    try:
        return _registry()[model_id]
    except KeyError:
        known = ", ".join(available_models()) or "(nessuna)"
        raise KeyError(
            f"Modello sconosciuto {model_id!r}. Disponibili: {known}."
        ) from None
