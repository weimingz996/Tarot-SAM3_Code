"""Online ReasonSeg candidate generation without final mask selection."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from reason_prompt import reasonseg_best_flow_reference as core
from reason_prompt.full_des_generation import generate as generate_full_description


class VisionQwen(core.VisionQwen):
    """Current reference client with Tarot's in-memory config constructor."""

    @classmethod
    def from_qwen_config(cls, config: Any) -> "VisionQwen":
        return cls(config.base_url, config.api_key, config.model)


def run_reasonseg_candidate_flow(
    *,
    full_des_qwen: Any,
    text_qwen: VisionQwen,
    sam3: Any,
    image_path: str | Path,
    query: str,
    output_dir: str | Path,
    sample: str = "live",
    visualize: bool = False,
) -> dict[str, Any]:
    """Return ReasonSeg mask candidates immediately before final voting."""

    del sample, visualize
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"ReasonSeg candidate output is not empty: {output}")
    engines = core.Engines(
        full_des_qwen=full_des_qwen,
        text_qwen=text_qwen,
        sam3=sam3,
        full_des_api=SimpleNamespace(generate=generate_full_description),
    )
    return core.run_candidate_sample(Path(image_path), query, output, engines)
