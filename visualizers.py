"""Registry of visualizer backends for the Textual UI (and future front-ends)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import waveform

ProgressCallback = Callable[[float, str], None]
RunFn = Callable[..., str]


@dataclass(frozen=True)
class Visualizer:
    id: str
    name: str
    description: str
    config_schema: list[dict[str, Any]]
    get_default_config: Callable[[], dict[str, Any]]
    run: RunFn


def _run_waveform(
    *,
    audio_path: str,
    output_path: str,
    config: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> str:
    return waveform.run(
        audio_path=audio_path,
        output_path=output_path,
        config=config,
        progress_callback=progress_callback,
    )


VISUALIZERS: dict[str, Visualizer] = {
    "waveform": Visualizer(
        id="waveform",
        name="Waveform",
        description="Monstercat-style mel bars with optional matrix rain/mask.",
        config_schema=waveform.CONFIG_SCHEMA,
        get_default_config=waveform.get_default_config,
        run=_run_waveform,
    ),
}


def list_visualizers() -> list[Visualizer]:
    return list(VISUALIZERS.values())


def get_visualizer(visualizer_id: str) -> Visualizer:
    try:
        return VISUALIZERS[visualizer_id]
    except KeyError as exc:
        known = ", ".join(VISUALIZERS)
        raise KeyError(f"Unknown visualizer {visualizer_id!r}. Known: {known}") from exc
