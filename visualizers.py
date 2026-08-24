"""Registry of visualizer backends for the Textual UI (and future front-ends)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from proglog import ProgressBarLogger

import waveform

ProgressCallback = Callable[[float, str], None]
RunFn = Callable[..., str]


# UI-facing schema for Waveform knobs (stays out of waveform.py).
WAVEFORM_CONFIG_SCHEMA: list[dict[str, Any]] = [
    {"key": "W", "label": "Width", "type": "int", "default": 1080, "group": "Canvas"},
    {"key": "H", "label": "Height", "type": "int", "default": 1920, "group": "Canvas"},
    {"key": "FPS", "label": "FPS", "type": "int", "default": 60, "group": "Canvas"},
    {"key": "N_BANDS", "label": "Bands", "type": "int", "default": 18, "group": "Canvas"},
    {"key": "GUTTER", "label": "Gutter (px)", "type": "int", "default": 4, "group": "Canvas"},
    {"key": "MAX_BAR_HEIGHT", "label": "Max bar height", "type": "int", "default": 1650, "group": "Canvas"},
    {"key": "BAR_BOTTOM_MARGIN", "label": "Bottom margin", "type": "int", "default": 0, "group": "Canvas"},
    {"key": "BAR_TOP_MARGIN", "label": "Top margin", "type": "int", "default": 70, "group": "Canvas"},
    {"key": "STATIC_COLOR", "label": "Bar color (RGB)", "type": "color", "default": "255,255,255", "group": "Colors"},
    {
        "key": "GRADIENT_COLOR_BOTTOM",
        "label": "Gradient bottom (RGB)",
        "type": "color",
        "default": "255,231,97",
        "group": "Colors",
    },
    {
        "key": "GRADIENT_COLOR_TOP",
        "label": "Gradient top (RGB)",
        "type": "color",
        "default": "255,96,234",
        "group": "Colors",
    },
    {"key": "USE_MATRIX_MASK", "label": "Static matrix mask", "type": "bool", "default": False, "group": "Matrix"},
    {"key": "USE_MATRIX_RAIN", "label": "Matrix rain", "type": "bool", "default": True, "group": "Matrix"},
    {
        "key": "MATRIX_CHARS_PER_BAR_ROW",
        "label": "Chars per bar row",
        "type": "int",
        "default": 1,
        "group": "Matrix",
    },
    {
        "key": "MATRIX_CELL_HEIGHT_RATIO",
        "label": "Cell height ratio",
        "type": "float",
        "default": 1.0,
        "group": "Matrix",
    },
    {"key": "MATRIX_LUM_CUTOFF", "label": "Lum cutoff", "type": "int", "default": 40, "group": "Matrix"},
    {"key": "MATRIX_FONT_THICKNESS", "label": "Font thickness", "type": "int", "default": 3, "group": "Matrix"},
    {"key": "MATRIX_RAIN_FPS", "label": "Rain FPS", "type": "int", "default": 20, "group": "Matrix"},
    {
        "key": "MATRIX_RAIN_SCREEN_TOP_BRIGHTNESS",
        "label": "Rain top brightness",
        "type": "int",
        "default": 96,
        "group": "Matrix",
    },
    {
        "key": "MATRIX_RAIN_SCREEN_BOTTOM_BRIGHTNESS",
        "label": "Rain bottom brightness",
        "type": "int",
        "default": 255,
        "group": "Matrix",
    },
    {
        "key": "MATRIX_RAIN_CHAR_MUTATION",
        "label": "Rain char mutation",
        "type": "float",
        "default": 0.125,
        "group": "Matrix",
    },
    {
        "key": "MATRIX_SEED",
        "label": "Matrix seed (empty=random)",
        "type": "optional_int",
        "default": "",
        "group": "Matrix",
    },
]


def waveform_defaults() -> dict[str, Any]:
    return {field["key"]: field["default"] for field in WAVEFORM_CONFIG_SCHEMA}


def _parse_color(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [int(value[0]), int(value[1]), int(value[2])]
    parts = [p.strip() for p in str(value).replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"Color must be R,G,B — got {value!r}")
    return [int(parts[0]), int(parts[1]), int(parts[2])]


def _coerce_waveform_config(config: dict[str, Any]) -> dict[str, Any]:
    """Turn UI form values into waveform.run(**overrides) kwargs."""
    out: dict[str, Any] = {}
    types = {f["key"]: f["type"] for f in WAVEFORM_CONFIG_SCHEMA}
    for key, raw in config.items():
        ftype = types.get(key)
        if ftype == "color":
            out[key] = _parse_color(raw)
        elif ftype == "optional_int":
            out[key] = None if raw is None or raw == "" else int(raw)
        elif ftype == "bool":
            out[key] = bool(raw)
        elif ftype == "int":
            out[key] = int(raw)
        elif ftype == "float":
            out[key] = float(raw)
        else:
            out[key] = raw
    if out.get("USE_MATRIX_MASK") and out.get("USE_MATRIX_RAIN"):
        out["USE_MATRIX_MASK"] = False
    return out


class UIProgressLogger(ProgressBarLogger):
    """MoviePy/proglog → UI progress_callback(fraction, message)."""

    def __init__(self, callback: ProgressCallback):
        super().__init__()
        self._callback = callback
        self._msg = "Rendering"
        self._phase = 0

    def callback(self, **changes):
        message = changes.get("message")
        if not message:
            return
        self._msg = str(message)
        if "writing video" in self._msg.lower():
            self._phase = 1
        self._callback(-1.0, self._msg)

    def bars_callback(self, bar, attr, value, old_value=None):
        if attr != "index":
            return
        total = self.bars.get(bar, {}).get("total") or 0
        if total <= 0:
            return
        local = max(0.0, min(1.0, float(value) / float(total)))
        if self._phase == 0 or bar == "chunk":
            fraction = 0.10 + 0.20 * local
        else:
            fraction = 0.30 + 0.69 * local
        self._callback(fraction, self._msg)


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
    overrides = _coerce_waveform_config(config)
    logger: Any = "bar"
    if progress_callback is not None:
        logger = UIProgressLogger(progress_callback)
    return waveform.run(
        audio_path=audio_path,
        output_path=output_path,
        logger=logger,
        progress_callback=progress_callback,
        **overrides,
    )


VISUALIZERS: dict[str, Visualizer] = {
    "waveform": Visualizer(
        id="waveform",
        name="Waveform",
        description="Monstercat-style mel bars with optional matrix rain/mask.",
        config_schema=WAVEFORM_CONFIG_SCHEMA,
        get_default_config=waveform_defaults,
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
