from math import ceil
from typing import Any, Callable
import os
import sys
import cv2
import numpy as np
import librosa
from moviepy import VideoClip, AudioFileClip
from scipy.ndimage import gaussian_filter1d

try:
    from proglog import ProgressBarLogger
except ImportError:  # pragma: no cover - moviepy normally pulls this in
    ProgressBarLogger = None  # type: ignore[misc, assignment]

DEBUG_BAR_HEIGHT = os.environ.get("DEBUG_BAR_HEIGHT", "").lower() in ("1", "true", "yes")
DEBUG_BARS_ONLY = DEBUG_BAR_HEIGHT or "--debug-bars" in sys.argv
DEBUG_MATRIX_ONLY = "--debug-matrix" in sys.argv
DEBUG_DIR = "debug"
DEBUG_FLAG_BARS = "bars"
DEBUG_FLAG_MATRIX = "matrix"


def debug_out(flag: str, filename: str) -> str:
    """Write path under debug/{flag}/; creates the directory if needed."""
    directory = os.path.join(DEBUG_DIR, flag)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


def log_section(title: str) -> None:
    print(f"\n[{title}]")


def log_item(key: str, value: object) -> None:
    print(f"  {key:<26} {value}")


def log_matrix_mask(
    *,
    enabled: bool,
    mode: str = "static",
    glyphs: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
    layout_seed: int | None = None,
    rain_ticks: int | None = None,
) -> None:
    """Single config block for matrix settings and (optional) build results."""
    log_section("Matrix mask")
    log_item("Enabled", enabled)
    if not enabled:
        return
    log_item("Mode", mode)
    log_item("Cell size", f"{MATRIX_CELL_W}x{MATRIX_CELL_H}px")
    log_item("Grid density", f"~{W // MATRIX_CELL_W}x{H // MATRIX_CELL_H} cells")
    log_item("Chars per bar row", MATRIX_CHARS_PER_BAR_ROW)
    log_item("Cell height ratio", MATRIX_CELL_HEIGHT_RATIO)
    log_item("Font thickness", MATRIX_FONT_THICKNESS)
    log_item("Charset size", len(MATRIX_CHARSET))
    log_item("Lum cutoff", MATRIX_LUM_CUTOFF)
    log_item("Seed", MATRIX_SEED if MATRIX_SEED is not None else "random per run")
    if mode == "rain":
        log_item("Rain FPS", MATRIX_RAIN_FPS)
        log_item("Rain mode", "continuous (head every tick)")
        log_item(
            "Screen fade (top->bottom)",
            f"{MATRIX_RAIN_SCREEN_TOP_BRIGHTNESS} -> {MATRIX_RAIN_SCREEN_BOTTOM_BRIGHTNESS}",
        )
        log_item("Char mutation", MATRIX_RAIN_CHAR_MUTATION)
        if rain_ticks is not None:
            log_item("Rain ticks", rain_ticks)
    if glyphs is not None:
        log_item("Glyphs loaded", glyphs)
    if scale_min is not None and scale_max is not None:
        log_item("Font scale range", f"{scale_min:.2f} - {scale_max:.2f}")
    if layout_seed is not None:
        log_item("Layout seed", layout_seed)


#### CONFIGURATION ############################################################

# Defaults — also used by the Textual UI schema. Mutated via apply_config().
W, H = 1080, 1920
N_BANDS = 18
GUTTER = 4  # px gap between bars
# Bar span in px when amp=1.0 (after peak normalization). NOT including BAR_BOTTOM_MARGIN.
MAX_BAR_HEIGHT = 1650
# Lift baseline off frame bottom; adds to "reach from bottom" (reach = margin + MAX_BAR_HEIGHT).
BAR_BOTTOM_MARGIN = 0
BAR_TOP_MARGIN = 70  # minimum y index for bar top at full amplitude

FPS = 60
AUDIO_PATH = "unused_promo_wav/when_david_heard_monstercat_promo.wav"
OUTPUT_PATH = "output_waveform.mov"
STATIC_COLOR = [255, 255, 255]
GRADIENT_COLOR_BOTTOM = [255, 231, 97]
GRADIENT_COLOR_TOP = [255, 96, 234]

# Mel / signal-chain knobs (formerly hard-coded mid-script)
FMIN, FMAX = 20, 7000
EXPONENT = 2.6
TILT_MIN, TILT_MAX = 0.8, 2.2
GRAVITY_ATTACK, GRAVITY_MAX_DECAY, GRAVITY_MIN_DECAY = 0.5, 0.15, 0.05
GAUSSIAN_FILTER = 1.0
SPECTRAL_DELAY_MAX_FRAMES = 4
USE_SPECTRAL_DELAY = True
USE_WASH_DELAY = True

# Matrix ASCII mask (static grid, CMatrix-inspired charset)
# Fewer on-screen glyphs → larger cells: use MATRIX_CHARS_PER_BAR_ROW=1 and/or
# raise MATRIX_CELL_HEIGHT_RATIO (taller rows). Grid math picks up the rest.
MATRIX_CHARS_PER_BAR_ROW = 1
MATRIX_CELL_HEIGHT_RATIO = 1
MATRIX_LUM_CUTOFF = 40  # lum 0–255 below this → black (no gradient bleed in gaps)
# Hershey fonts only draw ASCII reliably; katakana tiles are usually blank in OpenCV.
MATRIX_CHARSET = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "@#$%&*+=<>?/|\\{}[]"
)
MATRIX_SEED = None  # None = random layout each run; set an int (e.g. 0) to fix the pattern
MATRIX_FONT = cv2.FONT_HERSHEY_SIMPLEX
MATRIX_FONT_THICKNESS = 3  # OpenCV putText stroke weight (1=thin, 2+=bolder)
MATRIX_GLYPH_PAD = 0  # tight fit; glyphs may touch cell edges
# Dynamic rain (CMatrix-style falling columns)
MATRIX_RAIN_FPS = 20
# Screen-space glyph brightness by y (0–255): dim toward top, full at bottom.
MATRIX_RAIN_SCREEN_TOP_BRIGHTNESS = 96
MATRIX_RAIN_SCREEN_BOTTOM_BRIGHTNESS = 255
MATRIX_RAIN_CHAR_MUTATION = 0.125
USE_MATRIX_MASK = False
USE_MATRIX_RAIN = True

# Derived geometry (recomputed in apply_config / _recompute_derived)
BAR_WIDTH = (W - (N_BANDS * GUTTER)) / N_BANDS
BAR_BASE_Y = H - 1 - BAR_BOTTOM_MARGIN
MAX_DRAWABLE_HEIGHT = BAR_BASE_Y - BAR_TOP_MARGIN
PEAK_BAR_TOP_Y = BAR_BASE_Y - MAX_BAR_HEIGHT
PEAK_REACH_FROM_FRAME_BOTTOM = (H - 1) - PEAK_BAR_TOP_Y
_BAR_SLOT_W = int(BAR_WIDTH + GUTTER)
MATRIX_CELL_W = max(1, _BAR_SLOT_W // MATRIX_CHARS_PER_BAR_ROW)
MATRIX_CELL_H = max(8, int(MATRIX_CELL_W * MATRIX_CELL_HEIGHT_RATIO))

# Runtime state filled by run()
y = None
sr = None
audio_clip = None
duration = 0.0
S_norm = None
fmin, fmax = FMIN, FMAX
exponent = EXPONENT
tilt_min, tilt_max = TILT_MIN, TILT_MAX
gravity_attack = GRAVITY_ATTACK
gravity_max_decay = GRAVITY_MAX_DECAY
gravity_min_decay = GRAVITY_MIN_DECAY
gaussian_filter = GAUSSIAN_FILTER

# Schema for the Textual UI (and any future front-ends).
# type: int | float | bool | str | color (RGB "R,G,B") | optional_int (empty = None)
CONFIG_SCHEMA: list[dict[str, Any]] = [
    {"key": "W", "label": "Width", "type": "int", "default": 1080, "group": "Canvas"},
    {"key": "H", "label": "Height", "type": "int", "default": 1920, "group": "Canvas"},
    {"key": "FPS", "label": "FPS", "type": "int", "default": 60, "group": "Canvas"},
    {"key": "N_BANDS", "label": "Bands", "type": "int", "default": 18, "group": "Canvas"},
    {"key": "GUTTER", "label": "Gutter (px)", "type": "int", "default": 4, "group": "Canvas"},
    {"key": "MAX_BAR_HEIGHT", "label": "Max bar height", "type": "int", "default": 1650, "group": "Canvas"},
    {"key": "BAR_BOTTOM_MARGIN", "label": "Bottom margin", "type": "int", "default": 0, "group": "Canvas"},
    {"key": "BAR_TOP_MARGIN", "label": "Top margin", "type": "int", "default": 70, "group": "Canvas"},
    {
        "key": "STATIC_COLOR",
        "label": "Bar color (RGB)",
        "type": "color",
        "default": "255,255,255",
        "group": "Colors",
    },
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
    {"key": "FMIN", "label": "Mel fmin (Hz)", "type": "int", "default": 20, "group": "Signal"},
    {"key": "FMAX", "label": "Mel fmax (Hz)", "type": "int", "default": 7000, "group": "Signal"},
    {"key": "EXPONENT", "label": "Exponent", "type": "float", "default": 2.6, "group": "Signal"},
    {"key": "TILT_MIN", "label": "Tilt min", "type": "float", "default": 0.8, "group": "Signal"},
    {"key": "TILT_MAX", "label": "Tilt max", "type": "float", "default": 2.2, "group": "Signal"},
    {"key": "GRAVITY_ATTACK", "label": "Gravity attack", "type": "float", "default": 0.5, "group": "Signal"},
    {"key": "GRAVITY_MAX_DECAY", "label": "Gravity max decay", "type": "float", "default": 0.15, "group": "Signal"},
    {"key": "GRAVITY_MIN_DECAY", "label": "Gravity min decay", "type": "float", "default": 0.05, "group": "Signal"},
    {"key": "GAUSSIAN_FILTER", "label": "Gaussian sigma", "type": "float", "default": 1.0, "group": "Signal"},
    {
        "key": "SPECTRAL_DELAY_MAX_FRAMES",
        "label": "Spectral delay frames",
        "type": "int",
        "default": 4,
        "group": "Signal",
    },
    {"key": "USE_SPECTRAL_DELAY", "label": "Spectral delay", "type": "bool", "default": True, "group": "Signal"},
    {"key": "USE_WASH_DELAY", "label": "Wash delay", "type": "bool", "default": True, "group": "Signal"},
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


def get_default_config() -> dict[str, Any]:
    """Return a fresh copy of UI-facing defaults from CONFIG_SCHEMA."""
    return {field["key"]: field["default"] for field in CONFIG_SCHEMA}


def _parse_color(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [int(value[0]), int(value[1]), int(value[2])]
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(" ", "").split(",")]
        if len(parts) != 3:
            raise ValueError(f"Color must be R,G,B — got {value!r}")
        return [int(parts[0]), int(parts[1]), int(parts[2])]
    raise ValueError(f"Invalid color value: {value!r}")


def _recompute_derived() -> None:
    """Refresh geometry derived from canvas / matrix knobs."""
    global BAR_WIDTH, BAR_BASE_Y, MAX_DRAWABLE_HEIGHT, PEAK_BAR_TOP_Y
    global PEAK_REACH_FROM_FRAME_BOTTOM, _BAR_SLOT_W, MATRIX_CELL_W, MATRIX_CELL_H

    BAR_WIDTH = (W - (N_BANDS * GUTTER)) / N_BANDS
    BAR_BASE_Y = H - 1 - BAR_BOTTOM_MARGIN
    MAX_DRAWABLE_HEIGHT = BAR_BASE_Y - BAR_TOP_MARGIN
    PEAK_BAR_TOP_Y = BAR_BASE_Y - MAX_BAR_HEIGHT
    PEAK_REACH_FROM_FRAME_BOTTOM = (H - 1) - PEAK_BAR_TOP_Y
    _BAR_SLOT_W = int(BAR_WIDTH + GUTTER)
    MATRIX_CELL_W = max(1, _BAR_SLOT_W // MATRIX_CHARS_PER_BAR_ROW)
    MATRIX_CELL_H = max(8, int(MATRIX_CELL_W * MATRIX_CELL_HEIGHT_RATIO))

    if MAX_BAR_HEIGHT > MAX_DRAWABLE_HEIGHT:
        raise ValueError(
            f"MAX_BAR_HEIGHT ({MAX_BAR_HEIGHT}) exceeds drawable range "
            f"({MAX_DRAWABLE_HEIGHT}): BAR_BASE_Y={BAR_BASE_Y}, BAR_TOP_MARGIN={BAR_TOP_MARGIN}"
        )


def apply_config(config: dict[str, Any] | None = None) -> None:
    """Apply a config dict (from UI or CLI) onto module-level knobs."""
    global W, H, N_BANDS, GUTTER, MAX_BAR_HEIGHT, BAR_BOTTOM_MARGIN, BAR_TOP_MARGIN
    global FPS, AUDIO_PATH, OUTPUT_PATH, STATIC_COLOR, GRADIENT_COLOR_BOTTOM, GRADIENT_COLOR_TOP
    global FMIN, FMAX, EXPONENT, TILT_MIN, TILT_MAX
    global GRAVITY_ATTACK, GRAVITY_MAX_DECAY, GRAVITY_MIN_DECAY, GAUSSIAN_FILTER
    global SPECTRAL_DELAY_MAX_FRAMES, USE_SPECTRAL_DELAY, USE_WASH_DELAY
    global MATRIX_CHARS_PER_BAR_ROW, MATRIX_CELL_HEIGHT_RATIO, MATRIX_LUM_CUTOFF
    global MATRIX_SEED, MATRIX_FONT_THICKNESS, MATRIX_GLYPH_PAD
    global MATRIX_RAIN_FPS, MATRIX_RAIN_SCREEN_TOP_BRIGHTNESS
    global MATRIX_RAIN_SCREEN_BOTTOM_BRIGHTNESS, MATRIX_RAIN_CHAR_MUTATION
    global USE_MATRIX_MASK, USE_MATRIX_RAIN

    cfg = get_default_config()
    if config:
        cfg.update(config)

    W = int(cfg["W"])
    H = int(cfg["H"])
    N_BANDS = int(cfg["N_BANDS"])
    GUTTER = int(cfg["GUTTER"])
    MAX_BAR_HEIGHT = int(cfg["MAX_BAR_HEIGHT"])
    BAR_BOTTOM_MARGIN = int(cfg["BAR_BOTTOM_MARGIN"])
    BAR_TOP_MARGIN = int(cfg["BAR_TOP_MARGIN"])
    FPS = int(cfg["FPS"])
    STATIC_COLOR = _parse_color(cfg["STATIC_COLOR"])
    GRADIENT_COLOR_BOTTOM = _parse_color(cfg["GRADIENT_COLOR_BOTTOM"])
    GRADIENT_COLOR_TOP = _parse_color(cfg["GRADIENT_COLOR_TOP"])
    FMIN = int(cfg["FMIN"])
    FMAX = int(cfg["FMAX"])
    EXPONENT = float(cfg["EXPONENT"])
    TILT_MIN = float(cfg["TILT_MIN"])
    TILT_MAX = float(cfg["TILT_MAX"])
    GRAVITY_ATTACK = float(cfg["GRAVITY_ATTACK"])
    GRAVITY_MAX_DECAY = float(cfg["GRAVITY_MAX_DECAY"])
    GRAVITY_MIN_DECAY = float(cfg["GRAVITY_MIN_DECAY"])
    GAUSSIAN_FILTER = float(cfg["GAUSSIAN_FILTER"])
    SPECTRAL_DELAY_MAX_FRAMES = int(cfg["SPECTRAL_DELAY_MAX_FRAMES"])
    USE_SPECTRAL_DELAY = bool(cfg["USE_SPECTRAL_DELAY"])
    USE_WASH_DELAY = bool(cfg["USE_WASH_DELAY"])
    MATRIX_CHARS_PER_BAR_ROW = int(cfg["MATRIX_CHARS_PER_BAR_ROW"])
    MATRIX_CELL_HEIGHT_RATIO = float(cfg["MATRIX_CELL_HEIGHT_RATIO"])
    MATRIX_LUM_CUTOFF = int(cfg["MATRIX_LUM_CUTOFF"])
    MATRIX_FONT_THICKNESS = int(cfg["MATRIX_FONT_THICKNESS"])
    MATRIX_RAIN_FPS = int(cfg["MATRIX_RAIN_FPS"])
    MATRIX_RAIN_SCREEN_TOP_BRIGHTNESS = int(cfg["MATRIX_RAIN_SCREEN_TOP_BRIGHTNESS"])
    MATRIX_RAIN_SCREEN_BOTTOM_BRIGHTNESS = int(cfg["MATRIX_RAIN_SCREEN_BOTTOM_BRIGHTNESS"])
    MATRIX_RAIN_CHAR_MUTATION = float(cfg["MATRIX_RAIN_CHAR_MUTATION"])
    USE_MATRIX_MASK = bool(cfg["USE_MATRIX_MASK"])
    USE_MATRIX_RAIN = bool(cfg["USE_MATRIX_RAIN"])

    seed_raw = cfg.get("MATRIX_SEED", "")
    if seed_raw is None or seed_raw == "":
        MATRIX_SEED = None
    else:
        MATRIX_SEED = int(seed_raw)

    if "AUDIO_PATH" in (config or {}):
        AUDIO_PATH = str(config["AUDIO_PATH"])  # type: ignore[index]
    if "OUTPUT_PATH" in (config or {}):
        OUTPUT_PATH = str(config["OUTPUT_PATH"])  # type: ignore[index]

    if USE_MATRIX_MASK and USE_MATRIX_RAIN:
        # Prefer rain when both are somehow enabled.
        USE_MATRIX_MASK = False

    _recompute_derived()


_recompute_derived()

### WAVEFORM PRE-PROCESSING FUNCTIONS #########################################

def gravity(S_norm: np.ndarray, attack: float=0.5, max_decay: float=0.15, min_decay: float=0.05) -> np.ndarray:
    """
    Lessens amplitude variation between frames based on above coefficients.
    The attack and decay rates can be adjusted to control the speed of the effect for amplitude increase/decrease.
    Decay gradient is created from max and min decay to change decay rate PER band.

    Same deal with rolling average, this can strip detail from the waveform.
    """
    smoothed_S = np.zeros_like(S_norm)
    decay_gradient = np.linspace(min_decay, max_decay, num=N_BANDS)

    for t in range(1, S_norm.shape[1]):
        target = S_norm[:, t]
        prev = smoothed_S[:, t-1]
        
        smoothed_S[:, t] = np.where(target > prev, 
                                    prev + (target - prev) * attack, 
                                    prev - (prev - target) * decay_gradient)
    return smoothed_S


def rolling_average(data: np.ndarray, window_size: int=3) -> np.ndarray:
    """
    Applies a rolling average to the signal data, modifying values in place based on the
    neighboring values within the window size.
    
    In practice, this makes the waveform a whole lot less appealing IMO...
    """
    kernel = np.ones(window_size) / window_size
    for i in range (data.shape[0]):
        data[i, :] = np.convolve(data[i, :], kernel, mode='same')
    return data


def wash_delay(S_norm:np.ndarray) -> np.array:
    """
    Creates a visual effect by delaying the energy from treble/mid to bass, so
    values moves tO THE LEFT .
    """
    for t in range(1, S_norm.shape[1]):
        for b in range(1, S_norm.shape[0]):
            # Energy 'leaks' from right to left (treble/mid to bass)
            # This creates the visual 'wash'
            S_norm[b-1, t] += S_norm[b, t-1] * 0.1
    return S_norm


def spectral_delay(S_norm: np.ndarray, max_delay_frames: int=4) -> np.ndarray:
    """
    Wave-like delay effect that gets us closer to that rolling kick that Monstercat uses.
    """
    n_bands, _ = S_norm.shape
    delayed_data = np.zeros_like(S_norm)
    impact_idx = 5
    
    for b in range(n_bands):
        # Calculate delay based on distance from impact_idx
        dist = abs(b - impact_idx)
        delay = int((dist**1.2) * (max_delay_frames / n_bands))
        
        if delay > 0:
            # Shift the data forward in time by 'delay' frames
            delayed_data[b, delay:] = S_norm[b, :-delay]
        else:
            delayed_data[b, :] = S_norm[b, :]
            
    return delayed_data



### AUDIO SIGNAL CHAIN ###################################################

def build_signal_chain(audio_path: str) -> float:
    """
    Load audio, build the mel / FX chain, and stash results on module globals
    used by make_frame / amplitudes_at_time. Returns pre-peak max for logging.
    """
    global y, sr, audio_clip, duration, S_norm
    global fmin, fmax, exponent, tilt_min, tilt_max
    global gravity_attack, gravity_max_decay, gravity_min_decay, gaussian_filter

    y, sr = librosa.load(audio_path)
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration

    fmin, fmax = FMIN, FMAX
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_BANDS, fmin=fmin, fmax=fmax)

    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())

    if USE_SPECTRAL_DELAY:
        S_norm = spectral_delay(S_norm, max_delay_frames=SPECTRAL_DELAY_MAX_FRAMES)

    exponent = EXPONENT
    S_norm = np.power(S_norm, exponent)

    tilt_min, tilt_max = TILT_MIN, TILT_MAX
    tilt = np.linspace(tilt_min, tilt_max, N_BANDS)
    S_norm = (S_norm.T * tilt).T

    gravity_attack, gravity_max_decay, gravity_min_decay = (
        GRAVITY_ATTACK,
        GRAVITY_MAX_DECAY,
        GRAVITY_MIN_DECAY,
    )
    S_norm = gravity(S_norm, gravity_attack, gravity_max_decay, gravity_min_decay)

    if USE_WASH_DELAY:
        S_norm = wash_delay(S_norm)

    gaussian_filter = GAUSSIAN_FILTER
    S_norm = gaussian_filter1d(S_norm, sigma=gaussian_filter, axis=0)

    peak = float(S_norm.max())
    if peak > 0:
        S_norm = S_norm / peak
    return peak


def log_run_config(pre_peak: float, audio_path: str) -> None:
    """Print grouped startup config after the signal chain is ready."""
    log_section("Environment")
    log_item("OpenCV", cv2.__version__)
    if hasattr(cv2, "LINE_AA"):
        log_item("Anti-aliasing", "LINE_AA ready")
    else:
        log_item("Anti-aliasing", "LINE_AA not found")

    log_section("Canvas & bars")
    log_item("Resolution", f"{W} x {H}")
    log_item("FPS", FPS)
    log_item("Bands / gutter", f"{N_BANDS} / {GUTTER}px")
    log_item("Bar width", f"{BAR_WIDTH:.1f}px")
    log_item("Max bar height", f"{MAX_BAR_HEIGHT}px")
    log_item("Bar baseline Y", BAR_BASE_Y)
    log_item("Top margin", f"{BAR_TOP_MARGIN}px")
    log_item("At amp=1.0", f"top y={PEAK_BAR_TOP_Y}, reach from bottom={PEAK_REACH_FROM_FRAME_BOTTOM}px")

    log_section("Audio")
    log_item("Source", audio_path)
    log_item("Duration", f"{duration:.2f}s")
    log_item("Mel bands", N_BANDS)
    log_item("Mel range", f"{fmin}-{fmax} Hz")

    log_section("Signal processing")
    log_item("Spectral delay", "on" if USE_SPECTRAL_DELAY else "off")
    log_item("Exponent", exponent)
    log_item("Tilt EQ", f"{tilt_min} - {tilt_max}")
    log_item(
        "Gravity",
        f"attack={gravity_attack}, decay={gravity_min_decay}-{gravity_max_decay}",
    )
    log_item("Wash delay", "on" if USE_WASH_DELAY else "off")
    log_item("Gaussian blur", f"sigma={gaussian_filter}")
    log_item("Peak normalize", f"pre-scale max={pre_peak:.4f}")


def amplitudes_at_time(t: float) -> np.ndarray:
    float_idx = (t / duration) * (S_norm.shape[1] - 1)
    idx_floor = int(np.floor(float_idx))
    idx_ceil = int(np.ceil(float_idx))
    weight = float_idx - idx_floor
    return (1 - weight) * S_norm[:, idx_floor] + weight * S_norm[:, idx_ceil]


def measure_mask_bar_extent(mask: np.ndarray) -> dict[str, int]:
    ys = np.where(mask[:, :, 0] > 0)[0]
    if ys.size == 0:
        return {"top": -1, "bottom": -1, "span": 0}
    top, bottom = int(ys.min()), int(ys.max())
    return {"top": top, "bottom": bottom, "span": bottom - top + 1}


### WAVEFORM RENDERING FUNCTIONS ###############################################

def amplitude_to_bar_height(amp: float) -> float:
    """Map normalized amplitude [0, 1] to bar height in pixels."""
    return float(np.clip(amp, 0.0, 1.0)) * MAX_BAR_HEIGHT


def bar_y_bounds(bar_height: float) -> tuple[int, int]:
    """Return (y_bottom, y_top) in image coordinates (y grows downward)."""
    y_bottom = BAR_BASE_Y
    y_top = max(BAR_TOP_MARGIN, y_bottom - bar_height)
    return int(round(y_bottom)), int(round(y_top))


def bar_x_bounds(band_index: int) -> tuple[float, float]:
    """Return (x_left, x_right) for a band index."""
    x_left = band_index * (BAR_WIDTH + GUTTER)
    return x_left, x_left + BAR_WIDTH


def draw_rounded_bars(frame: np.ndarray, amplitudes: np.ndarray) -> None:
    """
    Draws bar with rounded corners at designated coordinates.
    Handles curve logic for the rounded corners with given radius.
    """

    corner_radius = 0.1  # 10% rounding
    radius = int(BAR_WIDTH * corner_radius)
    print(f"--- bar rounded corner radius: [{corner_radius}]")


    bar_polygons = []
    for band_idx, amp in enumerate[Any](amplitudes):
        bar_height = amplitude_to_bar_height(amp)
        x1, x2 = bar_x_bounds(band_idx)
        y1, y2 = bar_y_bounds(bar_height)

        # Ensure radius isn't larger than half the bar width
        bar_width = abs(x2 - x1)
        radius = min(radius, bar_width // 2)
        
        points = []
        points.append([x2, y1]) # bottom right
        points.append([x1, y1]) # bottom left
        points.append([x1, y2 + radius]) # left vertical side up to the start of the curve
        
        # top left curve
        for deg in range(180, 270, 10): # 10-degree steps for smoothness
            angle = np.radians(deg)
            px = x1 + radius + radius * np.cos(angle)
            py = y2 + radius + radius * np.sin(angle)
            points.append([px, py])
            
        # top right curve
        for deg in range(270, 360, 10):
            angle = np.radians(deg)
            px = x2 - radius + radius * np.cos(angle)
            py = y2 + radius + radius * np.sin(angle)
            points.append([px, py])
            
        points.append([x2, y2 + radius]) # right vertical side down
        bar_polygons.append(np.array(points, dtype=np.int32))

    # Draw all bars in one step.
    cv2.fillPoly(frame, bar_polygons, STATIC_COLOR, lineType=cv2.LINE_AA)


def draw_bars(frame: np.ndarray, amplitudes: np.ndarray) -> None:
    bar_polygons = []
    for band_idx, amp in enumerate[Any](amplitudes):
        bar_height = amplitude_to_bar_height(amp)
        x1, x2 = bar_x_bounds(band_idx)
        y1, y2 = bar_y_bounds(bar_height)

        bot_left_pt = (int(round(x1)), y1)
        top_right_pt = (int(round(x2)), y2)
        bot_right_pt = (int(round(x2)), y1)
        top_left_pt = (int(round(x1)), y2)

        # Use polygons for LINE_AA anti-aliasing (spatial smoothing).
        rect_points = np.array([bot_left_pt, bot_right_pt, top_right_pt, top_left_pt], dtype=np.int32)

        bar_polygons.append(rect_points)

    # Draw all bars in one step.
    cv2.fillPoly(frame, bar_polygons, STATIC_COLOR, lineType=cv2.LINE_AA)


def apply_bloom(frame: np.ndarray, layers: int=4, base_ksize: int=31, intensity: float=0.45) -> np.ndarray:
    """    
    layers      -- number of blur passes (each doubles the kernel size)
    base_ksize  -- starting kernel width (must be odd)
    intensity   -- strength multiplier per layer (decays with each pass)
    """
    bloom = frame.astype(np.float32)
    result = bloom.copy()

    for i in range(layers):
        ksize = base_ksize + i * 2 * base_ksize  # 31, 93, 155, 217 …
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        blurred = cv2.GaussianBlur(bloom, (ksize, ksize), 0)
        weight = intensity / (i + 1)
        result += blurred * weight

    return np.clip(result, 0, 255).astype(np.uint8)


def _glyph_scale_for_char(ch: str) -> tuple[float, int]:
    """Largest scale for this character to fill one matrix cell."""
    inner_w = MATRIX_CELL_W - 2 * MATRIX_GLYPH_PAD
    inner_h = MATRIX_CELL_H - 2 * MATRIX_GLYPH_PAD
    thickness = MATRIX_FONT_THICKNESS
    scale = 2.0
    while scale >= 0.15:
        (tw, th), _baseline = cv2.getTextSize(ch, MATRIX_FONT, scale, thickness)
        if tw <= inner_w and th <= inner_h:
            return scale, thickness
        scale -= 0.02
    return 0.2, thickness


def _draw_glyph_tile(ch: str, scale: float, thickness: int) -> np.ndarray:
    tile = np.zeros((MATRIX_CELL_H, MATRIX_CELL_W), dtype=np.uint8)
    (tw, th), baseline = cv2.getTextSize(ch, MATRIX_FONT, scale, thickness)
    x = MATRIX_GLYPH_PAD + max(0, (MATRIX_CELL_W - 2 * MATRIX_GLYPH_PAD - tw) // 2)
    y = MATRIX_CELL_H - MATRIX_GLYPH_PAD - baseline
    cv2.putText(tile, ch, (x, y), MATRIX_FONT, scale, 255, thickness, cv2.LINE_8)
    return tile


def build_matrix_glyph_atlas(charset: str) -> tuple[np.ndarray, str, float, float]:
    """One luminance tile per renderable charset entry; each glyph scaled to fill its cell."""
    tiles: list[np.ndarray] = []
    renderable: list[str] = []
    scales: list[float] = []
    for ch in charset:
        scale, thickness = _glyph_scale_for_char(ch)
        tile = _draw_glyph_tile(ch, scale, thickness)
        if tile.max() > 0:
            tiles.append(tile)
            renderable.append(ch)
            scales.append(scale)
    if not tiles:
        raise ValueError("MATRIX_CHARSET has no glyphs OpenCV can render")
    return np.stack(tiles, axis=0), "".join(renderable), min(scales), max(scales)


def _matrix_grid_size() -> tuple[int, int]:
    grid_w = (W + MATRIX_CELL_W - 1) // MATRIX_CELL_W
    grid_h = (H + MATRIX_CELL_H - 1) // MATRIX_CELL_H
    return grid_w, grid_h


def _build_matrix_rain_screen_ramp() -> np.ndarray:
    """Per-pixel y multiplier, shape (H, 1). Linear: top=SCREEN_TOP, bottom=SCREEN_BOTTOM."""
    y = np.arange(H, dtype=np.float32)
    t = y / max(H - 1, 1)
    ramp = (
        MATRIX_RAIN_SCREEN_TOP_BRIGHTNESS
        + t * (MATRIX_RAIN_SCREEN_BOTTOM_BRIGHTNESS - MATRIX_RAIN_SCREEN_TOP_BRIGHTNESS)
    )
    return ramp.astype(np.uint8)[:, np.newaxis]


_MATRIX_RAIN_SCREEN_RAMP: np.ndarray | None = None


def _matrix_rain_screen_ramp() -> np.ndarray:
    global _MATRIX_RAIN_SCREEN_RAMP
    if _MATRIX_RAIN_SCREEN_RAMP is None:
        _MATRIX_RAIN_SCREEN_RAMP = _build_matrix_rain_screen_ramp()
    return _MATRIX_RAIN_SCREEN_RAMP


def compose_lum_from_grid(
    atlas: np.ndarray,
    char_grid: np.ndarray,
    bright_grid: np.ndarray,
) -> np.ndarray:
    """Atlas blit: glyph stroke lum = tile * bright_cell (255=active, 0=empty)."""
    grid_h, grid_w = char_grid.shape
    tiles = atlas[char_grid.astype(np.intp)]
    tiles = (
        tiles.astype(np.uint16) * bright_grid[..., None, None] // 255
    ).astype(np.uint8)
    lum = tiles.transpose(0, 2, 1, 3).reshape(
        grid_h * MATRIX_CELL_H, grid_w * MATRIX_CELL_W
    )
    return lum[:H, :W]


def apply_matrix_rain_screen_ramp(lum: np.ndarray) -> np.ndarray:
    """Apply screen-vertical brightness once, after glyph lum is composed."""
    ramp = _matrix_rain_screen_ramp()
    return (lum.astype(np.uint16) * ramp // 255).astype(np.uint8)


def _modulate_frame_with_lum(
    frame: np.ndarray,
    mask: np.ndarray,
    lum: np.ndarray,
    *,
    use_lum_cutoff: bool = True,
) -> np.ndarray:
    """
    Multiply bar pixels by glyph luminance; gaps stay black.
    Static uses MATRIX_LUM_CUTOFF to drop anti-aliased bleed. Rain uses lum>0 only
    so screen-faded (dim) strokes are not clipped.
    """
    bar = mask[:, :, 0] > 0
    glyph = lum.astype(np.float32) / 255.0
    if use_lum_cutoff:
        glyph = np.where(lum >= MATRIX_LUM_CUTOFF, glyph, 0.0)
    else:
        glyph = np.where(lum > 0, glyph, 0.0)
    factor = glyph[:, :, np.newaxis]
    out = frame.astype(np.float32)
    out[bar] = out[bar] * factor[bar]
    return np.clip(out, 0, 255).astype(np.uint8)


def build_matrix_glyph_field() -> np.ndarray:
    """Full-frame static luminance field: random char per grid cell."""
    atlas, charset, scale_min, scale_max = build_matrix_glyph_atlas(MATRIX_CHARSET)
    grid_w, grid_h = _matrix_grid_size()
    seed = MATRIX_SEED if MATRIX_SEED is not None else int(np.random.default_rng().integers(0, 2**32))
    rng = np.random.default_rng(seed)
    char_grid = rng.integers(0, len(charset), (grid_h, grid_w))
    bright_grid = np.full((grid_h, grid_w), 255, dtype=np.uint8)
    lum = compose_lum_from_grid(atlas, char_grid, bright_grid)
    log_matrix_mask(
        enabled=True,
        mode="static",
        glyphs=len(charset),
        scale_min=scale_min,
        scale_max=scale_max,
        layout_seed=seed,
    )
    return lum


GLOBAL_MATRIX_GLYPH_LUM: np.ndarray | None = None


def _ensure_matrix_glyph_field() -> np.ndarray:
    """Build glyph field on first apply_matrix_mask call (lazy, once per run)."""
    global GLOBAL_MATRIX_GLYPH_LUM
    if GLOBAL_MATRIX_GLYPH_LUM is None:
        GLOBAL_MATRIX_GLYPH_LUM = build_matrix_glyph_field()
    return GLOBAL_MATRIX_GLYPH_LUM


def apply_matrix_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Modulate gradient bar pixels with a static ASCII glyph luminance field.
    Keeps hue on glyph strokes; gaps (low lum) go fully black.
    """
    lum = _ensure_matrix_glyph_field()
    return _modulate_frame_with_lum(frame, mask, lum)


class MatrixRainSimulator:
    """Per-column code rain: scroll down one row per tick, new head every tick (no gaps)."""

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        n_chars: int,
        rng: np.random.Generator,
    ) -> None:
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_chars = n_chars
        self.char_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        self.bright_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        self.tick = 0
        for gx in range(grid_w):
            for row in range(grid_h):
                self.char_grid[row, gx] = int(rng.integers(0, n_chars))
                self.bright_grid[row, gx] = 255

    def _advance_column(self, gx: int, rng: np.random.Generator) -> None:
        col_c = self.char_grid[:, gx].copy()
        col_b = self.bright_grid[:, gx].copy()
        self.char_grid[1:, gx] = col_c[:-1]
        self.bright_grid[1:, gx] = col_b[:-1]
        self.char_grid[0, gx] = int(rng.integers(0, self.n_chars))
        self.bright_grid[0, gx] = 255
        for row in range(1, self.grid_h):
            if self.bright_grid[row, gx] == 0:
                continue
            if rng.random() < MATRIX_RAIN_CHAR_MUTATION:
                self.char_grid[row, gx] = int(rng.integers(0, self.n_chars))

    def step(self, rng: np.random.Generator) -> None:
        self.tick += 1
        for gx in range(self.grid_w):
            self._advance_column(gx, rng)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self.char_grid.copy(), self.bright_grid.copy()


_MATRIX_RAIN_ATLAS: np.ndarray | None = None
_MATRIX_RAIN_SNAPSHOTS: list[tuple[np.ndarray, np.ndarray]] | None = None
_MATRIX_RAIN_LUM_CACHE: dict[int, np.ndarray] = {}
_MATRIX_RAIN_SEED: int | None = None


def precompute_matrix_rain_snapshots() -> None:
    """Simulate rain ticks once at startup; store char/bright grids per tick."""
    global _MATRIX_RAIN_ATLAS, _MATRIX_RAIN_SNAPSHOTS, _MATRIX_RAIN_LUM_CACHE
    global _MATRIX_RAIN_SEED, _MATRIX_RAIN_SCREEN_RAMP

    _MATRIX_RAIN_SCREEN_RAMP = None

    atlas, charset, scale_min, scale_max = build_matrix_glyph_atlas(MATRIX_CHARSET)
    grid_w, grid_h = _matrix_grid_size()
    seed = MATRIX_SEED if MATRIX_SEED is not None else int(np.random.default_rng().integers(0, 2**32))
    rng = np.random.default_rng(seed)
    n_ticks = max(1, int(ceil(duration * MATRIX_RAIN_FPS)))

    sim = MatrixRainSimulator(grid_h, grid_w, len(charset), rng)
    snapshots: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_ticks):
        snapshots.append(sim.snapshot())
        sim.step(rng)

    _MATRIX_RAIN_ATLAS = atlas
    _MATRIX_RAIN_SNAPSHOTS = snapshots
    _MATRIX_RAIN_LUM_CACHE = {}
    _MATRIX_RAIN_SEED = seed

    log_matrix_mask(
        enabled=True,
        mode="rain",
        glyphs=len(charset),
        scale_min=scale_min,
        scale_max=scale_max,
        layout_seed=seed,
        rain_ticks=n_ticks,
    )


def _ensure_matrix_rain() -> None:
    if _MATRIX_RAIN_SNAPSHOTS is None:
        precompute_matrix_rain_snapshots()


def _rain_tick_index(t: float) -> int:
    if _MATRIX_RAIN_SNAPSHOTS is None:
        return 0
    idx = int(t * MATRIX_RAIN_FPS)
    return min(idx, len(_MATRIX_RAIN_SNAPSHOTS) - 1)


def _rain_lum_at_time(t: float) -> np.ndarray:
    _ensure_matrix_rain()
    assert _MATRIX_RAIN_ATLAS is not None
    assert _MATRIX_RAIN_SNAPSHOTS is not None

    tick = _rain_tick_index(t)
    lum = _MATRIX_RAIN_LUM_CACHE.get(tick)
    if lum is None:
        char_grid, bright_grid = _MATRIX_RAIN_SNAPSHOTS[tick]
        lum = compose_lum_from_grid(_MATRIX_RAIN_ATLAS, char_grid, bright_grid)
        lum = apply_matrix_rain_screen_ramp(lum)
        _MATRIX_RAIN_LUM_CACHE[tick] = lum
    return lum


def apply_matrix_rain_mask(frame: np.ndarray, mask: np.ndarray, t: float) -> np.ndarray:
    """Modulate bar pixels with a time-varying code-rain luminance field."""
    lum = _rain_lum_at_time(t)
    return _modulate_frame_with_lum(frame, mask, lum, use_lum_cutoff=False)


def create_gradient(top_fraction: float = 0.2):
    """top_fraction: fraction of height (from top) that is pure GRADIENT_COLOR_TOP before blending."""
    bottom = np.array(GRADIENT_COLOR_BOTTOM, dtype=np.float64)
    top = np.array(GRADIENT_COLOR_TOP, dtype=np.float64)
    top_band = int(H * top_fraction)
    grad = np.zeros((H, W, 3), dtype='uint8')
    for y in range(H):
        if y < top_band:
            grad[y, :] = top.astype(np.uint8)
        else:
            # blend from top to bottom over the remaining height
            mix = 1.0 - (y - top_band) / (H - top_band)
            grad[y, :] = np.clip(bottom + (top - bottom) * mix, 0, 255).astype(np.uint8)
    return grad

def create_vignette():
    # Create a 2D Gaussian mask
    kernel_x = cv2.getGaussianKernel(W, W/2)
    kernel_y = cv2.getGaussianKernel(H, H/2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    return mask[:, :, np.newaxis] # Shape (H, W, 1) for broadcasting

GLOBAL_GRADIENT: np.ndarray | None = None
VIGNETTE_MASK: np.ndarray | None = None

def _save_matrix_atlas_sheet(atlas: np.ndarray, charset: str, out_path: str) -> None:
    """Contact sheet of every glyph tile for visual charset verification."""
    n = len(charset)
    cols = min(16, max(1, n))
    rows = (n + cols - 1) // cols
    pad = 4
    label_h = 12
    sheet = np.zeros(
        (rows * (MATRIX_CELL_H + pad + label_h), cols * (MATRIX_CELL_W + pad)),
        dtype=np.uint8,
    )
    for i, ch in enumerate(charset):
        row, col = divmod(i, cols)
        y0 = row * (MATRIX_CELL_H + pad + label_h)
        x0 = col * (MATRIX_CELL_W + pad)
        sheet[y0 : y0 + MATRIX_CELL_H, x0 : x0 + MATRIX_CELL_W] = atlas[i]
        cv2.putText(
            sheet,
            ch if ch.isprintable() else "?",
            (x0, y0 + MATRIX_CELL_H + label_h - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            180,
            1,
            cv2.LINE_8,
        )
    cv2.imwrite(out_path, sheet)


def _apply_crt_postfx(frame: np.ndarray) -> np.ndarray:
    """Scanlines + chromatic aberration (same as make_frame post-matrix)."""
    out = frame.copy()
    out[::3, :, :] = (out[::3, :, :] * 0.5).astype(np.uint8)
    b, g, r = cv2.split(out)
    r = np.roll(r, 2, axis=1)
    b = np.roll(b, -2, axis=1)
    return cv2.merge([b, g, r])


def debug_matrix_glyphs() -> None:
    """
    Export matrix debug images (no video encode).
    Run: python3 waveform.py --debug-matrix
    """
    global GLOBAL_MATRIX_GLYPH_LUM
    GLOBAL_MATRIX_GLYPH_LUM = None
    flag = DEBUG_FLAG_MATRIX
    written: list[str] = []

    atlas, charset, _scale_min, _scale_max = build_matrix_glyph_atlas(MATRIX_CHARSET)
    atlas_path = debug_out(flag, "atlas.png")
    _save_matrix_atlas_sheet(atlas, charset, atlas_path)
    written.append(atlas_path)

    if USE_MATRIX_MASK:
        lum = build_matrix_glyph_field()
        GLOBAL_MATRIX_GLYPH_LUM = lum
        for name, img in (
            ("lum.png", lum),
            ("lum_color.png", cv2.cvtColor(lum, cv2.COLOR_GRAY2BGR)),
        ):
            path = debug_out(flag, name)
            cv2.imwrite(path, img)
            written.append(path)

    if USE_MATRIX_RAIN:
        precompute_matrix_rain_snapshots()
        rain_times = [0.0, 0.25, 0.5, 1.0]
        for t_sample in rain_times:
            if t_sample > duration:
                continue
            rain_lum = _rain_lum_at_time(t_sample)
            tag = f"{int(t_sample * 1000):04d}"
            for name, img in (
                (f"rain_lum_t{tag}.png", rain_lum),
                (f"rain_lum_t{tag}_color.png", cv2.cvtColor(rain_lum, cv2.COLOR_GRAY2BGR)),
            ):
                path = debug_out(flag, name)
                cv2.imwrite(path, img)
                written.append(path)

    peak_flat = int(np.argmax(S_norm))
    peak_col = np.unravel_index(peak_flat, S_norm.shape)[1]
    t_peak = (peak_col / (S_norm.shape[1] - 1)) * duration
    amps = amplitudes_at_time(t_peak)
    mask = np.zeros((H, W, 3), dtype=np.uint8)
    draw_bars(mask, amps)
    composite = cv2.bitwise_and(GLOBAL_GRADIENT, mask)

    if USE_MATRIX_MASK:
        matrix_only = apply_matrix_mask(composite.copy(), mask)
        path = debug_out(flag, "on_bars.png")
        cv2.imwrite(path, matrix_only)
        written.append(path)
        lum = _ensure_matrix_glyph_field()
        raw_glyph_view = composite.astype(np.float32)
        bar = mask[:, :, 0] > 0
        raw_glyph_view[bar] = (
            raw_glyph_view[bar] * (lum[bar, np.newaxis].astype(np.float32) / 255.0)
        )
        path = debug_out(flag, "raw_multiply.png")
        cv2.imwrite(path, np.clip(raw_glyph_view, 0, 255).astype(np.uint8))
        written.append(path)
        path = debug_out(flag, "with_crt_chromatic.png")
        cv2.imwrite(path, _apply_crt_postfx(matrix_only))
        written.append(path)

    if USE_MATRIX_RAIN:
        rain_on_bars = apply_matrix_rain_mask(composite.copy(), mask, t_peak)
        path = debug_out(flag, "rain_on_bars.png")
        cv2.imwrite(path, rain_on_bars)
        written.append(path)
        path = debug_out(flag, "rain_with_crt_chromatic.png")
        cv2.imwrite(path, _apply_crt_postfx(rain_on_bars))
        written.append(path)

    clipped = sum(1 for i, ch in enumerate(charset) if atlas[i].max() < 32)
    log_section("Matrix debug exports")
    log_item("Directory", os.path.join(DEBUG_DIR, flag))
    log_item("Renderable", f"{len(charset)}/{len(MATRIX_CHARSET)} chars")
    log_item("Weak tiles", clipped)
    log_item("Files", ", ".join(os.path.basename(p) for p in written))


def debug_bar_geometry() -> None:
    """Write bar debug PNGs at global peak; run with DEBUG_BAR_HEIGHT=1 or --debug-bars."""
    flag = DEBUG_FLAG_BARS
    peak_flat = int(np.argmax(S_norm))
    peak_band, peak_col = np.unravel_index(peak_flat, S_norm.shape)
    t_peak = (peak_col / (S_norm.shape[1] - 1)) * duration
    amps = amplitudes_at_time(t_peak)
    max_h = amplitude_to_bar_height(float(amps.max()))
    _, expected_top = bar_y_bounds(max_h)

    mask = np.zeros((H, W, 3), dtype=np.uint8)
    draw_bars(mask, amps)
    extent = measure_mask_bar_extent(mask)

    overlay = GLOBAL_GRADIENT.copy()
    cv2.line(overlay, (0, expected_top), (W - 1, expected_top), (0, 0, 255), 2)
    cv2.line(overlay, (0, BAR_BASE_Y), (W - 1, BAR_BASE_Y), (0, 255, 0), 2)
    cv2.line(overlay, (0, BAR_TOP_MARGIN), (W - 1, BAR_TOP_MARGIN), (255, 0, 0), 2)
    composite = cv2.bitwise_and(GLOBAL_GRADIENT, mask)
    # matrix_frame = apply_matrix_mask(composite, mask)
    # cv2.imwrite(debug_out(flag, "matrix.png"), matrix_frame)

    log_section("Bar height debug (peak frame)")
    log_item("Directory", os.path.join(DEBUG_DIR, flag))
    log_item("Time", f"{t_peak:.3f}s")
    log_item("Peak band / amp", f"{peak_band} / {amps.max():.6f}")
    log_item("Expected top Y", expected_top)
    log_item("Measured span", f"{extent['span']}px (top={extent['top']}, bottom={extent['bottom']})")
    written = []
    for name, img in (
        ("mask.png", mask),
        ("overlay.png", overlay),
        ("composite.png", composite),
    ):
        path = debug_out(flag, name)
        cv2.imwrite(path, img)
        written.append(path)
    log_item("Files", ", ".join(os.path.basename(p) for p in written))


def make_frame(t: float) -> np.ndarray:
    """
    This function is called by MoviePy for every frame of the video.
    't' is the current time in seconds.
    """

    # create canvas with transparent background and mask canvas for bar drawing
    frame = np.zeros((H, W, 3), dtype='uint8')
    mask = np.zeros((H, W, 3), dtype='uint8')

    current_amplitudes = amplitudes_at_time(t)

    # ROUNDED BAR OPTION
    # draw_rounded_bars(frame, current_amplitudes)

    # RECTANGULAR BAR OPTION
    draw_bars(mask, current_amplitudes)

    # Gradient color masking
    assert GLOBAL_GRADIENT is not None
    frame = cv2.bitwise_and(GLOBAL_GRADIENT, mask)

    # Matrix ASCII mask — driven by USE_MATRIX_* config flags
    if USE_MATRIX_MASK:
        frame = apply_matrix_mask(frame, mask)
    elif USE_MATRIX_RAIN:
        frame = apply_matrix_rain_mask(frame, mask, t)

    # CRT Scanlines
    frame[::3, :,:] = (frame[::3, :, :] * 0.5).astype('uint8')

    # Chromatic aberation
    b, g, r = cv2.split(frame)
    r = np.roll(r, 2, axis=1)  # Shift Red channel 2px right
    b = np.roll(b, -2, axis=1) # Shift Blue channel 2px left
    frame = cv2.merge([b, g, r])

    # Bloom
    frame = apply_bloom(frame)

    return frame


### VIDEO RENDER AND EXPORT ###################################################


def _reset_runtime_caches() -> None:
    """Clear per-run caches so a second generate() does not reuse stale state."""
    global GLOBAL_MATRIX_GLYPH_LUM, GLOBAL_GRADIENT, VIGNETTE_MASK
    global _MATRIX_RAIN_ATLAS, _MATRIX_RAIN_SNAPSHOTS, _MATRIX_RAIN_LUM_CACHE
    global _MATRIX_RAIN_SEED, _MATRIX_RAIN_SCREEN_RAMP

    GLOBAL_MATRIX_GLYPH_LUM = None
    GLOBAL_GRADIENT = None
    VIGNETTE_MASK = None
    _MATRIX_RAIN_ATLAS = None
    _MATRIX_RAIN_SNAPSHOTS = None
    _MATRIX_RAIN_LUM_CACHE = {}
    _MATRIX_RAIN_SEED = None
    _MATRIX_RAIN_SCREEN_RAMP = None


class _CallbackProgressLogger(ProgressBarLogger if ProgressBarLogger is not None else object):
    """Proglog adapter that forwards MoviePy bar progress to a UI callback."""

    def __init__(self, callback: Callable[[float, str], None]):
        if ProgressBarLogger is not None:
            super().__init__()
        self._callback = callback
        self._last_message = "Rendering"
        self._phase = 0  # 0=audio, 1=video

    def callback(self, **changes):  # type: ignore[override]
        message = changes.get("message")
        if message:
            self._last_message = str(message)
            lower = self._last_message.lower()
            if "writing video" in lower:
                self._phase = 1
            elif "writing audio" in lower or "building video" in lower:
                self._phase = 0
            self._callback(-1.0, self._last_message)

    def bars_callback(self, bar, attr, value, old_value=None):  # type: ignore[override]
        if ProgressBarLogger is None:
            return
        if attr != "index":
            return
        total = self.bars.get(bar, {}).get("total") or 0
        if total <= 0:
            return
        local = max(0.0, min(1.0, float(value) / float(total)))
        # Keep overall progress monotonic: audio fills 10–30%, video 30–99%.
        if self._phase == 0 or bar == "chunk":
            fraction = 0.10 + 0.20 * local
            self._phase = 0
        else:
            fraction = 0.30 + 0.69 * local
            self._phase = 1
        label = self._last_message or str(bar)
        self._callback(fraction, label)


def run(
    *,
    audio_path: str | None = None,
    output_path: str | None = None,
    config: dict[str, Any] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    debug_bars: bool = False,
    debug_matrix: bool = False,
) -> str:
    """
    Generate a waveform visualizer video.

    progress_callback(fraction, message):
      fraction in [0, 1], or -1 when only a status message is available.
    Returns the output path written.
    """
    global GLOBAL_GRADIENT, VIGNETTE_MASK, AUDIO_PATH, OUTPUT_PATH

    cfg = dict(config or {})
    if audio_path is not None:
        cfg["AUDIO_PATH"] = audio_path
    if output_path is not None:
        cfg["OUTPUT_PATH"] = output_path

    apply_config(cfg)
    _reset_runtime_caches()

    audio_path = audio_path or AUDIO_PATH
    output_path = output_path or OUTPUT_PATH
    AUDIO_PATH = audio_path
    OUTPUT_PATH = output_path

    def report(fraction: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(fraction, message)

    report(0.0, "Loading audio & building signal chain")
    peak = build_signal_chain(audio_path)
    log_run_config(peak, audio_path)

    GLOBAL_GRADIENT = create_gradient()
    VIGNETTE_MASK = create_vignette()

    if debug_bars:
        debug_bar_geometry()
        report(1.0, "Debug bars complete")
        return debug_out(DEBUG_FLAG_BARS, "composite.png")

    if debug_matrix:
        debug_matrix_glyphs()
        report(1.0, "Debug matrix complete")
        return debug_out(DEBUG_FLAG_MATRIX, "atlas.png")

    report(0.05, "Preparing matrix / masks")
    if USE_MATRIX_MASK:
        _ensure_matrix_glyph_field()
    elif USE_MATRIX_RAIN:
        precompute_matrix_rain_snapshots()
    else:
        log_matrix_mask(enabled=False)

    log_section("Export")
    log_item("Output", output_path)
    log_item("Codec", "prores_ks (4444, yuva444p10le)")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    assert audio_clip is not None
    clip = VideoClip(make_frame, duration=duration)
    clip = clip.with_audio(audio_clip)

    logger: Any = "bar"
    if progress_callback is not None and ProgressBarLogger is not None:
        logger = _CallbackProgressLogger(progress_callback)

    report(0.1, "Writing video")
    print()
    clip.write_videofile(
        output_path,
        fps=FPS,
        codec="prores_ks",
        logger=logger,
        ffmpeg_params=[
            "-profile:v", "4",          # '4' is the ProRes 4444 profile
            "-pix_fmt", "yuva444p10le"  # 10-bit YUV + Alpha
        ],
    )
    log_item("Status", "complete")
    print()
    report(1.0, "Complete")
    return output_path


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    debug_bars = DEBUG_BAR_HEIGHT or "--debug-bars" in argv
    debug_matrix = "--debug-matrix" in argv
    run(debug_bars=debug_bars, debug_matrix=debug_matrix)


if __name__ == "__main__":
    main()
