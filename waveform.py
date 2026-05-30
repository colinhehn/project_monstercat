from typing import Any
import os
import sys
import cv2
import numpy as np
import librosa
from moviepy import VideoClip, AudioFileClip
from scipy.ndimage import gaussian_filter1d

DEBUG_BAR_HEIGHT = os.environ.get("DEBUG_BAR_HEIGHT", "").lower() in ("1", "true", "yes")
DEBUG_BARS_ONLY = DEBUG_BAR_HEIGHT or "--debug-bars" in sys.argv
DEBUG_MATRIX_ONLY = "--debug-matrix" in sys.argv


def log_section(title: str) -> None:
    print(f"\n[{title}]")


def log_item(key: str, value: object) -> None:
    print(f"  {key:<26} {value}")


def log_matrix_mask(
    *,
    enabled: bool,
    glyphs: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
    layout_seed: int | None = None,
) -> None:
    """Single config block for matrix settings and (optional) build results."""
    log_section("Matrix mask")
    log_item("Enabled", enabled)
    if not enabled:
        return
    log_item("Cell size", f"{MATRIX_CELL_W}x{MATRIX_CELL_H}px")
    log_item("Grid density", f"~{W // MATRIX_CELL_W}x{H // MATRIX_CELL_H} cells")
    log_item("Chars per bar row", MATRIX_CHARS_PER_BAR_ROW)
    log_item("Cell height ratio", MATRIX_CELL_HEIGHT_RATIO)
    log_item("Font thickness", MATRIX_FONT_THICKNESS)
    log_item("Charset size", len(MATRIX_CHARSET))
    log_item("Lum cutoff", MATRIX_LUM_CUTOFF)
    log_item("Seed", MATRIX_SEED if MATRIX_SEED is not None else "random per run")
    if glyphs is not None:
        log_item("Glyphs loaded", glyphs)
    if scale_min is not None and scale_max is not None:
        log_item("Font scale range", f"{scale_min:.2f} - {scale_max:.2f}")
    if layout_seed is not None:
        log_item("Layout seed", layout_seed)


#### CONFIGURATION ############################################################

# TODO: make these configurable on script runtime with Textual CLI (GitHub).
W, H = 1080, 1920
N_BANDS = 18
GUTTER = 4 # px gap between bars
BAR_WIDTH = (W - (N_BANDS * GUTTER)) / N_BANDS # total available width / number of bars
# Bar span in px when amp=1.0 (after peak normalization). NOT including BAR_BOTTOM_MARGIN.
MAX_BAR_HEIGHT = 1650
# Lift baseline off frame bottom; adds to "reach from bottom" (reach = margin + MAX_BAR_HEIGHT).
BAR_BOTTOM_MARGIN = 0
BAR_TOP_MARGIN = 70  # minimum y index for bar top at full amplitude

BAR_BASE_Y = H - 1 - BAR_BOTTOM_MARGIN
MAX_DRAWABLE_HEIGHT = BAR_BASE_Y - BAR_TOP_MARGIN
PEAK_BAR_TOP_Y = BAR_BASE_Y - MAX_BAR_HEIGHT
PEAK_REACH_FROM_FRAME_BOTTOM = (H - 1) - PEAK_BAR_TOP_Y
if MAX_BAR_HEIGHT > MAX_DRAWABLE_HEIGHT:
    raise ValueError(
        f"MAX_BAR_HEIGHT ({MAX_BAR_HEIGHT}) exceeds drawable range "
        f"({MAX_DRAWABLE_HEIGHT}): BAR_BASE_Y={BAR_BASE_Y}, BAR_TOP_MARGIN={BAR_TOP_MARGIN}"
    )

FPS = 60
AUDIO_PATH = "Pattern 5.wav"
STATIC_COLOR = [255, 255, 255]
GRADIENT_COLOR_BOTTOM = [255, 231, 97]
GRADIENT_COLOR_TOP = [255, 96, 234]

# Matrix ASCII mask (static grid, CMatrix-inspired charset)
# Fewer on-screen glyphs → larger cells: use MATRIX_CHARS_PER_BAR_ROW=1 and/or
# raise MATRIX_CELL_HEIGHT_RATIO (taller rows). Grid math picks up the rest.
MATRIX_CHARS_PER_BAR_ROW = 1
MATRIX_CELL_HEIGHT_RATIO = 1
_BAR_SLOT_W = int(BAR_WIDTH + GUTTER)
MATRIX_CELL_W = max(1, _BAR_SLOT_W // MATRIX_CHARS_PER_BAR_ROW)
MATRIX_CELL_H = max(8, int(MATRIX_CELL_W * MATRIX_CELL_HEIGHT_RATIO))
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
# Keep in sync with make_frame: True when apply_matrix_mask(...) is uncommented.
USE_MATRIX_MASK = True

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

### The Essentials #####

# Load audio file into MoviePy
y, sr = librosa.load(AUDIO_PATH)
audio_clip = AudioFileClip(AUDIO_PATH)
duration = audio_clip.duration

# Mel Spectogram transform with range 20-7000Hz
fmin, fmax = 20, 7000
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_BANDS, fmin=fmin, fmax=fmax)

# Convert to decibels
S_db = librosa.power_to_db(S, ref=np.max)

# Normalize 0 - 1 for visual representation
S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())

### Fine Tweaks #####

# spectral delay effect
S_norm = spectral_delay(S_norm)

# Exponential curve to accentuate low frequencies
exponent = 2.6
S_norm = np.power(S_norm, exponent)

# TILT EQ to boost treble
tilt_min, tilt_max = 0.8, 2.2
tilt = np.linspace(tilt_min, tilt_max, N_BANDS)
S_norm = (S_norm.T * tilt).T

### WAVEFORM PRE-PROCESSING EXECUTION #########################################

# gravity / liquid effect
gravity_attack, gravity_max_decay, gravity_min_decay = 0.5, 0.15, 0.05
S_norm = gravity(S_norm, gravity_attack, gravity_max_decay, gravity_min_decay)

# wash delay effect
S_norm = wash_delay(S_norm)

# rolling average (less detail in waveform, more smooth)
# S_norm = rolling_average(S_norm, 5)

# rubber-band effect, spread of frequency data across x axis
gaussian_filter = 1.0
S_norm = gaussian_filter1d(S_norm, sigma=gaussian_filter, axis=0) # sigma dictates blur strength (1.0-2.0 usually solid)

# Scale so the loudest moment maps to amp=1 -> MAX_BAR_HEIGHT px from BAR_BASE_Y.
peak = S_norm.max()
if peak > 0:
    S_norm = S_norm / peak


def log_run_config(pre_peak: float) -> None:
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
    log_item("Source", AUDIO_PATH)
    log_item("Duration", f"{duration:.2f}s")
    log_item("Mel bands", N_BANDS)
    log_item("Mel range", f"{fmin}-{fmax} Hz")

    log_section("Signal processing")
    log_item("Spectral delay", "on")
    log_item("Exponent", exponent)
    log_item("Tilt EQ", f"{tilt_min} - {tilt_max}")
    log_item(
        "Gravity",
        f"attack={gravity_attack}, decay={gravity_min_decay}-{gravity_max_decay}",
    )
    log_item("Wash delay", "on")
    log_item("Gaussian blur", f"sigma={gaussian_filter}")
    log_item("Peak normalize", f"pre-scale max={pre_peak:.4f}")


log_run_config(peak)


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


def build_matrix_glyph_field() -> np.ndarray:
    """Full-frame static luminance field: random char per grid cell."""
    atlas, charset, scale_min, scale_max = build_matrix_glyph_atlas(MATRIX_CHARSET)
    grid_w = (W + MATRIX_CELL_W - 1) // MATRIX_CELL_W
    grid_h = (H + MATRIX_CELL_H - 1) // MATRIX_CELL_H
    seed = MATRIX_SEED if MATRIX_SEED is not None else np.random.randint(0, 2**32)
    rng = np.random.default_rng(seed)
    char_grid = rng.integers(0, len(charset), (grid_h, grid_w))

    lum = np.zeros((H, W), dtype=np.uint8)
    for gy in range(grid_h):
        y0 = gy * MATRIX_CELL_H
        y1 = min(y0 + MATRIX_CELL_H, H)
        th = y1 - y0
        for gx in range(grid_w):
            x0 = gx * MATRIX_CELL_W
            x1 = min(x0 + MATRIX_CELL_W, W)
            tw = x1 - x0
            lum[y0:y1, x0:x1] = atlas[char_grid[gy, gx], :th, :tw]
    log_matrix_mask(
        enabled=True,
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
    bar = mask[:, :, 0] > 0
    glyph = lum.astype(np.float32) / 255.0
    glyph = np.where(lum >= MATRIX_LUM_CUTOFF, glyph, 0.0)
    factor = glyph[:, :, np.newaxis]
    out = frame.astype(np.float32)
    out[bar] = out[bar] * factor[bar]
    return np.clip(out, 0, 255).astype(np.uint8)


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

GLOBAL_GRADIENT = create_gradient()
VIGNETTE_MASK = create_vignette()

def _save_matrix_atlas_sheet(atlas: np.ndarray, charset: str) -> None:
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
    cv2.imwrite("debug_matrix_atlas.png", sheet)


def debug_matrix_glyphs() -> None:
    """
    Export matrix debug images (no video encode).
    Run: python3 waveform.py --debug-matrix
    """
    global GLOBAL_MATRIX_GLYPH_LUM
    GLOBAL_MATRIX_GLYPH_LUM = None

    atlas, charset, _scale_min, _scale_max = build_matrix_glyph_atlas(MATRIX_CHARSET)
    _save_matrix_atlas_sheet(atlas, charset)
    lum = build_matrix_glyph_field()
    GLOBAL_MATRIX_GLYPH_LUM = lum

    cv2.imwrite("debug_matrix_lum.png", lum)
    cv2.imwrite("debug_matrix_lum_color.png", cv2.cvtColor(lum, cv2.COLOR_GRAY2BGR))

    peak_flat = int(np.argmax(S_norm))
    peak_col = np.unravel_index(peak_flat, S_norm.shape)[1]
    t_peak = (peak_col / (S_norm.shape[1] - 1)) * duration
    amps = amplitudes_at_time(t_peak)
    mask = np.zeros((H, W, 3), dtype=np.uint8)
    draw_bars(mask, amps)
    composite = cv2.bitwise_and(GLOBAL_GRADIENT, mask)

    matrix_only = apply_matrix_mask(composite.copy(), mask)
    cv2.imwrite("debug_matrix_on_bars.png", matrix_only)

    raw_glyph_view = composite.astype(np.float32)
    bar = mask[:, :, 0] > 0
    raw_glyph_view[bar] = (
        raw_glyph_view[bar] * (lum[bar, np.newaxis].astype(np.float32) / 255.0)
    )
    cv2.imwrite("debug_matrix_raw_multiply.png", np.clip(raw_glyph_view, 0, 255).astype(np.uint8))

    with_post = matrix_only.copy()
    with_post[::3, :, :] = (with_post[::3, :, :] * 0.5).astype(np.uint8)
    b, g, r = cv2.split(with_post)
    r = np.roll(r, 2, axis=1)
    b = np.roll(b, -2, axis=1)
    with_post = cv2.merge([b, g, r])
    cv2.imwrite("debug_matrix_with_crt_chromatic.png", with_post)

    clipped = sum(1 for i, ch in enumerate(charset) if atlas[i].max() < 32)
    log_section("Matrix debug exports")
    log_item("Renderable", f"{len(charset)}/{len(MATRIX_CHARSET)} chars")
    log_item("Weak tiles", clipped)
    log_item("Files", "debug_matrix_atlas.png, debug_matrix_lum.png, ...")


def debug_bar_geometry() -> None:
    """Write debug_bar_*.png at global peak; run with DEBUG_BAR_HEIGHT=1 or --debug-bars."""
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
    # cv2.imwrite("debug_bar_matrix.png", matrix_frame)

    log_section("Bar height debug (peak frame)")
    log_item("Time", f"{t_peak:.3f}s")
    log_item("Peak band / amp", f"{peak_band} / {amps.max():.6f}")
    log_item("Expected top Y", expected_top)
    log_item("Measured span", f"{extent['span']}px (top={extent['top']}, bottom={extent['bottom']})")
    cv2.imwrite("debug_bar_mask.png", mask)
    cv2.imwrite("debug_bar_overlay.png", overlay)
    cv2.imwrite("debug_bar_composite.png", composite)
    log_item("Files", "debug_bar_mask.png, debug_bar_overlay.png, debug_bar_composite.png")


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
    frame = cv2.bitwise_and(GLOBAL_GRADIENT, mask)

    # Matrix ASCII mask (static CMatrix-style glyphs on bars)
    frame = apply_matrix_mask(frame, mask)

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

if DEBUG_BARS_ONLY:
    debug_bar_geometry()
    sys.exit(0)

if DEBUG_MATRIX_ONLY:
    debug_matrix_glyphs()
    sys.exit(0)

OUTPUT_PATH = "output_waveform.mov"

if USE_MATRIX_MASK:
    _ensure_matrix_glyph_field()
else:
    log_matrix_mask(enabled=False)

log_section("Export")
log_item("Output", OUTPUT_PATH)
log_item("Codec", "prores_ks (4444, yuva444p10le)")

clip = VideoClip(make_frame, duration=duration)
clip = clip.with_audio(audio_clip)
print()
clip.write_videofile(
    OUTPUT_PATH,
    fps=FPS,
    codec="prores_ks",
    logger="bar",
    ffmpeg_params=[
        "-profile:v", "4",          # '4' is the ProRes 4444 profile
        "-pix_fmt", "yuva444p10le"  # 10-bit YUV + Alpha
    ],
)
log_item("Status", "complete")
print()