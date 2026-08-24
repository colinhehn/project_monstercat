"""
Center-line wobble visualizer.

Mel-band amplitudes displace a single white horizontal line through screen
center. Quiet audio stays flat; frequency peaks push the polyline vertically.
No CRT, matrix, bloom, or gradient — black background, white stroke.
"""

import cv2
import numpy as np
import librosa
from moviepy import VideoClip, AudioFileClip
from scipy.ndimage import gaussian_filter1d


def log_section(title: str) -> None:
    print(f"\n[{title}]")


def log_item(key: str, value: object) -> None:
    print(f"  {key:<26} {value}")


#### CONFIGURATION ############################################################

W, H = 1080, 1920
FPS = 60
N_BANDS = 64
LINE_THICKNESS = 3
LINE_COLOR = (255, 255, 255)  # BGR white
# Vertical travel from center at amp=1.0 (clipped to frame).
MAX_DISPLACEMENT = 420

AUDIO_PATH = "unused_promo_wav/when_david_heard_monstercat_promo.wav"
OUTPUT_PATH = "output_wobble.mov"

CENTER_Y = H // 2
if MAX_DISPLACEMENT < 0:
    raise ValueError(f"MAX_DISPLACEMENT must be >= 0, got {MAX_DISPLACEMENT}")
if N_BANDS < 2:
    raise ValueError(f"N_BANDS must be >= 2 for a polyline, got {N_BANDS}")


### PRE-PROCESSING ############################################################

def gravity(
    S_norm: np.ndarray,
    attack: float = 0.5,
    max_decay: float = 0.15,
    min_decay: float = 0.05,
) -> np.ndarray:
    """Temporal attack/decay smoothing per band so the line does not flicker."""
    smoothed_S = np.zeros_like(S_norm)
    decay_gradient = np.linspace(min_decay, max_decay, num=N_BANDS)

    for t in range(1, S_norm.shape[1]):
        target = S_norm[:, t]
        prev = smoothed_S[:, t - 1]
        smoothed_S[:, t] = np.where(
            target > prev,
            prev + (target - prev) * attack,
            prev - (prev - target) * decay_gradient,
        )
    return smoothed_S


### AUDIO SIGNAL CHAIN ########################################################

y, sr = librosa.load(AUDIO_PATH)
audio_clip = AudioFileClip(AUDIO_PATH)
duration = audio_clip.duration

fmin, fmax = 20, 7000
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_BANDS, fmin=fmin, fmax=fmax)
S_db = librosa.power_to_db(S, ref=np.max)
S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())

gravity_attack, gravity_max_decay, gravity_min_decay = 0.5, 0.15, 0.05
S_norm = gravity(S_norm, gravity_attack, gravity_max_decay, gravity_min_decay)

gaussian_sigma = 1.0
S_norm = gaussian_filter1d(S_norm, sigma=gaussian_sigma, axis=0)

peak = S_norm.max()
if peak > 0:
    S_norm = S_norm / peak


def log_run_config(pre_peak: float) -> None:
    log_section("Environment")
    log_item("OpenCV", cv2.__version__)

    log_section("Canvas & line")
    log_item("Resolution", f"{W} x {H}")
    log_item("FPS", FPS)
    log_item("Bands", N_BANDS)
    log_item("Center Y", CENTER_Y)
    log_item("Max displacement", f"{MAX_DISPLACEMENT}px")
    log_item("Line thickness", LINE_THICKNESS)
    log_item("Line color (BGR)", LINE_COLOR)

    log_section("Audio")
    log_item("Source", AUDIO_PATH)
    log_item("Duration", f"{duration:.2f}s")
    log_item("Mel range", f"{fmin}-{fmax} Hz")

    log_section("Signal processing")
    log_item(
        "Gravity",
        f"attack={gravity_attack}, decay={gravity_min_decay}-{gravity_max_decay}",
    )
    log_item("Gaussian blur", f"sigma={gaussian_sigma}")
    log_item("Peak normalize", f"pre-scale max={pre_peak:.4f}")


log_run_config(peak)


def amplitudes_at_time(t: float) -> np.ndarray:
    float_idx = (t / duration) * (S_norm.shape[1] - 1)
    idx_floor = int(np.floor(float_idx))
    idx_ceil = int(np.ceil(float_idx))
    weight = float_idx - idx_floor
    return (1 - weight) * S_norm[:, idx_floor] + weight * S_norm[:, idx_ceil]


### RENDERING #################################################################

def line_points(amplitudes: np.ndarray) -> np.ndarray:
    """Map band amplitudes to polyline vertices across the frame width."""
    points = np.empty((N_BANDS, 2), dtype=np.int32)
    denom = N_BANDS - 1
    for i, amp in enumerate(amplitudes):
        x = int(round((i / denom) * (W - 1)))
        y = int(round(CENTER_Y - float(np.clip(amp, 0.0, 1.0)) * MAX_DISPLACEMENT))
        y = int(np.clip(y, 0, H - 1))
        points[i, 0] = x
        points[i, 1] = y
    return points


def draw_wobble_line(frame: np.ndarray, amplitudes: np.ndarray) -> None:
    pts = line_points(amplitudes)
    cv2.polylines(
        frame,
        [pts],
        isClosed=False,
        color=LINE_COLOR,
        thickness=LINE_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def make_frame(t: float) -> np.ndarray:
    """MoviePy frame callback: black canvas + white center-line wobble."""
    frame = np.zeros((H, W, 3), dtype="uint8")
    draw_wobble_line(frame, amplitudes_at_time(t))
    return frame


### VIDEO RENDER AND EXPORT ###################################################

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
        "-profile:v", "4",
        "-pix_fmt", "yuva444p10le",
    ],
)
log_item("Status", "complete")
print()
