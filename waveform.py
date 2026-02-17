from typing import Any
import cv2
import numpy as np
import librosa
from moviepy import VideoClip, AudioFileClip
from scipy.ndimage import gaussian_filter1d

#### CONFIGURATION ############################################################

# TODO: make these configurable on script runtime with Textual CLI (GitHub).
W, H = 1400, 1400
N_BANDS = 24
GUTTER = 4 # px gap between bars
BAR_WIDTH = ((W-400) - (N_BANDS * GUTTER)) / N_BANDS # total available width / number of bars
MAX_BAR_HEIGHT = 1200

FPS = 60
AUDIO_PATH = "unused_promo_wav/toietmoit_house_monstercat_promo.wav"
BAND_COLOR = [255, 242, 97]
COLOR_BOTTOM = [255, 242, 97]
COLOR_TOP = [255, 153, 241]

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
    print(f"--- gravity effect applied with decay gradient: [attack = {attack}, max_decay = {max_decay}, min_decay = {min_decay}]")
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



### DEPENDENCY VALIDATION #####################################################

print(f"OpenCV Version: {cv2.__version__}")

# Check if LINE_AA is available (it should be for 60FPS rendering).
if hasattr(cv2, 'LINE_AA'):
    print("Anti-aliasing (LINE_AA) is ready for 60FPS.")
else:
    print("Warning: LINE_AA not found. Check your OpenCV installation.")

print(f"waveform params: [W = {W}, H = {H}, N_BANDS = {N_BANDS}, GUTTER = {GUTTER},")
print(f"BAR_WIDTH = {BAR_WIDTH}, MAX_BAR_HEIGHT = {MAX_BAR_HEIGHT}]")
print("----------------------------------------")


### AUDIO SIGNAL CHAIN ###################################################

### The Essentials #####

# Load audio file into MoviePy
y, sr = librosa.load(AUDIO_PATH)
audio_clip = AudioFileClip(AUDIO_PATH)
duration = audio_clip.duration

# Mel Spectogram transform with range 20-7000Hz
fmin, fmax = 20, 7000
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_BANDS, fmin=fmin, fmax=fmax)
print(f"--- mel spectrogram params: [n_mels = {N_BANDS}, fmin = {fmin}, fmax = {fmax}]")

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
print(f"--- exponential factor to accuentuate low frequencies: [power = {exponent}]")

# TILT EQ to boost treble
tilt_min, tilt_max = 0.8, 2.2
tilt = np.linspace(tilt_min, tilt_max, N_BANDS)
S_norm = (S_norm.T * tilt).T
print(f"--- tilt eq applied with value: [{tilt_min} - {tilt_max}]")



### WAVEFORM PRE-PROCESSING EXECUTION #########################################

# gravity / liquid effect
S_norm = gravity(S_norm)

# wash delay effect
S_norm = wash_delay(S_norm)

# rolling average (less detail in waveform, more smooth)
# S_norm = rolling_average(S_norm, 5)

# rubber-band effect, spread of frequency data across x axis
gaussian_filter = 1.0
S_norm = gaussian_filter1d(S_norm, sigma=gaussian_filter, axis=0) # sigma dictates blur strength (1.0-2.0 usually solid)
print(f"--- gaussian filter applied: [sigma = {gaussian_filter}]")



### WAVEFORM RENDERING FUNCTIONS ###############################################

def draw_rounded_bars(frame: np.ndarray, amplitudes: np.ndarray) -> None:
    """
    Draws bar with rounded corners at designated coordinates.
    Handles curve logic for the rounded corners with given radius.
    """

    corner_radius = 0.5  # 50% rounding
    radius = int(BAR_WIDTH * corner_radius)
    print(f"--- bar rounded corner radius: [{corner_radius}]")


    bar_polygons = []
    for i, amp in enumerate[Any](amplitudes):
        # TODO: Place hard cap for bar height to prevent overflow. Max has gotta
        #       be adjustable or automatic since it's different for each song...
        bar_height = amp *  MAX_BAR_HEIGHT

        # x1 based on num of previous 'i' bars and gutters
        x1 = (i * (BAR_WIDTH + GUTTER)) + 200 # 200 centers the wave form with the borders
        x2 = x1 + BAR_WIDTH

        y1 = 1200
        y2 = y1 - bar_height

        # Ensure radius isn't larger than half the bar width
        bar_width = abs(x2 - x1)
        radius = min(radius, bar_width // 2)
        
        points = []
        points.append([x2, y1]) # bottom right
        points.append([x1, y1]) # bottom left
        points.append([x1, y2 + radius]) # left vertical side up to the start of the curve
        
        # top left curve
        for i in range(180, 270, 10): # 10-degree steps for smoothness
            angle = np.radians(i)
            px = x1 + radius + radius * np.cos(angle)
            py = y2 + radius + radius * np.sin(angle)
            points.append([px, py])
            
        # top right curve
        for i in range(270, 360, 10):
            angle = np.radians(i)
            px = x2 - radius + radius * np.cos(angle)
            py = y2 + radius + radius * np.sin(angle)
            points.append([px, py])
            
        points.append([x2, y2 + radius]) # right vertical side down
        bar_polygons.append(np.array(points, dtype=np.int32))

    # Draw all bars in one step.
    cv2.fillPoly(frame, bar_polygons, BAND_COLOR, lineType=cv2.LINE_AA)


def draw_bars(frame: np.ndarray, amplitudes: np.ndarray) -> None:
    bar_polygons = []
    for i, amp in enumerate[Any](amplitudes):
        # TODO: Place hard cap for bar height to prevent overflow. Max has gotta
        #       be adjustable or automatic since it's different for each song...
        bar_height = amp *  MAX_BAR_HEIGHT

        # x1 based on num of previous 'i' bars and gutters
        x1 = (i * (BAR_WIDTH + GUTTER)) + 200 # 200 centers the wave form with the borders
        x2 = x1 + BAR_WIDTH

        y1 = 1200
        y2 = y1 - bar_height

        # Round points for consistency
        bot_left_pt = (int(round(x1)), int(round(y1)))
        top_right_pt = (int(round(x2)), int(round(y2)))
        bot_right_pt = (int(round(x2)), int(round(y1)))
        top_left_pt = (int(round(x1)), int(round(y2)))

        # Use polygons for LINE_AA anti-aliasing (spatial smoothing).
        rect_points = np.array([bot_left_pt, bot_right_pt, top_right_pt, top_left_pt], dtype=np.int32)

        bar_polygons.append(rect_points)

    # Draw all bars in one step.
    cv2.fillPoly(frame, bar_polygons, BAND_COLOR, lineType=cv2.LINE_AA)


def apply_bloom(frame: np.ndarray) -> np.ndarray:
    # Create waveform mask
    bloom_mask = cv2.GaussianBlur(frame, (25, 25), 0)
    
    # Add back to frame to add glow
    return cv2.addWeighted(frame, 1.0, bloom_mask, 0.5, 0)


def make_frame(t: float) -> np.ndarray:
    """
    This function is called by MoviePy for every frame of the video.
    't' is the current time in seconds.
    """
    
    # create canvas with transparent background
    frame = np.zeros((H, W, 3), dtype='uint8')

    # float index for linear interpolation (temporal smoothing)
    float_idx = (t / duration) * (S_norm.shape[1] - 1)
    idx_floor = int(np.floor(float_idx))
    idx_ceil = int(np.ceil(float_idx))

    # weight of current and next frame index
    weight  = float_idx - idx_floor

    # (1-w)*a + w*b where a is current frame index and b is next frame index
    current_amplitudes = (1 - weight) * S_norm[:, idx_floor] + weight * S_norm[:, idx_ceil]

    # ROUNDED BAR OPTION
    draw_rounded_bars(frame, current_amplitudes)

    # RECTANGULAR BAR OPTION
    # draw_bars(frame, current_amplitudes)

    # Visualizer post-processing
    frame = apply_bloom(frame)

    return frame



### VIDEO RENDER AND EXPORT ###################################################

clip = VideoClip(make_frame, duration=duration)
clip = clip.with_audio(audio_clip)
clip.write_videofile(
    "output_waveform.mov",
    fps=FPS,
    codec="prores_ks",
    ffmpeg_params=[
        "-profile:v", "4",          # '4' is the ProRes 4444 profile
        "-pix_fmt", "yuva444p10le"  # 10-bit YUV + Alpha
    ]
)