from typing import Any
import cv2
import numpy as np
import librosa
from moviepy import VideoClip, AudioFileClip


### DEPENDENCY VALIDATION #####################################################

print(f"OpenCV Version: {cv2.__version__}")

# Check if LINE_AA is available (it should be for 60FPS rendering).
if hasattr(cv2, 'LINE_AA'):
    print("Anti-aliasing (LINE_AA) is ready for 60FPS.")
else:
    print("Warning: LINE_AA not found. Check your OpenCV installation.")



#### CONFIGURATION ############################################################

# TODO: make these configurable on script runtime with Textual CLI (GitHub).
W, H = 1080, 1920
FPS = 60
AUDIO_PATH = "unused_promo_wav/toietmoit_house_monstercat_promo.wav"
N_BANDS = 24
BAND_COLOR = [255, 255, 255]

y, sr = librosa.load(AUDIO_PATH)
audio_clip = AudioFileClip(AUDIO_PATH)
duration = audio_clip.duration

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_BANDS)
S_db = librosa.power_to_db(S, ref=np.max)
S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())



### WAVEFORM PRE-PROCESSING FUNCTIONS #########################################

def gravity(S_norm: np.ndarray, attack: float=0.8, decay: float=0.5):
    """
    Lessens amplitude variation between frames based on above coefficients.
    The attack and decay rates can be adjusted to control the speed of the effect for amplitude increase/decrease.

    Same deal with rolling average, this can strip detail from the waveform.
    """
    smoothed_S = np.zeros_like(S_norm)

    for t in range(1, S_norm.shape[1]):
        for b in range(S_norm.shape[0]):
            target = S_norm[b, t]
            prev = smoothed_S[b, t-1]
        
            if target > prev:
                smoothed_S[b, t] = prev + (target - prev) * attack
            else:
                smoothed_S[b, t] = prev - (prev - target) * decay
    return smoothed_S

def rolling_average(data: np.ndarray, window_size: int=3):
    """
    Applies a rolling average to the signal data, modifying values in place based on the
    neighboring values within the window size.
    
    In practice, this makes the waveform a whole lot less appealing IMO...
    """
    kernel = np.ones(window_size) / window_size
    for i in range (data.shape[0]):
        data[i, :] = np.convolve(data[i, :], kernel, mode='same')
    return data



### WAVEFORM PRE-PROCESSING EXECUTION #########################################

S_norm = gravity(S_norm)
# S_norm = rolling_average(S_norm, 5)



### WAVEFORM RENDERING FUNCTION ###############################################

def make_frame(t):
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
    
    # keep width as float (not //) for spatial smoothing (LINE_AA)
    bar_width = W / N_BANDS

    # Vector rasterization for frame at time 't'.
    bar_polygons = []
    for i, amp in enumerate[Any](current_amplitudes):
        # max bar height is 700px
        bar_height = amp * 700
        x1 = i * bar_width
        y1 = (H // 2) + 350  # Center it vertically
        x2 = (x1 + bar_width) - 5 # 5px gap between bars
        y2 = y1 - bar_height
        
        rect_points = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.int32)

        bar_polygons.append(rect_points)

    # Draw all bars in one step.
    cv2.fillPoly(frame, bar_polygons, BAND_COLOR, lineType=cv2.LINE_AA)

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