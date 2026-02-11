import numpy as np
import librosa
from moviepy import VideoClip, AudioFileClip
from moviepy.video.VideoClip import ColorClip

# CONFIG
# TODO: make these configurable on script runtime with Textual CLI (GitHub).
W, H = 1080, 1920
FPS = 60
AUDIO_PATH = "unused_promo_wav/blue_solo_monstercat_promo.wav"
N_BANDS = 24
BAND_COLOR = [255, 255, 255]

y, sr = librosa.load(AUDIO_PATH)
audio_clip = AudioFileClip(AUDIO_PATH)
duration = audio_clip.duration

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_BANDS)
S_db = librosa.power_to_db(S, ref=np.max)
S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())

def make_frame(t):
    """
    This function is called by MoviePy for every frame of the video.
    't' is the current time in seconds.
    """
    
    # create canvas with transparent background
    frame = np.zeros((H, W, 3), dtype='uint8')
    
    # Calculate which index in our S_norm matrix corresponds to time 't'
    idx = int((t / duration) * S_norm.shape[1])
    if idx >= S_norm.shape[1]: idx = S_norm.shape[1] - 1
    
    current_amplitudes = S_norm[:, idx]
    
    bar_width = W // N_BANDS
    for i, amp in enumerate(current_amplitudes):
        # max bar height is 700px
        bar_height = int(amp * 700)
        
        x1 = i * bar_width
        y1 = H // 2 + 350  # Center it vertically
        x2 = x1 + bar_width - 3 # -3 for a small gap between bars
        y2 = y1 - bar_height
        
        # use NumPy slicing to "draw" the rectangle on the array
        frame[y2:y1, x1:x2] = BAND_COLOR
        
    return frame

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