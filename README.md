### An Audio Visualizer for Music Promo Videos...

I just feel like an audio visualizer (kinda like monstercat's) on some short or
long-form content would go so crazy. Video in the background (or black screen
and we color or gradient the visualizer...) with the "waveform" in front?? cmon
now.

I need to promote some music projects I'm working on, will never give a single
dollar to Adobe, and am lowkey interested in audio (or audiovisual) programming,
no matter how rudimentary!!

#### What's Going On Here?

Moviepy calls make_frame for each frame in the video (with a given FPS, codec,
etc.) and make_frame draws the pixels for each of them! This is compiled into a
VideoClip, which is exported as a file.

- Uses a Mel Spectrogram (Mel Filter Bank) to shape the input audio signal to
  "fit" human hearing and look nicer on display. Cuts out a lot of high
  frequencies in audio signal that humans can't hear and/or don't care about!

Librosa reads input audio file as well, performs Short-Time Fourier Transform
(STFT), divide frequencies into bins for each time window in the audio, and
reflect the amplitude of those bins visually via a NumPy 2D array (filling in
each pixel to draw a bar on the screen). That's what those bars on the screen
are! This is waveform programming lol

going for the monstercat (aka the goats) look for the sake of nostalgia

#### Matrix ASCII mask (optional)

Static grid of CMatrix-inspired characters (katakana, digits, letters) can modulate
bar brightness while keeping the gradient colors — a digital/code look without
animating or calling the `cmatrix` binary.

1. Uncomment `frame = apply_matrix_mask(frame, mask)` in `make_frame` (after the gradient composite). The glyph field is built lazily on the first frame.
2. Tune density via `MATRIX_CHARS_PER_BAR_ROW` (`1` = fewer, larger glyphs per bar row) and `MATRIX_CELL_HEIGHT_RATIO` (higher = fewer rows). Also `MATRIX_CHARSET`, `MATRIX_LUM_CUTOFF`, and `MATRIX_SEED`. Cell width is derived from bar width; height follows the ratio.

Quick preview at the loudest frame: uncomment the matching lines in `debug_bar_geometry()` then run `python3 waveform.py --debug-bars` to write `debug_bar_matrix.png`.

#### Matrix debug (charset / glyph verification)

If bars show thin lines instead of letters, glyphs were likely clipped (text larger than the cell). Run:

```bash
python3 waveform.py --debug-matrix
```

Check outputs in order:

1. `debug_matrix_atlas.png` — every character in `MATRIX_CHARSET` must be readable in its cell.
2. `debug_matrix_lum.png` — full-frame static glyph grid.
3. `debug_matrix_on_bars.png` — matrix on bars **without** CRT/chromatic (ground truth).
4. `debug_matrix_with_crt_chromatic.png` — same as `make_frame` post-effects (scanlines/chromatic can obscure glyphs).

Iterate on `MATRIX_CHARS_PER_BAR_ROW`, `MATRIX_CELL_HEIGHT_RATIO`, and `MATRIX_CHARSET` until (1) looks correct, then (3).

---

<br>

This repo will have a lot of audio for now that I'm gonna be using for insta,
tiktok, youtube promos shortly... Eventually if this thing is novel enough (and
has a TUI) maybe I'll make it more open-source friendly

---

#### TODO or something

- Make the transition between frames on the waveform smoother. [DONE]
- Bars thicker and less of them? [DONE]
- Full audio signal chain to "match" Monstercat somewhat [DONE for now]
- CPU limiting (right now uses 95%!!! no way) [WIP]
- Amplitude hard cap (prevent bar overflow) [WIP]
- Add color to the waveform, and gradients??? Maybe the gradient could be
  moving... panning linearly as an overlay or pass-through or something. [WIP]
- bloom [WIP]
- TUI with Rich or Textual
- In the future it could all be configurable.
  - Be able to adjust the frequency bins to fit different types of audio for
    optimal visual clarity. A guitar solo will not render very interestingly on
    this visualizer right now...
