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

---

<br>

This repo will have a lot of audio for now that I'm gonna be using for insta,
tiktok, youtube promos shortly... Eventually if this thing is novel enough (and
has a TUI) maybe I'll make it more open-source friendly

---

#### TODO or something

- Make the transition between frames on the waveform smoother.
- Bars thicker, and less of them? In the future it could all be configurable.
- TUI with Rich or Textual
- Add color to the waveform, and gradients??? Maybe the gradient could be
  moving... panning linearly as an overlay or pass-through or something.
- Be able to adjust the frequency bins to fit different types of audio for
  optimal visual clarity. A guitar solo will not render very interestingly on
  this visualizer right now...
