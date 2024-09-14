import numpy as np
import time
from PySide6.QtCore import QObject
import sounddevice as sd
from scipy.signal import butter, lfilter

class Pacer(QObject):
    def __init__(self):
        super().__init__()

        theta = np.linspace(0, 2 * np.pi, 40)
        self.cos_theta = np.cos(theta)
        self.sin_theta = np.sin(theta)
        self.last_breathing_rate = 1
        self.phase = 0

        # Sound settings
        self.fs = 44100  # Sampling frequency (Hz)
        self.sound_phase = 0  # Phase accumulator for sound wave
        self.stream = sd.OutputStream(
            samplerate=self.fs,
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()

        # Adjusted noise settings for ocean wave-like sound
        self.lowcut = 50.0   # Lower frequency cut-off for filtering
        self.highcut = 1000.0  # Upper frequency cut-off for filtering

        # Precompute bandpass filter coefficients for ocean wave effect
        self.b_band, self.a_band = self.butter_bandpass(self.lowcut, self.highcut, self.fs, order=4)
        self.zi_band = np.zeros(max(len(self.a_band), len(self.b_band)) - 1)

        # Precompute coefficients for pink noise generation
        self.b_pink, self.a_pink = butter(1, 0.1, btype='low')  # Adjust the cutoff for pink noise
        self.zi_pink = np.zeros(max(len(self.a_pink), len(self.b_pink)) - 1)

    def breathing_pattern(self, breathing_rate, time):
        """Returns radius of pacer disk.

        Radius is modulated according to sinusoidal breathing pattern
        and scaled between 0 and 1.
        """
        if breathing_rate != self.last_breathing_rate:
            # Maintaining continuity by adjusting phase when breathing rate is changed
            self.phase = time - self.last_breathing_rate * (time - self.phase) / breathing_rate
            self.last_breathing_rate = breathing_rate

        radius = 0.5 + 0.5 * np.sin(2 * np.pi * breathing_rate / 60 * (time - self.phase))
        return radius

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        """Create a bandpass filter."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def audio_callback(self, outdata, frames, time_info, status):
        """Callback function for streaming audio."""
        # Get time array for this buffer
        t = (np.arange(frames) + self.sound_phase) / self.fs
        self.sound_phase += frames

        # Calculate amplitude modulation with minimum volume level
        breathing_rate_hz = self.last_breathing_rate / 60  # Convert bpm to Hz
        amplitude_mod = 0.75 + 0.25 * np.sin(2 * np.pi * breathing_rate_hz * t)

        # Generate white noise
        white_noise = np.random.normal(0, 1, frames)

        # Generate pink noise by filtering white noise
        pink_noise, self.zi_pink = lfilter(self.b_pink, self.a_pink, white_noise, zi=self.zi_pink)

        # Apply bandpass filter with state preservation
        filtered_noise, self.zi_band = lfilter(self.b_band, self.a_band, pink_noise, zi=self.zi_band)

        # Modulate amplitude
        modulated_noise = amplitude_mod * filtered_noise

        # Normalize to prevent clipping
        max_val = np.max(np.abs(modulated_noise))
        if max_val > 1.0:
            modulated_noise = modulated_noise / max_val

        outdata[:] = modulated_noise.reshape(-1, 1)

    def update(self, breathing_rate):
        """Update radius of pacer disc.

        Make current disk radius a function of real time (i.e., don't
        precompute radii with fixed time interval) in order to compensate for
        jitter or delay in QTimer calls.
        """
        radius = self.breathing_pattern(breathing_rate, time.time())
        x = radius * self.cos_theta
        y = radius * self.sin_theta
        return (x, y)

    def stop_sound(self):
        """Stop the audio stream when done."""
        self.stream.stop()
        self.stream.close()
