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

        # Noise settings for ocean wave-like sound
        self.noise_buffer = np.zeros(self.fs)  # 1-second buffer for noise
        self.lowcut = 100.0  # Lower frequency cut-off for filtering
        self.highcut = 1200.0  # Upper frequency cut-off for filtering

        # Precompute filter coefficients
        self.b, self.a = self.butter_bandpass(self.lowcut, self.highcut, self.fs)
        # Initialize filter state
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)
        # Initialize phase for amplitude modulation
        self.sound_phase = 0

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

        # Calculate amplitude modulation
        breathing_rate_hz = self.last_breathing_rate / 60  # Convert bpm to Hz
        amplitude = 0.5 + 0.5 * np.sin(2 * np.pi * breathing_rate_hz * t)

        # Generate white noise
        noise = np.random.normal(0, 1, frames)

        # Apply bandpass filter with state preservation
        filtered_noise, self.zi = lfilter(self.b, self.a, noise, zi=self.zi)

        # Modulate amplitude
        modulated_noise = amplitude * filtered_noise

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
