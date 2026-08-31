from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import pygame


class AudioAlarm:
    def __init__(self, sound_path: str | Path | None = None):
        self.sound_path = Path(sound_path) if sound_path else Path("assets/alarm.wav")
        self._fallback_generated = False
        self._channel = None

        if self.sound_path.exists():
            pygame.mixer.init()
            self._sound = pygame.mixer.Sound(str(self.sound_path))
            self._sound_loaded = True
        else:
            self._sound = None
            self._sound_loaded = False
            generated = self._generate_fallback_wav(self.sound_path)
            if generated:
                pygame.mixer.init()
                self._sound = pygame.mixer.Sound(str(self.sound_path))
                self._sound_loaded = True

    def _generate_fallback_wav(self, path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            frequency = 880
            duration = 1.2
            sample_rate = 22050
            total_samples = int(sample_rate * duration)
            amplitude = 32767
            samples = np.sin(2 * np.pi * frequency * np.arange(total_samples) / sample_rate)
            pcm = np.int16(samples * amplitude)

            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm.tobytes())
            return True
        except Exception:
            return False

    def play(self):
        if not self._sound_loaded or self._sound is None:
            return False
        try:
            if not pygame.mixer.get_busy():
                self._sound.play(loops=-1)
            return True
        except Exception:
            return False

    def stop(self):
        if self._sound is None:
            return
        try:
            self._sound.stop()
        except Exception:
            pass

    def is_available(self):
        return self._sound_loaded
