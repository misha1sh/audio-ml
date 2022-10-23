import torchaudio.transforms as T_audio
import torch

class Spectrogramer:
    def __init__(self,
                sample_rate,
                n_fft,
                win_length,
                hop_length,
                n_mels):
        n_stft = n_fft // 2 + 1
        center = True
        pad_mode = "reflect"
        norm = "slaney"
        mel_scale = "htk"
        power = 1.0

        self._wave2mel = T_audio.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            pad_mode=pad_mode,
            power=power,
            norm=norm,
            onesided=True,
            n_mels=n_mels,
            mel_scale=mel_scale,
        )

        self._wave2spec = T_audio.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode=pad_mode,
            power=power,
        )

        self._wave2_full_spec = T_audio.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode=pad_mode,
            power=None,
        )

        self._full_spec2_wave = T_audio.InverseSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode=pad_mode,
        )

        self._spec2mel = T_audio.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_stft=n_stft,
            norm=norm,
            mel_scale=mel_scale,
        )

        self._mel2spec = T_audio.InverseMelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_stft=n_stft,
            norm=norm,
            mel_scale=mel_scale,
            tolerance_loss=1e-3,
            tolerance_change=1e-5,
            max_iter=1000
        )


        self._spec2wave = T_audio.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=power,
        )

        # self._mel2wave = torch.nn.Sequential(
        #         self._mel2spec,
        #         self._spec2wave
        #     )

    def wave2spec(self, wave):
        self._wave2spec.to(wave.device)
        return self._wave2spec(wave)

    def wave2mel(self, wave):
        self._wave2mel.to(wave.device)
        return self._wave2mel(wave)

    def spec2mel(self, spec):
        self._spec2mel.to(spec.device)
        return self._spec2mel(spec)

    def mel2spec(self, mel):
        self._mel2spec.to(mel.device)
        return self._mel2spec(mel)

    def spec2wave(self, spec):
        self._spec2wave.to(spec.device)
        return self._spec2wave(spec)

    def mel2wave(self, mel):
        return self.spec2wave(self.mel2spec(mel))

    def spec2wave_with_phase(self, spec, wave_orig):
        wave_orig = wave_orig.to(spec.device)
        self._wave2_full_spec.to(spec.device)
        self._full_spec2_wave.to(spec.device)
        full_spec = self._wave2_full_spec(wave_orig)

        # mag = torch.abs(full_spec) # **power
        mag = spec
        ang = torch.angle(full_spec)

        full_spec = torch.polar(mag, ang)

        return self._full_spec2_wave(full_spec)

    def mel2wave_with_phase(self, mel, wave_orig):
        spec = self.mel2spec(mel)
        return self.spec2wave_with_phase(spec, wave_orig)
