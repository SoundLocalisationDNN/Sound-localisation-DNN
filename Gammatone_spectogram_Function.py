import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from gammatone.gtgram import gtgram


def hz_to_erb(f):
    """Convert frequency in Hz to ERB number."""
    return 21.4 * np.log10(4.37 * f / 1000 + 1)


def erb_to_hz(erb):
    """Convert ERB number to frequency in Hz."""
    return (10**(erb / 21.4) - 1) * 1000 / 4.37


def erb_space(fmin, fmax, num):
    """
    Generate 'num' center frequencies between fmin and fmax 
    that are linearly spaced on the ERB scale.
    """
    erb_min = hz_to_erb(fmin)
    erb_max = hz_to_erb(fmax)
    erbs = np.linspace(erb_min, erb_max, num)
    freqs = erb_to_hz(erbs)
    return freqs


def compute_gammatone_spectrogram(audio_segment, sample_rate, fmin, fmax, 
                                  window_time=0.025, hop_time=0.0125, num_filters=100, to_db=True):
    """
    Compute a gammatone filter bank spectrogram for each channel of the input audio segment,
    optionally converting to dB scale.
    
    Parameters:
      audio_segment (np.ndarray): Audio data array (mono or stereo).
      sample_rate (int): Sampling rate.
      fmin (float): Minimum frequency for the filter bank.
      fmax (float): Maximum frequency for the filter bank.
      window_time (float): Window length (s) for each frame (default 25 ms).
      hop_time (float): Hop time (s) between frames (default 12.5 ms).
      num_filters (int): Number of gammatone filters.
      to_db (bool): If True, convert amplitude to dB scale (20*log10). Defaults to True.
    
    Returns:
      spectrograms_db (list of np.ndarray): One spectrogram per channel in dB (if to_db), else linear.
      center_freqs (np.ndarray): Center frequencies for each filter (Hz).
    """
    # Ensure audio_segment is 2D: (n_samples, n_channels)
    if audio_segment.ndim == 1:
        audio_segment = audio_segment.reshape(-1, 1)

    # Compute center frequencies on the ERB scale
    center_freqs = erb_space(fmin, fmax, num_filters)

    spectrograms = []
    eps = 1e-8
    for ch in range(audio_segment.shape[1]):
        channel_data = audio_segment[:, ch]
        # Create gammatone spectrogram via gtgram
        spec = gtgram(channel_data, sample_rate, window_time, hop_time, num_filters, fmin, fmax)
        if to_db:
            spec = 20 * np.log10(spec + eps)
        spectrograms.append(spec)

    return spectrograms, center_freqs


def plot_gammatone_spectrograms(spectrograms, center_freqs, hop_time, fmin, fmax):
    """
    Plot the gammatone spectrograms in dB on a corrected frequency axis.
    """
    num_channels = len(spectrograms)
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 4 * num_channels))
    if num_channels == 1:
        axes = [axes]

    num_filters = len(center_freqs)
    num_frames = spectrograms[0].shape[1]

    # Compute time edges (one more than the number of frames)
    time_edges = np.linspace(0, num_frames * hop_time, num_frames + 1)

    # Compute frequency edges from the center frequencies
    freq_edges = np.zeros(num_filters + 1)
    freq_edges[1:-1] = (center_freqs[:-1] + center_freqs[1:]) / 2.0
    freq_edges[0] = center_freqs[0] - (center_freqs[1] - center_freqs[0]) / 2.0
    freq_edges[-1] = center_freqs[-1] + (center_freqs[-1] - center_freqs[-2]) / 2.0

    for idx, spec in enumerate(spectrograms):
        ax = axes[idx]
        pcm = ax.pcolormesh(time_edges, freq_edges, spec, shading='auto', cmap='viridis')
        ax.set_title(f'Channel {idx+1} Gammatone Spectrogram (dB)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(fmin, fmax)
        fig.colorbar(pcm, ax=ax, label='Amplitude (dB)')

    plt.tight_layout()
    plt.show()


# === Example usage ===
if __name__ == '__main__':
    fs = 48000           # Sampling rate in Hz
    duration = 0.5       # Duration in seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Generate a chirp: frequency increasing from 4000 Hz to 8000 Hz
    chirp_signal = chirp(t, f0=4000, f1=8000, t1=duration, method='linear')

    # Define filter bank and spectrogram parameters
    fmin = 50          # Minimum frequency (Hz)
    fmax = 10000       # Maximum frequency (Hz)
    window_time = 0.05 # 50 ms Hamming window
    hop_time = 0.025   # 25 ms hop time

    # Compute the spectrogram (mono signal) in dB
    spectrograms_db, center_freqs = compute_gammatone_spectrogram(
        chirp_signal, fs, fmin, fmax, window_time=window_time,
        hop_time=hop_time, num_filters=100, to_db=True
    )

    # Plot the dB spectrogram
    plot_gammatone_spectrograms(spectrograms_db, center_freqs, hop_time, fmin, fmax)
