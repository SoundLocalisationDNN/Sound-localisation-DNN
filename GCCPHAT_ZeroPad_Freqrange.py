import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def gcc_phat(sig1, sig2, fs, max_delay=0.00065, interp_factor=4,
             fmin=50.0, fmax=8000.0):
    """
    Compute the GCC-PHAT between two signals with zero-padding, interpolation,
    and a band-pass of [fmin, fmax] Hz on the PHAT spectrum.
    
    Parameters:
        sig1, sig2: 1D numpy arrays for the two channels.
        fs: Sampling frequency in Hz.
        max_delay: Maximum delay (in seconds) to consider.
        interp_factor: Factor by which to interpolate the time-domain signal.
        fmin, fmax: Minimum and maximum frequencies (Hz) to keep in PHAT.
        
    Returns:
        lags_limited: Array of lag values (in samples of the original rate)
                      within ±max_delay.
        cross_corr_limited: Cross-correlation values corresponding to lags_limited.
        effective_fs: The effective sampling rate after interpolation.
        n_fft: FFT length used before interpolation.
    """
    # Trim to equal length & window
    n = min(len(sig1), len(sig2))
    sig1 = sig1[:n] * np.hanning(n)
    sig2 = sig2[:n] * np.hanning(n)

    # FFT length (no zero-padding yet)
    n_fft = n

    # Compute FFTs
    SIG1 = np.fft.fft(sig1, n=n_fft)
    SIG2 = np.fft.fft(sig2, n=n_fft)

    # Cross-spectrum + PHAT weight
    R = SIG1 * np.conj(SIG2)
    eps = 1e-10
    R_phat = R / (np.abs(R) + eps)

    # --- Band-pass in the frequency domain ---
    # Build frequency vector for FFT bins
    freqs = np.fft.fftfreq(n_fft, d=1.0/fs)
    # Mask out bins with |f| < fmin or |f| > fmax
    band_mask = (np.abs(freqs) >= fmin) & (np.abs(freqs) <= fmax)
    R_phat *= band_mask

    # Zero-pad the PHAT spectrum for interpolation
    interp_n_fft = interp_factor * n_fft
    R_phat_padded = np.zeros(interp_n_fft, dtype=complex)
    half = n_fft // 2
    # keep DC to Nyquist
    R_phat_padded[:half+1] = R_phat[:half+1]
    # keep negative freqs
    R_phat_padded[-half:] = R_phat[-half:]

    # Cross-correlation via IFFT & shift
    cross_corr = np.fft.ifft(R_phat_padded)
    cross_corr = np.real(cross_corr)
    cross_corr = np.fft.fftshift(cross_corr)

    # Build lags (in original-sample units)
    effective_fs = fs * interp_factor
    lags = np.arange(-interp_n_fft//2, interp_n_fft//2)
    lags_in_orig_fs = lags / interp_factor

    # Limit to ±max_delay
    max_samples = int(effective_fs * max_delay)
    valid = np.abs(lags) <= max_samples
    lags_limited = lags_in_orig_fs[valid]
    cross_corr_limited = cross_corr[valid]

    return lags_limited, cross_corr_limited, effective_fs, n_fft

if __name__ == '__main__':

    # Read the stereo WAV file (update filename as needed)
    fs, data = wavfile.read(r"path")

    if data.ndim < 2 or data.shape[1] < 2:
        raise ValueError("Input WAV file must have at least two channels.")

    sig1 = data[:, 0]
    sig2 = data[:, 1]

    # Upsampling factor
    interp_factor = 4

    # Compute GCC-PHAT cross-correlation with upsampling
    lags_limited, cc_limited, effective_fs, n_fft = gcc_phat(
        sig1, sig2, fs, max_delay=0.00065, interp_factor=interp_factor
    )

    # Find the peak in the cross-correlation
    peak_idx = np.argmax(np.abs(cc_limited))
    peak_lag = lags_limited[peak_idx]
    peak_val = cc_limited[peak_idx]

    # Convert peak lag to time in milliseconds (using original sampling rate)
    delay_ms = peak_lag / fs * 1000


    print(f"Original sampling rate: {fs} Hz")
    print(f"Effective sampling rate after interpolation: {effective_fs} Hz (×{interp_factor})")
    print(f"FFT length before interpolation: {n_fft}")
    print(f"FFT length after interpolation: {n_fft * interp_factor}")
    print(f"Peak delay: {delay_ms:.4f} ms")


    # Convert lag indices to time in milliseconds for plotting
    lag_times_ms = lags_limited / fs * 1000
    peak_time_ms = peak_lag / fs * 1000


    # Plot the cross-correlation, the peak, and the quadratic interpolation curve
    plt.figure(figsize=(8, 4))

    # Main plot of the cross-correlation
    # plt.subplot(211)
    plt.plot(lag_times_ms, cc_limited, 'b-', linewidth=1, label='Cross-Correlation')
    plt.plot(peak_time_ms, peak_val, 'bo', label=f'Peak ({delay_ms:.4f} ms)')
    plt.title(f'GCC-PHAT with Zero-Padding (Interpolated {interp_factor}×)')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.legend()

    # Zoomed-in plot around the peak
    # plt.subplot(212)
    # zoom_range = 0.05  # ms around the peak
    # valid_indices = np.where(np.abs(lag_times_ms - peak_time_ms) < zoom_range)
    # plt.plot(lag_times_ms[valid_indices], cc_limited[valid_indices], 'b.-', label='Cross-Correlation')
    # plt.plot(peak_time_ms, peak_val, 'bo', label=f'Peak ({delay_ms:.4f} ms)')
    # plt.title('Zoomed View Around Peak')
    plt.xlabel('Delay [ms]')
    # plt.ylabel('Correlation')
    # plt.grid(True)
    # plt.legend()

    plt.tight_layout()
    plt.show()