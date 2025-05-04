import os 
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
from scipy.signal import butter, filtfilt

windowsize = 0.5

def play_segment(segment, sample_rate):
    """
    Play an audio segment using sounddevice.
    """
    print("Playing audio segment...")
    sd.play(segment, sample_rate)
    sd.wait()

def bandpass_filter(data, sample_rate, lowcut=50, highcut=8000, order=5):
    """
    Apply a Butterworth bandpass filter to the data.
    """
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    # Apply filter along axis=0 (samples) - works for mono or multi-channel
    return filtfilt(b, a, data, axis=0)

def compute_band_energy(segment, sample_rate, low_freq=75, high_freq=5000, plot=False):
    """
    Compute the spectral energy in the specified frequency band.
    
    Parameters:
      segment    : Audio segment (numpy array); shape (N,) for mono or (N, channels) for multi-channel.
      sample_rate: Sampling rate in Hz.
      low_freq   : Lower frequency bound (default 100 Hz).
      high_freq  : Upper frequency bound (default 4000 Hz).
      plot       : If True, plot the FFT magnitude spectrum for the first channel.
      
    Returns:
      band_energy: The total spectral energy within the [low_freq, high_freq] band.
    """
    # Compute FFT along the time axis (axis=0)
    fft_vals = np.fft.rfft(segment, axis=0)
    # Compute the frequency bins based on the number of time samples
    freqs = np.fft.rfftfreq(segment.shape[0], d=1/sample_rate)
    
    # Compute the energy spectrum (magnitude squared)
    energy_spectrum = np.abs(fft_vals)**2
    
    # Create a boolean mask for frequencies within the desired band
    band_indices = (freqs >= low_freq) & (freqs <= high_freq)
    
    # Sum the energy over the frequency band; if multi-channel, sum over all channels.
    if segment.ndim == 1:
        band_energy = np.sum(energy_spectrum[band_indices])
    else:
        band_energy = np.sum(energy_spectrum[band_indices, :])
    
    # Plot the FFT if requested (plotting only the first channel if multi-channel)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        # If multi-channel, plot the first channel
        if segment.ndim > 1:
            plt.plot(freqs, np.abs(fft_vals[:, 0]), label='Channel 0')
        else:
            plt.plot(freqs, np.abs(fft_vals), label='FFT Magnitude')
        plt.axvspan(low_freq, high_freq, color='red', alpha=0.3, label='Energy Band')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title("FFT Magnitude Spectrum")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return band_energy



def process_wav_files(folder_path ,windowsize = 0.5, energy_threshold = 0.2e20):
    kept_segments = []
    kept_labels = []
    kept_sample_rates = []
    kept_energies = []
    
    removed_segments = []
    removed_labels = []
    removed_sample_rates = []
    removed_energies = []
    
    # Regular expression to capture the angle from filenames like "angle_45_20230315.wav"
    angle_pattern = re.compile(r'angle_([-+]?[0-9]*\.?[0-9]+)_')
    
    # Set a threshold for spectral energy in the 100-4000Hz band.
    #energy_threshold = 0 #0.2e20 # Adjust this threshold as needed
    
    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            match = angle_pattern.search(filename)
            if not match:
                print(f"unexpected naming (set label to zero): {filename}")
                angle = float(0)
            else:
                angle = float(match.group(1))
            file_path = os.path.join(folder_path, filename)
            sample_rate, data = wavfile.read(file_path)
            
            if len(data) <= 2 * sample_rate:
                print(f"File {filename} is too short to remove 1 second from both ends. Skipping.")
                continue

            # Remove first and last second of the recording
            trimmed_data = data[sample_rate:len(data)] #-sample_rate
            seg_length = int(windowsize * sample_rate)
            total_samples = len(trimmed_data)
            num_segments = total_samples // seg_length
            
            for i in range(num_segments):
                start = i * seg_length
                end = start + seg_length
                segment = trimmed_data[start:end]
                
                if len(segment) == seg_length:
                    # Apply bandpass filter from 50Hz to 8000Hz
                    segment = bandpass_filter(segment, sample_rate, lowcut=50, highcut=8000, order=3)
                    
                    # Compute spectral energy in the 100-4000Hz band
                    band_energy = compute_band_energy(segment, sample_rate)
                    
                    # Normalize using a single scaling factor across all channels
                    # max_val = np.max(np.abs(segment))
                    # if max_val > 0:
                    #     segment = segment / max_val
                    
                    if band_energy < energy_threshold:
                        removed_segments.append(segment)
                        removed_labels.append(angle)
                        removed_sample_rates.append(sample_rate)
                        removed_energies.append(band_energy)
                    else:
                        kept_segments.append(segment)
                        kept_labels.append(angle)
                        kept_sample_rates.append(sample_rate)
                        kept_energies.append(band_energy)
                    

    
    # Return both kept and removed segments along with their energies and labels
    return (np.array(kept_segments), np.array(kept_labels), kept_sample_rates, kept_energies,
            np.array(removed_segments), np.array(removed_labels), removed_sample_rates, removed_energies)

def plot_segment(segment, sample_rate, label, segment_index):
    """
    Plot a given audio segment using Matplotlib.
    """
    t = np.linspace(0, len(segment) / sample_rate, num=len(segment))
    plt.figure(figsize=(10, 4))
    plt.plot(t, segment)
    plt.title(f"Audio Segment {segment_index} (Label: {label})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_band_energies(kept_labels, kept_energies, removed_labels, removed_energies):
    """
    Plot the band energies for kept and removed segments using their labels.
    """
    plt.figure(figsize=(10, 5))
    # Plot kept segments with green markers
    plt.scatter(kept_labels, kept_energies, color='green', label='Kept Segments')
    # Plot removed segments with red markers
    plt.scatter(removed_labels, removed_energies, color='red', label='Removed Segments')
    
    plt.xlabel("Label (Angle)")
    plt.ylabel("Band Energy (100-4000 Hz)")
    plt.title("Spectral Energy of Segments vs. Labels")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_energy_histogram(kept_energies, removed_energies):
    """
    Plot histograms of the band energies for kept and removed segments.
    """
    plt.figure(figsize=(10, 5))
    plt.hist(kept_energies, bins=100, alpha=0.7, label='Kept Segments', color='green')
    plt.hist(removed_energies, bins=100, alpha=0.7, label='Removed Segments', color='red')
    plt.xlabel("Band Energy (100-4000 Hz)")
    plt.ylabel("Number of Segments")
    plt.title("Histogram of Spectral Energy")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Example usage
    folder = r'testfolder\DataSet1'  # Replace with your folder path
    (kept_segments, kept_labels, kept_sample_rates, kept_energies,
    removed_segments, removed_labels, removed_sample_rates, removed_energies) = process_wav_files(folder)

    print(f"Total removed segments extracted: {len(removed_segments)}")

    # Plot the band energies for all segments using the labels
    print("Plotting band energies for all segments with labels...")
    plot_band_energies(kept_labels, kept_energies, removed_labels, removed_energies)

    # Option 1: Plot the first kept segment as an example
    if len(kept_segments) > 0:
        print("Plotting the first kept segment...")
        plot_segment(kept_segments[0], kept_sample_rates[0], kept_labels[0], 1)
        for idx, segment in enumerate(kept_segments):
            print(f"Plotting kept segment {idx+1} with label {kept_labels[idx]}")
            plot_segment(segment, kept_sample_rates[idx], kept_labels[idx], idx+1)
            play_segment(segment, kept_sample_rates[idx])

    plot_energy_histogram(kept_energies, removed_energies)




