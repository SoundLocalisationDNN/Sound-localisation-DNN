import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import math
import concurrent.futures 
import random


# Import attached functions
from Gammatone_spectogram_Function import compute_gammatone_spectrogram
from GCCPHAT_ZeroPad_Freqrange import gcc_phat
from Wav_splitter import process_wav_files, bandpass_filter



def augment_audio_data(segment, sr, angle):
    """
    Augment audio data with various transformations to improve model generalization.
    Randomly applies realistic augmentations while retaining ITD and ILD for sound localization.
    Returns a list of tuples (augmented_segment, augmented_angle).
    """
    augmented_data = []
    # Always include the original data.
    augmented_data.append((segment, angle))
    

    augmentations = [

        lambda seg: apply_random_noise(seg, noise_type="pink", snr_range=(30, 50)),
        lambda seg: apply_random_noise(seg, noise_type="white", snr_range=(30, 50))
    ]
    
    # For each augmentation, apply it with a 30% probability.
    for aug in augmentations:
        if random.random() < 0:
            augmented_segment = aug(segment)
            augmented_data.append((augmented_segment, angle))
    
    return augmented_data


def apply_random_noise(
    segment: np.ndarray,
    noise_type: str = "white",
    snr_range: tuple[float, float] = (35, 50)
) -> tuple[np.ndarray, float]:
    """
    Apply additive noise (white or pink) to a 2-channel audio segment at a random SNR.

    :param segment: 2D numpy array of shape (N, 2) with float samples (e.g., in [-1, 1]).
    :param noise_type: 'white' or 'pink'.
    :param snr_range: tuple (min_snr_db, max_snr_db) for random SNR selection in dB.
    :returns: (noisy_segment, snr_used_db)

    The function generates independent noise patterns per channel, selects a random SNR
    within the given range, scales the noise to achieve that SNR, adds it to the segment,
    and returns the noisy audio plus the chosen SNR value.
    """
    if segment.ndim != 2 or segment.shape[1] != 2:
        raise ValueError("Input segment must be a 2-channel array of shape (N, 2)")

    N, C = segment.shape
    if noise_type not in ("white", "pink"):
        raise ValueError("noise_type must be 'white' or 'pink'")

    # 1. Select random SNR
    min_snr, max_snr = snr_range
    snr_db = float(np.random.uniform(min_snr, max_snr))

    # 2. Generate noise
    # White noise: normal distribution
    noise = np.random.randn(N, C).astype(segment.dtype)

    if noise_type == "pink":
        # Shape spectrum by 1/sqrt(f) for pink noise
        f = np.fft.rfftfreq(N, d=1.0)
        f[0] = f[1] if N > 1 else 1.0
        scale = 1.0 / np.sqrt(f)
        for ch in range(C):
            X = np.fft.rfft(noise[:, ch]) * scale
            noise[:, ch] = np.fft.irfft(X, n=N)
        # Normalize to unit variance
        noise = noise / np.std(noise, axis=0, keepdims=True)

    # 3. Scale noise to desired SNR
    sig_power   = np.mean(segment**2, axis=0)
    noise_power = np.mean(noise**2, axis=0)
    desired_noise_power = sig_power / (10**(snr_db / 10))
    scale_factors = np.sqrt(desired_noise_power / noise_power)

    noisy_segment = segment + noise * scale_factors
    noisy_segment = bandpass_filter(noisy_segment, 44100, lowcut=50, highcut=8000, order=3)

    return noisy_segment



# =============================================================================
# Helper function for per-segment processing (to be run in parallel)
# =============================================================================

def process_segment(seg, sr, angle, window_time, hop_time, fmin, fmax):
    """
    Process one audio segment:
      - Check for stereo channels
      - Compute gammatone spectrograms for each channel
      - Compute GCC-PHAT features between channels
      - Convert angle to [sin(angle), cos(angle)]
    Returns a tuple (features, label) or None if invalid.
    Includes timers to log the time spent on sub-tasks.
    """
    start_total = time.time()
    if seg.ndim != 2 or seg.shape[1] != 2:
        # Return None if the segment does not have two channels.
        return None

    
    specs, center_freqs = compute_gammatone_spectrogram(seg, sr, fmin, fmax,
                                                         window_time=window_time,
                                                         hop_time=hop_time)
    

    # Time the GCC-PHAT computation
    sig1 = seg[:, 0]
    sig2 = seg[:, 1]
    lags, cc, effective_fs, n_fft = gcc_phat(sig1, sig2, sr)
    
    total_time = time.time() - start_total
    # Log timings for this segment (prints from subprocesses may appear asynchronously)
    print(f"Segment processed in {total_time:.3f}s ")

    # Package features (spectrograms and the cc vector)
    feat = {
        "gammatone_left": specs[0],
        "gammatone_right": specs[1],
        "gcc": cc
    }
    # Convert angle to radians and then to sine and cosine representation
    rad = math.radians(angle)
    label_vec = [math.sin(rad), math.cos(rad)]
    return feat, label_vec

# =============================================================================
#Data Preprocessing Function
# =============================================================================

def preprocess_audio_segments(segments, sample_rates, angles, window_time, hop_time, fmin, fmax):
    """
    For each audio segment:
      - Compute gammatone spectrogram for each channel
      - Compute GCC-PHAT features using the two channels
      - Convert the azimuth angle (in degrees) to sin and cos components
    Returns:
      features_list: List of dictionaries with keys "gammatone_left", "gammatone_right", "gcc"
      labels_list: List of [sin(angle), cos(angle)]
    This version uses parallel processing and includes timers.
    """
    features_list = []
    labels_list = []
    
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for seg, sr, angle in zip(segments, sample_rates, angles):
            futures.append(executor.submit(process_segment, seg, sr, angle,
                                             window_time, hop_time, fmin, fmax))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                feat, label_vec = result
                features_list.append(feat)
                labels_list.append(label_vec)
    total_time = time.time() - start
    print(f"Total feature extraction time: {total_time:.3f}s for {len(features_list)} segments")
    return features_list, labels_list

# =============================================================================
# PyTorch Dataset
# =============================================================================

class AudioDataset(Dataset):
    def __init__(self, features_list, labels_list):
        self.features_list = features_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        feat = self.features_list[idx]
        # For the spectrograms, add a channel dimension (assumes shape: [freq, time])
        spec_left = torch.tensor(feat["gammatone_left"], dtype=torch.float32).unsqueeze(0)
        spec_right = torch.tensor(feat["gammatone_right"], dtype=torch.float32).unsqueeze(0)
        # For GCC-PHAT, treat it as a 1D “image” with one channel.
        gcc = torch.tensor(feat["gcc"], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels_list[idx], dtype=torch.float32)
        return {"spec_left": spec_left, "spec_right": spec_right, "gcc": gcc, "label": label}

# =============================================================================
# PyTorch CNN Regression Model
# =============================================================================

class AudioRegressionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec_conv = nn.Sequential(
            # Block 1
            nn.Conv2d(3,  64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128,256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # GCC branch
        self.gcc_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2,2),
            nn.Conv1d(64,128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2,2),
        )

        # Fusion
        fusion_dim = 256 + 128 
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, spec_left, spec_right, gcc):
  
        diff = spec_left - spec_right
        x = torch.cat([spec_left, spec_right, diff], dim=1)  # (B,3,F,T)
        h = self.spec_conv(x)                               # (B,256, F/8, T/8)
        f_spec = h.mean(dim=[2,3])                          # (B,256)

        g = self.gcc_conv(gcc)                              # (B,128, L/4)
        f_gcc = g.mean(dim=2)                               # (B,128)

        out = torch.cat([f_spec, f_gcc], dim=1)              # (B,384)
        return self.fc(out)                                 # (B,2)


# =============================================================================
# Model Training
# =============================================================================
    
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-3, weight_decay=1e-5):
    """
    Train the sound localization model with improved techniques.
    Implements a reduce-on-plateau learning rate scheduler that decreases the LR when the validation loss stops improving.
    Allows stopping during runtime (e.g., via KeyboardInterrupt) while using the best model state.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Use ReduceLROnPlateau scheduler to reduce LR when validation loss plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    
    def angular_error(pred, target):
        pred_angle = torch.atan2(pred[:, 0], pred[:, 1]) * 180 / math.pi
        target_angle = torch.atan2(target[:, 0], target[:, 1]) * 180 / math.pi
        diff = (pred_angle - target_angle + 180) % 360 - 180
        return torch.abs(diff)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    best_model_state = None
    
    print("Training started with improved pipeline (reduce on plateau)")
    try:
        for epoch in range(num_epochs):
            epoch_start = time.time()
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                spec_left = batch['spec_left']
                spec_right = batch['spec_right']
                gcc = batch['gcc']
                labels_batch = batch['label']
                
                optimizer.zero_grad()
                outputs = model(spec_left, spec_right, gcc)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            model.eval()
            val_loss = 0.0
            angular_errors = []
            with torch.no_grad():
                for batch in val_loader:
                    spec_left = batch['spec_left']
                    spec_right = batch['spec_right']
                    gcc = batch['gcc']
                    labels_batch = batch['label']
                    
                    outputs = model(spec_left, spec_right, gcc)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()
                    batch_errors = angular_error(outputs, labels_batch)
                    angular_errors.append(batch_errors)
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            all_errors = torch.cat(angular_errors)
            mae = torch.mean(all_errors).item()
            history['val_mae'].append(mae)
            
            # Step the ReduceLROnPlateau scheduler using the validation loss.
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            epoch_duration = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val MAE (degrees): {mae:.2f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Epoch Duration: {epoch_duration:.2f}s")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user! Loading best model state...")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Best model state loaded based on validation loss.")
    
    return model, history
 
    
# =============================================================================
# Main Script: Data Loading, Preprocessing, and Model Training
# =============================================================================

if __name__ == '__main__':
    overall_start = time.time()
    
    # Import missing libraries
    import librosa
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    
    # Parameters for spectrogram and GCC computations
    window_time = 0.1   # 100 ms window
    hop_time = window_time * 0.5    # 50% overlap
    fmin = 50             # Minimum frequency for gammatone filter 
    fmax = 8000           # Maximum frequency 
    
    # List of folder paths where the 2-channel WAV files are stored
    folder_paths = [r"path"]

    
    all_kept_segments = []
    all_kept_labels = []
    all_kept_sample_rates = []
    
    # Process each folder and combine the results
    for folder_path in folder_paths:
        print(f"Processing folder: {folder_path}")
        kept_segments, kept_labels, kept_sample_rates, kept_energies, _, _, _, _ = process_wav_files(folder_path)
        print(f"  Extracted {len(kept_segments)} segments from {folder_path}")
        
        # Apply data augmentation to each segment
        augmented_segments = []
        augmented_labels = []
        augmented_sample_rates = []
        
        for segment, angle, sr in zip(kept_segments, kept_labels, kept_sample_rates):
            # Apply augmentation
            augmented_data = augment_audio_data(segment, sr, angle)
            
            # Add to lists
            for aug_segment, aug_angle in augmented_data:
                augmented_segments.append(aug_segment)
                augmented_labels.append(aug_angle)
                augmented_sample_rates.append(sr)
        
        print(f"  After augmentation: {len(augmented_segments)} segments")
        all_kept_segments.extend(augmented_segments)
        all_kept_labels.extend(augmented_labels)
        all_kept_sample_rates.extend(augmented_sample_rates)
    
    total_segments = len(all_kept_segments)
    print(f"Total segments after augmentation: {total_segments}")
    
    if total_segments == 0:
        raise ValueError("No valid segments were extracted. Check the folder paths and WAV file format.")
    
    # Precompute features
    features, labels = preprocess_audio_segments(all_kept_segments, all_kept_sample_rates, all_kept_labels,
                                               window_time, hop_time, fmin, fmax)
    print(f"Precomputed features and converted angle labels for {len(features)} segments.")
    
    # -------------------- Combined Normalization for Spectrograms --------------------
    left_specs = np.array([f["gammatone_left"] for f in features])
    right_specs = np.array([f["gammatone_right"] for f in features])
    gcc_features = np.array([f["gcc"] for f in features])
    
    # Merge left and right spectrograms to compute a single normalization statistic.
    merged_specs = np.concatenate((left_specs.flatten(), right_specs.flatten()))
    merged_mean, merged_std = np.mean(merged_specs), np.std(merged_specs)
    
    # Normalize both left and right using the same merged mean/std.
    for f in features:
        f["gammatone_left"] = (f["gammatone_left"] - merged_mean) / (merged_std + 1e-8)
        f["gammatone_right"] = (f["gammatone_right"] - merged_mean) / (merged_std + 1e-8)
        f["gcc"] = (f["gcc"] - np.mean(gcc_features)) / (np.std(gcc_features) + 1e-8)
    
    print("Features normalized.")
    
    # Create a PyTorch Dataset
    dataset = AudioDataset(features, labels)
    
    # Split dataset into train, validation, and test
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Instantiate the improved CNN model
    model = AudioRegressionCNN()
    
    # Print model summary
    print("Model architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train the model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-5
    )
    
    torch.save({
            'model_state_dict': model.state_dict(),
            'feature_normalization': {
                'merged_mean': merged_mean,
                'merged_std': merged_std,
                'gcc_mean': np.mean(gcc_features),
                'gcc_std': np.std(gcc_features)
            },
            'training_history': history
        }, 'sound_localisation_model_CCN.pth')
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    all_angles = []
    
    with torch.no_grad():
        for batch in test_loader:
            spec_left = batch['spec_left'].to(device)
            spec_right = batch['spec_right'].to(device)
            gcc = batch['gcc'].to(device)
            labels_batch = batch['label'].to(device)
            
            outputs = model(spec_left, spec_right, gcc)
            loss = nn.MSELoss()(outputs, labels_batch)
            test_loss += loss.item()
            
            # Convert to angles
            preds_deg = torch.atan2(outputs[:, 0], outputs[:, 1]) * 180 / math.pi
            gt_deg = torch.atan2(labels_batch[:, 0], labels_batch[:, 1]) * 180 / math.pi
            
            # Calculate the circular difference
            diff = (preds_deg - gt_deg + 180) % 360 - 180
            
            all_preds.append(preds_deg.cpu())
            all_labels.append(gt_deg.cpu())
            all_angles.append(torch.abs(diff).cpu())
    
    # Calculate final metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_angles = torch.cat(all_angles)
    
    avg_test_loss = test_loss / len(test_loader)
    mae = torch.mean(all_angles).item()
    rmse = torch.sqrt(torch.mean(all_angles**2)).item()
    
    # Calculate accuracy within different thresholds
    acc_5deg = (all_angles <= 5).float().mean().item() * 100
    acc_10deg = (all_angles <= 10).float().mean().item() * 100
    acc_15deg = (all_angles <= 15).float().mean().item() * 100
    
    print("\nTest Results:")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Mean Absolute Error: {mae:.2f}°")
    print(f"Root Mean Square Error: {rmse:.2f}°")
    print(f"Accuracy within 5°: {acc_5deg:.2f}%")
    print(f"Accuracy within 10°: {acc_10deg:.2f}%")
    print(f"Accuracy within 15°: {acc_15deg:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['val_mae'], color='g')
    plt.title('Validation MAE (degrees)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    
    plt.subplot(2, 2, 3)
    plt.plot(history['learning_rates'], color='r')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    # Plot confusion matrix of predicted vs actual angles
    plt.subplot(2, 2, 4)
    plt.scatter(all_labels.numpy(), all_preds.numpy(), alpha=0.3)
    plt.plot([-180, 180], [-180, 180], 'r--')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.title('Predicted vs Actual Angles')
    plt.xlabel('Actual Angle (degrees)')
    plt.ylabel('Predicted Angle (degrees)')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print(f"Total execution time: {(time.time() - overall_start)/60:.2f} minutes")
    print("Training complete! Model saved as 'sound_localization_model.pth'")
    
            # ===================== Unseen Data Testing Section =====================
    # Specify the folder path containing unseen .wav files.
    unseen_folder = r"path"
    
    print("\nStarting testing on unseen data...")
    # Process unseen WAV files using the same function as for training data.
    unseen_segments, unseen_labels, unseen_sample_rates, _, _, _, _, _ = process_wav_files(unseen_folder,0.5,0)
    
    if len(unseen_segments) == 0:
        raise ValueError("No valid segments found in the unseen data folder!")
    
    print(f"Number of segments extracted from unseen data: {len(unseen_segments)}")
    
    # Optionally, you can choose to augment the unseen data if desired.
    # Here we simply use the original segments.
    unseen_features, unseen_label_vectors = preprocess_audio_segments(
        unseen_segments, unseen_sample_rates, unseen_labels,
        window_time, hop_time, fmin, fmax
    )
    
    # -------------------- Apply the same normalization --------------------
    # Use the normalization parameters computed during training.
    norm_params = {
        'merged_mean': merged_mean,
        'merged_std': merged_std,
        'gcc_mean': np.mean(gcc_features),
        'gcc_std': np.std(gcc_features)
    }
    
    for f in unseen_features:
        f["gammatone_left"] = (f["gammatone_left"] - norm_params['merged_mean']) / (norm_params['merged_std'] + 1e-8)
        f["gammatone_right"] = (f["gammatone_right"] - norm_params['merged_mean']) / (norm_params['merged_std'] + 1e-8)
        f["gcc"] = (f["gcc"] - norm_params['gcc_mean']) / (norm_params['gcc_std'] + 1e-8)
    
    # Create a dataset and DataLoader for unseen data.
    unseen_dataset = AudioDataset(unseen_features, unseen_label_vectors)
    unseen_loader = DataLoader(unseen_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Ensure the model is in evaluation mode and on the proper device.
    model.eval()
    model.to(device)
    
    all_unseen_preds = []
    all_unseen_gt = []  # Only if ground truth labels are available
    with torch.no_grad():
        for batch in unseen_loader:
            spec_left = batch['spec_left'].to(device)
            spec_right = batch['spec_right'].to(device)
            gcc = batch['gcc'].to(device)
            outputs = model(spec_left, spec_right, gcc)
            # Convert network outputs from sine/cosine representation to angles (in degrees).
            preds_deg = torch.atan2(outputs[:, 0], outputs[:, 1]) * 180 / math.pi
            all_unseen_preds.append(preds_deg.cpu())
            
            # If ground truth is provided in unseen_labels, convert it as well.
            if 'label' in batch:
                gt_deg = torch.atan2(batch['label'][:, 0], batch['label'][:, 1]) * 180 / math.pi
                all_unseen_gt.append(gt_deg.cpu())
    
    all_unseen_preds = torch.cat(all_unseen_preds)
    print("\n--- Unseen Data Prediction Results ---")
    print(f"Total segments processed: {len(unseen_dataset)}")
    print("Predicted angles (degrees):")
    print(all_unseen_preds.numpy())
    
    # If ground truth is available, print additional statistics.
    if all_unseen_gt:
        all_unseen_gt = torch.cat(all_unseen_gt)
        print("Ground truth angles (degrees):")
        print(all_unseen_gt.numpy())
        # Calculate and print summary statistics.
        differences = (all_unseen_preds - all_unseen_gt + 180) % 360 - 180
        mae_unseen = torch.mean(torch.abs(differences)).item()
        rmse_unseen = torch.sqrt(torch.mean(differences**2)).item()
        print(f"Mean Absolute Error on unseen data: {mae_unseen:.2f}°")
        print(f"Root Mean Square Error on unseen data: {rmse_unseen:.2f}°")
    else:
        # If no ground truth is provided, print summary statistics of predictions.
        print("Summary of predicted angles:")
        print(f"Min: {all_unseen_preds.min().item():.2f}°, Max: {all_unseen_preds.max().item():.2f}°, Mean: {all_unseen_preds.mean().item():.2f}°")


    
    plt.figure(figsize=(6, 6))
    plt.scatter(all_unseen_gt.numpy(), all_unseen_preds.numpy(), alpha=0.3)
    plt.plot([-180, 180], [-180, 180], 'r--')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.title('Predicted vs Actual Angles')
    plt.xlabel('Actual Angle (degrees)')
    plt.ylabel('Predicted Angle (degrees)')
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()
    
    acc_25deg = (torch.abs(differences) <= 25).float().mean().item() * 100
    print(f"Accuracy within 25°: {acc_25deg:.2f}%")