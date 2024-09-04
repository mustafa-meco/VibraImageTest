from typing import Final

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def capture_frames(video_path, num_frames=30):
    """
    Captures frames from a video or camera.

    Args:
        video_path (str): Path to the video file or 0 for webcam.
        num_frames (int, optional): Number of frames to capture. Defaults to 30.

    Returns:
        list: List of grayscale frames.
    """
    camera = cv2.VideoCapture(video_path)
    if not camera.isOpened():
        print("Error opening video/camera!")
        return []

    frames = []
    for _ in range(num_frames):
        ret, frame = camera.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            break

    camera.release()
    cv2.destroyAllWindows()
    return frames

def capture_frames_Fin(video_path, num_frames=30):
    """
    Captures frames from a video or camera.

    Args:
        video_path (str): Path to the video file or 0 for webcam.
        num_frames (int, optional): Number of frames to capture. Defaults to 30.

    Returns:
        list: List of grayscale frames.
    """
    camera = cv2.VideoCapture(video_path)
    Fin = camera.get(cv2.CAP_PROP_FPS)
    if not camera.isOpened():
        print("Error opening video/camera!")
        return []

    frames = []
    for _ in range(num_frames):
        ret, frame = camera.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            break

    camera.release()
    cv2.destroyAllWindows()
    return frames, Fin

def extract_features(amplitude_vibraimage, frequency_vibraimage):
    """
    Extracts features (histograms) from vibraimages.

    Args:
        amplitude_vibraimage (numpy.ndarray): Amplitude vibraimage.
        frequency_vibraimage (numpy.ndarray): Frequency vibraimage.

    Returns:
        tuple: Tuple containing amplitude and frequency histograms (numpy arrays).
    """
    amplitude_hist, _ = np.histogram(amplitude_vibraimage.flatten(), bins=20)
    frequency_hist, _ = np.histogram(frequency_vibraimage.flatten(), bins=20)
    return amplitude_hist, frequency_hist


def visualize_results(frames, amplitude_vibraimage, frequency_vibraimage, amplitude_hist, frequency_hist):
    """
    Visualizes frames, vibraimages, and histograms.

    Args:
        frames (list): List of grayscale frames.
        amplitude_vibraimage (numpy.ndarray): Amplitude vibraimage.
        frequency_vibraimage (numpy.ndarray): Frequency vibraimage.
        amplitude_hist (numpy.ndarray): Amplitude histogram.
        frequency_hist (numpy.ndarray): Frequency histogram.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(2, 3, 1)
    plt.imshow(frames[0], cmap='gray')
    plt.title("Frame Example")

    plt.subplot(2, 3, 2)
    plt.imshow(amplitude_vibraimage, cmap='jet')
    plt.title("Amplitude Vibraimage")

    plt.subplot(2, 3, 3)
    plt.imshow(frequency_vibraimage, cmap='viridis')
    plt.title("Frequency Approximation")

    plt.subplot(2, 3, 5)
    plt.plot(amplitude_hist)
    plt.title("Amplitude Histogram")

    plt.subplot(2, 3, 6)
    plt.plot(frequency_hist)
    plt.title("Frequency Histogram")

    plt.tight_layout()
    plt.show()

def save_results(amplitude_vibraimage, frequency_vibraimage, amplitude_hist, frequency_hist, output_prefix="output"):
    """
    Saves vibraimages and histograms to files.

    Args:
        amplitude_vibraimage (numpy.ndarray): Amplitude vibraimage.
        frequency_vibraimage (numpy.ndarray): Frequency vibraimage.
        amplitude_hist (numpy.ndarray): Amplitude histogram.
        frequency_hist (numpy.ndarray): Frequency histogram.
        output_prefix (str, optional): Prefix for output files. Defaults to "output".
    """

    os.makedirs(f"saved_results/{output_prefix}", exist_ok=True)
    # if folder exist print warning
    if os.path.exists(f"saved_results/{output_prefix}"):
        print("Warning: Folder already exists. Files will be overwritten.")

    folder = f"saved_results/{output_prefix}"

    # Save vibraimages
    cv2.imwrite(f"{folder}/{output_prefix}_amplitude.png", amplitude_vibraimage)
    cv2.imwrite(f"{folder}/{output_prefix}_frequency.png", frequency_vibraimage)

    # Save histograms
    np.save(f"{folder}/{output_prefix}_amplitude_hist.npy", amplitude_hist)
    np.save(f"{folder}/{output_prefix}_frequency_hist.npy", frequency_hist)

    print(f"Results saved with prefix '{output_prefix}'.")

def calculate_amplitude(frames):
    N = len(frames)  # Number of frames
    height, width = frames[0].shape  # Dimensions of each frame

    # Initialize the amplitude array with zeros
    amplitude = np.zeros((height, width))

    # Calculate the amplitude component for each point (x, y)
    for x in range(height):
        for y in range(width):
            # Sum the absolute differences between consecutive frames
            diff_sum = 0
            for i in range(N - 1):
                diff_sum += abs(frames[i][x, y] - frames[i + 1][x, y])

            # Average the differences
            amplitude[x, y] = diff_sum / (N - 1)

    return amplitude

# Frequency Vibraimage
def calculate_frequency(frames, Fin):
    N = len(frames)  # Number of frames
    height, width = frames[0].shape  # Dimensions of each frame

    # Initialize the frequency array with zeros
    frequency = np.zeros((height, width))

    # Calculate the frequency component for each point (x, y)
    for x in range(height):
        for y in range(width):
            # Sum the indicator values for each consecutive frame difference
            indicator_sum = 0
            for i in range(N - 1):
                if abs(frames[i][x, y] - frames[i + 1][x, y]) > 0:
                    indicator_sum += 1

            # Calculate the frequency component
            frequency[x, y] = Fin * indicator_sum / (N - 1)

    return frequency


def calculate_fft_frequency_amplitude(frames, Fin):
    N = len(frames)  # Number of frames
    height, width = frames[0].shape  # Dimensions of each frame

    # Initialize the frequency and amplitude arrays with zeros
    frequency = np.zeros((height, width))
    amplitude = np.zeros((height, width))

    # Calculate the frequency and amplitude components for each point (x, y)
    for x in range(height):
        for y in range(width):
            # Extract the time series signal for point (x, y)
            signal = np.array([frames[i][x, y] for i in range(N)])

            # Apply FFT to the signal
            fft_result = np.fft.fft(signal)

            # Compute the magnitude (amplitude) of the FFT result
            fft_magnitude = np.abs(fft_result)

            # Sum the magnitudes (excluding the DC component) for the total amplitude
            amplitude[x, y] = np.sum(fft_magnitude[1:N // 2])

            # Find the dominant frequency (ignore the DC component at index 0)
            dominant_frequency_index = np.argmax(fft_magnitude[1:N // 2]) + 1

            # Calculate the corresponding frequency in Hz
            frequency[x, y] = (dominant_frequency_index * Fin) / N

    return amplitude, frequency