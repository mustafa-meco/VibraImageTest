import cv2
import matplotlib.pyplot as plt
import numpy as np

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