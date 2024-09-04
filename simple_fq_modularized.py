from vibra_utils import *

def generate_vibraimages(frames):
    """
    Generates amplitude and frequency vibraimages from a list of frames.

    Args:
        frames (list): List of grayscale frames.

    Returns:
        tuple: Tuple containing amplitude and frequency vibraimages (numpy arrays).
    """
    amplitude_vibraimage = calculate_amplitude(frames)
    frequency_vibraimage = calculate_frequency()

    return amplitude_vibraimage, frequency_vibraimage


def main():
    video_path = "Videos/00.mp4"  # Replace with your video path or 0 for webcam
    frames = capture_frames(video_path, num_frames=30)
    if frames:
        amplitude_vibraimage, frequency_vibraimage = generate_vibraimages(frames)
        amplitude_hist, frequency_hist = extract_features(amplitude_vibraimage, frequency_vibraimage)
        visualize_results(frames, amplitude_vibraimage, frequency_vibraimage, amplitude_hist, frequency_hist)
    else:
        print("No frames captured!")

if __name__ == "__main__":
    main()



