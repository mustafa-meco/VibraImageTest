from vibra_utils import *

def preprocess_frames(frames):
  """
  Preprocesses frames by converting them to a 3D numpy array.

  Args:
      frames (list): List of grayscale frames.

  Returns:
      numpy.ndarray: 3D numpy array representing frames (time, height, width).
  """
  frames = np.array(frames, dtype=np.float32)
  return frames


def generate_amplitude_vibraimage(frames):
  """
  Generates the amplitude vibraimage using equation 1.

  Args:
      frames (numpy.ndarray): 3D numpy array representing frames.

  Returns:
      numpy.ndarray: Amplitude vibraimage.
  """
  amplitude_vibraimage = np.zeros_like(frames[0], dtype=np.float32)
  for i in range(1, len(frames)):
    amplitude_vibraimage += np.abs(frames[i] - frames[i - 1])
  amplitude_vibraimage /= (len(frames) - 1)
  return amplitude_vibraimage


def generate_frequency_vibraimage(frames):
  """
  Generates the frequency vibraimage using Fourier Transform.

  Args:
      frames (numpy.ndarray): 3D numpy array representing frames.

  Returns:
      numpy.ndarray: Frequency vibraimage.
  """
  frequency_vibraimage = np.zeros_like(frames[0], dtype=np.float32)
  for y in range(frames.shape[1]):
    for x in range(frames.shape[2]):
      pixel_time_series = frames[:, y, x]
      fft_result = np.fft.fft(pixel_time_series)
      frequencies = np.fft.fftfreq(len(pixel_time_series))
      peak_frequency = np.abs(frequencies[np.argmax(np.abs(fft_result[1:])) + 1])
      frequency_vibraimage[y, x] = peak_frequency
  return frequency_vibraimage


def main():
  video_path = "Videos/00.mp4"  # Replace with your video path or 0 for webcam
  frames = capture_frames(video_path, num_frames=30)
  if frames:
    frames = preprocess_frames(frames)
    amplitude_vibraimage = generate_amplitude_vibraimage(frames)
    frequency_vibraimage = generate_frequency_vibraimage(frames)
    amplitude_hist, frequency_hist = extract_features(amplitude_vibraimage, frequency_vibraimage)
    visualize_results(frames, amplitude_vibraimage, frequency_vibraimage, amplitude_hist, frequency_hist)
  else:
    print("No frames captured!")

if __name__ == "__main__":
    main()
