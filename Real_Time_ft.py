import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Parameters
num_frames = 30  # Number of frames to keep in the buffer for analysis
buffer = deque(maxlen=num_frames)  # Circular buffer to hold the frames

# --- 1. Start Video Capture ---
camera = cv2.VideoCapture(0)  # Assuming default camera

if not camera.isOpened():
    print("Camera Error!")
    exit()

plt.ion()  # Turn on interactive mode for real-time plotting
fig, axs = plt.subplots(2, 3, figsize=(12, 5))

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    buffer.append(gray_frame)

    if len(buffer) == num_frames:
        frames = np.array(buffer, dtype=np.float32)

        # --- Vibraimage Generation ---
        # Amplitude Vibraimage (Eq. 1 from source)
        amplitude_vibraimage = np.zeros_like(frames[0], dtype=np.float32)
        for i in range(1, len(frames)):
            amplitude_vibraimage += np.abs(frames[i] - frames[i - 1])
        amplitude_vibraimage /= (len(frames) - 1)

        # Frequency Vibraimage using Fourier Transform
        frequency_vibraimage = np.zeros_like(frames[0], dtype=np.float32)
        for y in range(frames.shape[1]):  # height
            for x in range(frames.shape[2]):  # width
                pixel_time_series = frames[:, y, x]
                fft_result = np.fft.fft(pixel_time_series)
                frequencies = np.fft.fftfreq(len(pixel_time_series))
                peak_frequency = np.abs(frequencies[np.argmax(np.abs(fft_result[1:])) + 1])
                frequency_vibraimage[y, x] = peak_frequency

        # --- Feature Extraction & Analysis ---
        amplitude_hist, _ = np.histogram(amplitude_vibraimage.flatten(), bins=20)
        frequency_hist, _ = np.histogram(frequency_vibraimage.flatten(), bins=20)

        # --- Visualization ---
        axs[0, 0].imshow(frames[0], cmap='gray')
        axs[0, 0].set_title("Frame Example")
        axs[0, 1].imshow(amplitude_vibraimage, cmap='jet')
        axs[0, 1].set_title("Amplitude Vibraimage")
        axs[0, 2].imshow(frequency_vibraimage, cmap='viridis')
        axs[0, 2].set_title("Frequency Vibraimage")

        axs[1, 1].cla()
        axs[1, 1].plot(amplitude_hist)
        axs[1, 1].set_title("Amplitude Histogram")

        axs[1, 2].cla()
        axs[1, 2].plot(frequency_hist)
        axs[1, 2].set_title("Frequency Histogram")

        plt.pause(0.001)  # Pause to allow for plot updates

    # Display the live feed in a separate window
    cv2.imshow("Live Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
plt.ioff()  # Turn off interactive mode
plt.show()

