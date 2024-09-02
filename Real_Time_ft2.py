import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading

# Parameters
num_frames = 20  # Reduce the number of frames in the buffer
buffer = deque(maxlen=num_frames)  # Circular buffer to hold the frames
frame_size = (640, 480)  # Reduce frame size for faster processing

# Precompute FFT frequencies
frequencies = np.fft.fftfreq(num_frames)

# Function to process frames in a separate thread
def process_frame():
    while True:
        if len(buffer) == num_frames:
            frames = np.array(buffer, dtype=np.float32)

            # --- Vibraimage Generation ---
            # Amplitude Vibraimage (Eq. 1 from source)
            amplitude_vibraimage = np.mean(np.abs(np.diff(frames, axis=0)), axis=0)

            # Frequency Vibraimage using vectorized Fourier Transform
            fft_result = np.fft.fft(frames, axis=0)
            peak_frequency = np.abs(frequencies[np.argmax(np.abs(fft_result[1:, :, :]), axis=0) + 1])

            # --- Visualization ---
            axs[0, 0].imshow(frames[0], cmap='gray')
            axs[0, 0].set_title("Frame Example")
            axs[0, 1].imshow(amplitude_vibraimage, cmap='jet')
            axs[0, 1].set_title("Amplitude Vibraimage")
            axs[0, 2].imshow(peak_frequency, cmap='viridis')
            axs[0, 2].set_title("Frequency Vibraimage")

            plt.pause(0.001)  # Pause to allow for plot updates

# --- Start Video Capture ---
camera = cv2.VideoCapture(0)  # Assuming default camera

if not camera.isOpened():
    print("Camera Error!")
    exit()

plt.ion()  # Turn on interactive mode for real-time plotting
fig, axs = plt.subplots(2, 3, figsize=(12, 5))

# Start the frame processing thread
thread = threading.Thread(target=process_frame)
thread.start()

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, frame_size)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    buffer.append(gray_frame)

    # Display the live feed in a separate window
    cv2.imshow("Live Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
plt.ioff()  # Turn off interactive mode
plt.show()

# Join the thread after completion
thread.join()
