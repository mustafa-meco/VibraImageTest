import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading

# Parameters
num_frames = 20  # Number of frames to keep in the buffer for analysis
buffer = deque(maxlen=num_frames)  # Circular buffer to hold the frames
frame_size = (640, 480)  # Reduce frame size for faster processing

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Precompute FFT frequencies
frequencies = np.fft.fftfreq(num_frames)

# Function to process frames in a separate thread
def process_frame():
    while True:
        if len(buffer) == num_frames:
            frames = np.array(buffer, dtype=np.float32)
            gray_frame = frames[0]

            # Detect face in the current frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                (x, y, w, h) = faces[0]  # Assume the first detected face
                face_roi = frames[:, y:y+h, x:x+w]

                # --- Vibraimage Generation within the face bounding box ---
                amplitude_vibraimage = np.mean(np.abs(np.diff(face_roi, axis=0)), axis=0)

                # Frequency Vibraimage using vectorized Fourier Transform
                fft_result = np.fft.fft(face_roi, axis=0)
                peak_frequency = np.abs(frequencies[np.argmax(np.abs(fft_result[1:, :, :]), axis=0) + 1])

                # --- Visualization ---
                axs[0, 0].imshow(gray_frame, cmap='gray')
                axs[0, 0].set_title("Detected Face")
                axs[0, 0].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))

                axs[0, 1].imshow(amplitude_vibraimage, cmap='jet')
                axs[0, 1].set_title("Amplitude Vibraimage (Face)")

                axs[0, 2].imshow(peak_frequency, cmap='viridis')
                axs[0, 2].set_title("Frequency Vibraimage (Face)")

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
