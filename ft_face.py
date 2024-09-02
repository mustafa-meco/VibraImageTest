import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load the face detection model (Haar Cascade) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 2. Image Acquisition ---
# camera = cv2.VideoCapture(0)  # Assuming default camera
camera = cv2.VideoCapture("Videos/00.mp4")


if not camera.isOpened():
    print("Camera Error!")
    exit()

frames = []
for _ in range(30):  # Capture 30 frames (example)
    ret, frame = camera.read()
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Grayscale
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Assume the first detected face
            x, y, w, h = faces[0]
            face_roi = gray_frame[y:y+h, x:x+w]  # Extract the face region
            frames.append(face_roi)
        cv2.waitKey(30)  # Adjust delay as needed for frame rate

camera.release()
cv2.destroyAllWindows()

if len(frames) == 0:
    print("No faces detected!")
    exit()

# Ensure all face regions have the same shape
min_height = min(frame.shape[0] for frame in frames)
min_width = min(frame.shape[1] for frame in frames)
frames = [cv2.resize(frame, (min_width, min_height)) for frame in frames]

# Convert frames to a 3D numpy array: (time, height, width)
frames = np.array(frames, dtype=np.float32)

# --- 3. Vibraimage Generation ---
# Amplitude Vibraimage
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

# --- 4. Feature Extraction & Analysis ---
# Example: Histograms
amplitude_hist, _ = np.histogram(amplitude_vibraimage.flatten(), bins=20)
frequency_hist, _ = np.histogram(frequency_vibraimage.flatten(), bins=20)

# --- 5. Visualization ---
plt.figure(figsize=(12, 5))

plt.subplot(2, 3, 1)
plt.imshow(frames[0], cmap='gray')  # Display the first face ROI as an example
plt.title("Face ROI Example")

plt.subplot(2, 3, 2)
plt.imshow(amplitude_vibraimage, cmap='jet')
plt.title("Amplitude Vibraimage")

plt.subplot(2, 3, 3)
plt.imshow(frequency_vibraimage, cmap='viridis')
plt.title("Frequency Vibraimage")

plt.subplot(2, 3, 5)
plt.plot(amplitude_hist)
plt.title("Amplitude Histogram")

plt.subplot(2, 3, 6)
plt.plot(frequency_hist)
plt.title("Frequency Histogram")

plt.tight_layout()
plt.show()