import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Image Acquisition ---
# camera = cv2.VideoCapture(0)  # Assuming default camera
camera = cv2.VideoCapture("Videos/00.mp4")


if not camera.isOpened():
    print("Camera Error!")
    exit()

frames = []
for _ in range(30):  # Capture 30 frames (example)
    ret, frame = camera.read()
    if ret:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # Grayscale
        cv2.waitKey(30)  # Adjust delay as needed for frame rate

camera.release()
cv2.destroyAllWindows()

# --- 2. Pre-processing (Simplified) ---
# In a full implementation, you'd likely have cropping,
# illumination correction, etc., here.

# --- 3. Vibraimage Generation ---
# Amplitude Vibraimage (Eq. 1 from paper)
amplitude_vibraimage = np.zeros_like(frames[0], dtype=np.float32)
for i in range(1, len(frames)):
    amplitude_vibraimage += np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))
amplitude_vibraimage /= (len(frames) - 1)

# Frequency Vibraimage (Simplified approach)
frequency_vibraimage = np.abs(frames[-1].astype(np.float32) - frames[0].astype(np.float32))

# --- 4. Feature Extraction & Analysis ---
# Example: Histograms
amplitude_hist, _ = np.histogram(amplitude_vibraimage.flatten(), bins=20)
frequency_hist, _ = np.histogram(frequency_vibraimage.flatten(), bins=20)

# --- 5. Visualization ---
plt.figure(figsize=(12, 5))

plt.subplot(2, 3, 1)
plt.imshow(frames[0], cmap='gray')  # Display the first frame as an example
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
