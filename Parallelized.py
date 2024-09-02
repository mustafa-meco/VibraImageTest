import cv2
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# --- 1. Image Acquisition ---
camera = cv2.VideoCapture(0)  # Assuming default camera

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

# Convert frames to a 3D numpy array: (time, height, width)
frames = np.array(frames, dtype=np.float32)

# --- 3. Vibraimage Generation ---
# Amplitude Vibraimage (Eq. 1 from source)
amplitude_vibraimage = np.zeros_like(frames[0], dtype=np.float32)
for i in range(1, len(frames)):
    amplitude_vibraimage += np.abs(frames[i] - frames[i - 1])
amplitude_vibraimage /= (len(frames) - 1)

# Function to calculate frequency for a block of rows
def calculate_frequency_block(row_range):
    result_block = np.zeros_like(frames[0][row_range], dtype=np.float32)
    for y in range(result_block.shape[0]):
        for x in range(result_block.shape[1]):
            pixel_time_series = frames[:, row_range.start + y, x]
            fft_result = np.fft.fft(pixel_time_series)
            frequencies = np.fft.fftfreq(len(pixel_time_series))
            peak_frequency = np.abs(frequencies[np.argmax(np.abs(fft_result[1:])) + 1])
            result_block[y, x] = peak_frequency
    return row_range, result_block

# Parallel computation
num_cores = cpu_count()  # Number of CPU cores
height = frames.shape[1]
chunk_size = height // num_cores

# Define the row ranges for each chunk
row_ranges = [slice(i, i + chunk_size) for i in range(0, height, chunk_size)]

# Use multiprocessing Pool to process each chunk in parallel
with Pool(processes=num_cores) as pool:
    results = pool.map(calculate_frequency_block, row_ranges)

# Combine results into a full frequency_vibraimage
frequency_vibraimage = np.zeros_like(frames[0], dtype=np.float32)
for row_range, result_block in results:
    frequency_vibraimage[row_range] = result_block

# --- 4. Feature Extraction & Analysis ---
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
plt.title("Frequency Vibraimage")

plt.subplot(2, 3, 5)
plt.plot(amplitude_hist)
plt.title("Amplitude Histogram")

plt.subplot(2, 3, 6)
plt.plot(frequency_hist)
plt.title("Frequency Histogram")

plt.tight_layout()
plt.show()