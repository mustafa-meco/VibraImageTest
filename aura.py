import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load the face detection model (Haar Cascade) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 2. Image Acquisition ---
camera = cv2.VideoCapture(0)  # Assuming default camera

if not camera.isOpened():
    print("Camera Error!")
    exit()

ret, frame = camera.read()
if ret:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Grayscale
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Assume the first detected face
        x, y, w, h = faces[0]

        # --- 3. Create Aura Effect ---
        # Create an aura mask
        aura_mask = np.zeros_like(frame, dtype=np.uint8)

        # Define the thickness of the aura
        aura_thickness = 20

        # Draw the aura (with increasing thickness)
        for i in range(1, aura_thickness + 1):
            color_intensity = 255 - (255 // aura_thickness) * i
            cv2.rectangle(aura_mask,
                          (x - i, y - i),
                          (x + w + i, y + h + i),
                          (color_intensity, color_intensity, 255),
                          thickness=1)

        # Blend the aura mask with the original frame
        aura_effect = cv2.addWeighted(frame, 1, aura_mask, 0.5, 0)

        # Draw the face bounding box for reference
        cv2.rectangle(aura_effect, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result
        plt.imshow(cv2.cvtColor(aura_effect, cv2.COLOR_BGR2RGB))
        plt.title("Aura Effect Around Face")
        plt.axis('off')
        plt.show()

    else:
        print("No faces detected!")

camera.release()
cv2.destroyAllWindows()
