import cv2
from vibra_utils import *
from ft_main_modular import generate_amplitude_vibraimage, generate_frequency_vibraimage, preprocess_frames


def load_face_detector():
  """
  Loads the Haar Cascade face detection model.

  Returns:
      cv2.CascadeClassifier: Loaded face cascade classifier.
  """
  return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(gray_frame, face_cascade):
  """
  Detects faces in a given frame using the cascade classifier.

  Args:
      gray_frame (numpy.ndarray): The gray frame to detect faces in.
      face_cascade (cv2.CascadeClassifier): Loaded face cascade classifier.

  Returns:
      list: List of face regions (bounding boxes) as (x, y, w, h) tuples or empty list if no faces found.
  """
  return face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


def extract_face_rois(frames, faces):
  """
  Extracts face regions of interest from a list of frames and their corresponding face detections.

  Args:
      frames (list): List of frames (numpy.ndarray).
      faces (list): List of face detections as (x, y, w, h) tuples for each frame.

  Returns:
      list: List of extracted face regions (numpy.ndarray).
  """
  face_rois = []
  for frame in frames:
    if  len(faces) > 0:
      x, y, w, h = faces[0]  # Assuming the first face detection
      face_roi = frame[y:y+h, x:x+w]
      face_rois.append(face_roi)
  return face_rois


def preprocess_faces(face_rois):
  """
  Preprocesses the list of face regions by ensuring all have the same size and converting to a 3D numpy array.

  Args:
      face_rois (list): List of face regions (numpy.ndarray).

  Returns:
      numpy.ndarray: 3D numpy array representing face regions (time, height, width).
  """
  if not face_rois:
    return None

  min_height = min(frame.shape[0] for frame in face_rois)
  min_width = min(frame.shape[1] for frame in face_rois)
  preprocessed_faces = [cv2.resize(frame, (min_width, min_height)) for frame in face_rois]
  return preprocess_frames(preprocessed_faces)


def main():
  video_path = "Videos/00.mp4"  # Replace with your video path or 0 for webcam
  face_cascade = load_face_detector()
  frames = capture_frames(video_path, num_frames=30)
  face_frames = []
  if not frames:
    print("No frames captured!")
    exit()
  for frame in frames:
      faces = detect_faces(frame, face_cascade)
      if len(faces) > 0:
        face_roi = extract_face_rois([frame], faces)[0]  # Assuming one face per frame
        face_frames.append(face_roi)

  if not face_frames:
      print("No faces detected!")
      exit()

  preprocessed_frames = preprocess_faces(face_frames)

  if  len(preprocessed_frames) == 0:
      print("No faces detected after preprocessing!")
      exit()  # No faces detected after preprocessing

  amplitude_vibraimage = generate_amplitude_vibraimage(preprocessed_frames)
  frequency_vibraimage = generate_frequency_vibraimage(preprocessed_frames)
  amplitude_hist, frequency_hist = extract_features(amplitude_vibraimage, frequency_vibraimage)
  visualize_results(face_frames, amplitude_vibraimage, frequency_vibraimage, amplitude_hist, frequency_hist)


if __name__ == "__main__":
    main()
