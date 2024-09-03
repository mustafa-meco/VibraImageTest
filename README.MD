
# VibraImage Project

This project demonstrates various techniques for generating and visualizing vibraimages using Python, OpenCV, and Matplotlib. The project includes modularized scripts: `simple_fq_modularized.py`, `ft_main_modular.py`, and `ft_face_modular.py`, which utilize a utility module `vibra_utils.py` for common tasks.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

Install the required packages using pip:

```sh
pip install opencv-python numpy matplotlib
```

## Scripts

### `simple_fq_modularized.py`

This script captures frames from a webcam, converts them to grayscale, and generates amplitude and frequency vibraimages. It also visualizes the results using Matplotlib. The script uses functions from `vibra_utils.py` for capturing frames, extracting features, and visualizing results.

#### Usage

```sh
python simple_fq_modularized.py
```

#### Key Steps

1. **Image Acquisition**: Captures 30 frames from the webcam.
2. **Vibraimage Generation**: Calculates amplitude and frequency vibraimages.
3. **Feature Extraction & Analysis**: Generates histograms for amplitude and frequency.
4. **Visualization**: Displays the original frame, amplitude vibraimage, frequency vibraimage, and histograms.

#### Example Output

![simple_fq_modularized.py Output](Images/simple_fq.png)

#### Functions

- `generate_vibraimages(frames)`: Generates amplitude and frequency vibraimages from a list of frames.
- `main()`: Main function to capture frames, generate vibraimages, extract features, and visualize results.

### `ft_main_modular.py`

This script captures frames from a webcam, converts them to grayscale, and generates amplitude and frequency vibraimages using Fourier Transform. It also visualizes the results using Matplotlib. The script uses functions from `vibra_utils.py` for capturing frames, extracting features, and visualizing results.

#### Usage

```sh
python ft_main_modular.py
```

#### Key Steps

1. **Image Acquisition**: Captures 30 frames from the webcam.
2. **Vibraimage Generation**: Calculates amplitude and frequency vibraimages using Fourier Transform.
3. **Feature Extraction & Analysis**: Generates histograms for amplitude and frequency.
4. **Visualization**: Displays the original frame, amplitude vibraimage, frequency vibraimage, and histograms.

#### Example Output

![ft_main_modular.py Output](Images/ft_main_output.png)

#### Functions

- `preprocess_frames(frames)`: Preprocesses frames by converting them to a 3D numpy array.
- `generate_amplitude_vibraimage(frames)`: Generates the amplitude vibraimage using equation 1.
- `generate_frequency_vibraimage(frames)`: Generates the frequency vibraimage using Fourier Transform.
- `main()`: Main function to capture frames, preprocess them, generate vibraimages, extract features, and visualize results.

### `ft_face_modular.py`

This script captures frames from a webcam, detects faces using Haar Cascade, and generates amplitude and frequency vibraimages for the detected face regions. It also visualizes the results using Matplotlib. The script uses functions from `vibra_utils.py` for capturing frames, extracting features, and visualizing results.

#### Usage

```sh
python ft_face_modular.py
```

#### Key Steps

1. **Load Face Detection Model**: Loads the Haar Cascade model for face detection.
2. **Image Acquisition**: Captures 30 frames from the webcam and detects faces.
3. **Vibraimage Generation**: Calculates amplitude and frequency vibraimages for the detected face regions.
4. **Feature Extraction & Analysis**: Generates histograms for amplitude and frequency.
5. **Visualization**: Displays the face region, amplitude vibraimage, frequency vibraimage, and histograms.

#### Example Output

![ft_face_modular.py Output](Images/ft_face.png)

#### Functions

- `load_face_detector()`: Loads the Haar Cascade face detection model.
- `detect_faces(gray_frame, face_cascade)`: Detects faces in a given frame using the cascade classifier.
- `extract_face_rois(frames, faces)`: Extracts face regions of interest from a list of frames and their corresponding face detections.
- `preprocess_faces(face_rois)`: Preprocesses the list of face regions by ensuring all have the same size and converting to a 3D numpy array.
- `main()`: Main function to capture frames, detect faces, preprocess face regions, generate vibraimages, extract features, and visualize results.

## Utility Module

### `vibra_utils.py`

This module contains utility functions for capturing frames, extracting features, and visualizing results. It is used by the main scripts to avoid code repetition and improve modularity.

#### Functions

- `capture_frames(video_path, num_frames=30)`: Captures frames from a video or camera.
- `extract_features(amplitude_vibraimage, frequency_vibraimage)`: Extracts features (histograms) from vibraimages.
- `visualize_results(frames, amplitude_vibraimage, frequency_vibraimage, amplitude_hist, frequency_hist)`: Visualizes frames, vibraimages, and histograms.

