from vibra_utils import capture_frames, extract_features, visualize_results, save_results
from ft_main_modular import generate_amplitude_vibraimage, generate_frequency_vibraimage, preprocess_frames
from ft_face_modular import load_face_detector, detect_faces, extract_face_rois, preprocess_faces
from simple_fq_modularized import generate_vibraimages


def main():
    print("Welcome to VibraTest!")
    print("1. Camera")
    print("2. Video")

    operation_type = int(input("Choose you need camera or video (1, 2): "))
    if operation_type == 1:
        video_path = 0
    elif operation_type == 2:
        video_path = input("Enter the video path: ")
    else:
        print("Invalid operation type!")
        exit()

    num_frames = int(input("Enter the number of frames: "))
    print("Capturing frames...")
    frames = capture_frames(video_path, num_frames=num_frames)
    print("Frames captured!")

    if frames:
        print("1. Simplified frequency analysis")
        print("2. FFT frequency analysis")
        print("3. FFT with face detection")
        analysis_type = int(input(
            "Choose you need simplified frequency analysis or fft frequency analysis or fft with face detection (1, 2, 3): "))

        print("Performing analysis...")
        if analysis_type == 1:
            amplitude_vibraimage, frequency_vibraimage = generate_vibraimages(frames)

        elif analysis_type == 2:
            frames = preprocess_frames(frames)
            amplitude_vibraimage = generate_amplitude_vibraimage(frames)
            frequency_vibraimage = generate_frequency_vibraimage(frames)

        elif analysis_type == 3:
            face_cascade = load_face_detector()
            face_frames = []
            for frame in frames:
                faces = detect_faces(frame, face_cascade)
                if len(faces) > 0:
                    face_roi = extract_face_rois([frame], faces)[0]
                    face_frames.append(face_roi)
            preprocessed_frames = preprocess_faces(face_frames)
            if len(preprocessed_frames) == 0:
                print("No faces detected after preprocessing!")
                exit()
            amplitude_vibraimage = generate_amplitude_vibraimage(preprocessed_frames)
            frequency_vibraimage = generate_frequency_vibraimage(preprocessed_frames)
        else:
            print("Invalid analysis type!")
            exit()

        print("Analysis complete!")
        print("Extracting features...")
        amplitude_hist, frequency_hist = extract_features(amplitude_vibraimage, frequency_vibraimage)
        print("Features extracted!")

        print("Would you like to save the results? (y/n)")
        save_choice = input().strip().lower()
        if save_choice == 'y':
            output_prefix = input("Enter the prefix for the output files: ")
            save_results(amplitude_vibraimage, frequency_vibraimage, amplitude_hist, frequency_hist, output_prefix)

        print("Visualizing results...")
        visualize_results(frames, amplitude_vibraimage, frequency_vibraimage, amplitude_hist, frequency_hist)
    else:
        print("No frames captured!")


if __name__ == "__main__":
    main()
