import os

import cv2
import cv2.dnn
import mediapipe as mp
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from matplotlib import pyplot as plt
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from process import pad_and_crop, detect_faces


# https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb
def draw_landmarks_on_image(rgb_image, detection_result, draw_mesh=False):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        if draw_mesh:
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image


def remove_attr(face_blendshapes_list, min_significant_ratio=0.03, absolute_threshold=0.05, min_kept_n=10):
    """
    Identifies insignificant blendshape attributes that should be removed from plotting.

    Parameters:
    - face_blendshapes_list: List of face blendshape results (each item is a list of FaceBlendshape objects).
    - min_significant_ratio: The minimum fraction of frames where an attribute must exceed `absolute_threshold` to be kept.
                             (Default 0.03 means it must be significant in at least 3% of frames)
    - absolute_threshold: The score below which an attribute is considered insignificant. (Default 0.05)
    - min_kept_n: The minimum number of attributes to keep, regardless of significance. (Default 10)

    Returns:
    - List of attribute names that should be removed from plotting.
    """

    if not face_blendshapes_list or len(face_blendshapes_list) == 0:
        return []

    # Step 1: Extract all attribute names (assuming they are the same for all frames)
    all_attr_names = [cat.category_name for cat in face_blendshapes_list[0]]

    # Step 2: Initialize storage for scores
    attr_scores = {name: [] for name in all_attr_names}

    # Step 3: Collect scores across all frames
    total_frames = 0
    for frame_blendshapes in face_blendshapes_list:
        if frame_blendshapes is not None:
            for cat in frame_blendshapes:
                total_frames += 1
                attr_scores[cat.category_name].append(cat.score)

    # Step 4: Determine which attributes to remove
    if min_kept_n >= len(all_attr_names):
        return []  # No attributes to remove
    attrs_to_remove = []
    attr_score_list = list(attr_scores.items())
    attr_score_list.sort(key=lambda x: np.mean(x[1]), reverse=True)
    for attr, scores in attr_score_list[min_kept_n:]:
        scores = np.array(scores)  # Convert to NumPy array for easier computation

        # Count how many times this attribute's score is "significant"
        significant_count = np.sum(scores > absolute_threshold)

        # If it's significant in less than `min_significant_ratio` of frames, remove it
        if (significant_count / total_frames) < min_significant_ratio:
            attrs_to_remove.append(attr)

    return attrs_to_remove


def plot_face_blendshapes_bar_graph(face_blendshapes, remove_attrs=None):
    """
        Plots a horizontal bar graph of face blendshapes while allowing certain attributes to be removed.

        Parameters:
        - face_blendshapes: A list of FaceBlendshape objects.
        - remove_attrs: A list of attribute names to exclude from the plot (default: None).

        Returns:
        - A NumPy image array of the plot.
        """

    if remove_attrs is None:
        remove_attrs = []

    # Filter blendshapes by removing unwanted attributes
    filtered_blendshapes = [cat for cat in face_blendshapes if cat.category_name not in remove_attrs]

    # Extract data for plotting
    face_blendshapes_names = [cat.category_name for cat in filtered_blendshapes]
    face_blendshapes_scores = [cat.score for cat in filtered_blendshapes]
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.barh(face_blendshapes_ranks, face_blendshapes_scores)
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    for score, patch in zip(face_blendshapes_scores, bars.patches):
        ax.text(
            patch.get_width(),
            patch.get_y() + 0.5 * patch.get_height(),
            f"{score:.4f}",
            va="center"
        )

    ax.set_xlabel('Score')
    ax.set_title('Face Blendshapes')
    plt.tight_layout()

    # Draw the figure in memory
    fig.canvas.draw()

    # Extract RGBA buffer from canvas
    buf = fig.canvas.buffer_rgba()
    (w, h) = fig.canvas.get_width_height()

    # Convert to a NumPy array (h, w, 4)
    plot_img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)

    # Optionally remove the alpha channel for a 3-channel RGB image
    plot_img = plot_img[..., :3]  # shape => (h, w, 3)

    # Clean up
    plt.close(fig)

    return plot_img


def letterbox_pad(image, target_w, target_h):
    """
    Letterbox-pad (or black-pad) the given image to exactly target_w x target_h.
    Preserves aspect ratio of `image` and centers it on a black background.
    """
    if image is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    h, w, _ = image.shape
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Center the resized image
    x_start = (target_w - new_w) // 2
    y_start = (target_h - new_h) // 2
    padded[y_start:y_start + new_h, x_start:x_start + new_w] = resized

    return padded


def create_facial_expression_video(input_video_path, output_video_path):
    """
    1) Reads the input video frame-by-frame.
    2) For each frame, tries to detect a face and run the mesh + blendshapes.
       - If successful, crops & annotates face, and generates bar-plot image.
       - If failure, use blank face crop and blank bar-plot image.
    3) Finds the largest face crop dimension from any successful detection.
    4) Letterbox-pads all face crops and bar-plots to that max dimension.
    5) Concatenates face crop and bar-plot horizontally.
    6) Writes out the combined frames at the original FPS.
    """

    # --- 1) Setup Face Landmarker (similar to your existing code) ---
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="resources/face_landmarker.task"),
        running_mode=mp_vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        num_faces=1
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # --- 2) Read Input Video ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Lists to store each frame’s results
    face_crops = []
    bar_plots_param = []

    # We track largest width/height from successfully-detected faces
    max_face_w = 0
    max_face_h = 0

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detection = detect_faces(frame_bgr)  # or frame_rgb, depending on your detect_faces() usage

            if detection is not None:
                # Unpack bounding box
                x1, y1, x2, y2 = map(int, detection)
                margin_x = int((x2 - x1) * 0.333)
                margin_y = int((y2 - y1) * 0.333)
                x1, y1 = x1 - margin_x, y1 - margin_y
                x2, y2 = x2 + margin_x, y2 + margin_y

                # Crop face
                face_crop = pad_and_crop(frame_rgb, x1, y1, x2, y2)

                # Annotate face mesh on the crop (optional).
                #   - If you prefer the entire frame annotated, call draw_landmarks_on_image on `frame_rgb` with face region.
                #   - Otherwise, detect on the cropped region, then draw on it.
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop)
                result = face_landmarker.detect(mp_image)

                if result.face_landmarks:
                    # Draw mesh on the face crop
                    annotated_crop = draw_landmarks_on_image(face_crop, result)
                else:
                    annotated_crop = face_crop

                # Generate bar-plot image from blendshapes (if available)
                if result.face_blendshapes:
                    # bar_img = plot_face_blendshapes_bar_graph(result.face_blendshapes[0])
                    bar_img_param = result.face_blendshapes[0]
                else:
                    # blank bar if no blendshapes
                    bar_img_param = None

                # Record dimension for reference
                h, w, _ = annotated_crop.shape
                max_face_w = max(max_face_w, w)
                max_face_h = max(max_face_h, h)

                # Append to lists
                face_crops.append(annotated_crop)
                bar_plots_param.append(bar_img_param)

            else:
                # Detection fails => blank face crop & blank bar plot
                face_crops.append(None)
                bar_plots_param.append(None)

            pbar.update(1)

    cap.release()
    face_landmarker.close()

    # If no frames with a successful detection, just exit
    if max_face_w == 0 or max_face_h == 0:
        print("No successful face detection in the entire video. Aborting.")
        return

    bar_plots = []
    attr_to_remove = remove_attr(bar_plots_param)
    for bar_img_param in bar_plots_param:
        if bar_img_param is not None:
            bar_img = plot_face_blendshapes_bar_graph(bar_img_param, remove_attrs=attr_to_remove)
            bar_plots.append(bar_img)
        else:
            bar_plots.append(None)

    # --- 3) Letterbox-pad everything to max_face_w, max_face_h ---
    final_frames = []
    for face_crop, bar_img_param in zip(face_crops, bar_plots):
        # Face
        padded_face = letterbox_pad(face_crop, max_face_w, max_face_h)

        # Bar
        if bar_img_param is not None:
            # Pad bar to match the same height as face
            bar_h, bar_w, _ = bar_img_param.shape
            scale = max_face_h / bar_h
            new_bar_w = int(bar_w * scale)
            new_bar_h = int(bar_h * scale)

            resized_bar = cv2.resize(bar_img_param, (new_bar_w, new_bar_h))
            # Now letterbox pad the bar to the same (max_face_h, new width to match face or
            # keep it separate). Usually we just match heights and put them side by side.
            # For simplicity, we won't pad the bar to the same width—just match heights.
            # Then horizontally stack.
            combined = np.hstack((padded_face, resized_bar))
        else:
            # No bar => black canvas of same size as face
            black_bar = np.zeros_like(padded_face)
            combined = np.hstack((padded_face, black_bar))

        final_frames.append(combined)

    # --- 4) Write to video with original FPS ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_h, out_w, _ = final_frames[0].shape
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

    for frame_rgb in final_frames:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved to {output_video_path}")


def test():
    # Example usage:
    vid_path = "./st2_cut.mp4"
    exp = "./face_annotated.mp4"
    create_facial_expression_video(vid_path, exp)
    # exp = extract_facial_expressions(vid_path)
    # print(f'Extracted {len(exp)} expressions.')
    # print(exp[:5])


if __name__ == "__main__":
    test()
