import os

import cv2
import cv2.dnn
import mediapipe as mp
import numpy as np
import whisper
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

BASE_RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources")

GET_PATH = lambda x: os.path.join(BASE_RESOURCES_PATH, x)

BLENDSHAPE_TO_FURHAT = {
    "browDownLeft": "BROW_DOWN_LEFT",
    "browDownRight": "BROW_DOWN_RIGHT",
    "browlnnerUp": "BROW_INNER_UP",
    "browOuterUpLeft": "BROW_OUTER_UP_LEFT",
    "browOuterUpRight": "BROW_OUTER_UP_RIGHT",
    "cheekPuff": "CHEEK_PUFF",
    "cheekSquintLeft": "CHEEK_SQUINT_LEFT",
    "cheekSquintRight": "CHEEK_SQUINT_RIGHT",
    "eyeBlinkLeft": "EYE_BLINK_LEFT",
    "eyeBlinkRight": "EYE_BLINK_RIGHT",
    "eyeLookDownLeft": "EYE_LOOK_DOWN_LEFT",
    "eyeLookDownRight": "EYE_LOOK_DOWN_RIGHT",
    "eyeLookInLeft": "EYE_LOOK_IN_LEFT",
    "eyeLooklnRight": "EYE_LOOK_IN_RIGHT",
    "eyeLookOutLeft": "EYE_LOOK_OUT_LEFT",
    "eyeLookOutRight": "EYE_LOOK_OUT_RIGHT",
    "eyeLookUpLeft": "EYE_LOOK_UP_LEFT",
    "eyeLookUpRight": "EYE_LOOK_UP_RIGHT",
    "eyeSquintLeft": "EYE_SQUINT_LEFT",
    "eyeSquintRight": "EYE_SQUINT_RIGHT",
    "eyewideRight": "EYE_WIDE_RIGHT",
    "jawForward": "JAW_FORWARD",
    "jawLeft": "JAW_LEFT",
    "jawOpen": "JAW_OPEN",
    "jawRight": "JAW_RIGHT",
    "mouthClose": "MOUTH_CLOSE",
    "mouthDimpleLeft": "MOUTH_DIMPLE_LEFT",
    "mouthDimpleRight": "MOUTH_DIMPLE_RIGHT",
    "mouthFrownLeft": "MOUTH_FROWN_LEFT",
    "mouthFrownRight": "MOUTH_FROWN_RIGHT",
    "mouthFunneI": "MOUTH_FUNNEL",
    "mouthLeft": "MOUTH_LEFT",
    "mouthLowerDownLeft": "MOUTH_LOWER_DOWN_LEFT",
    "mouthLowerDownRight": "MOUTH_LOWER_DOWN_RIGHT",
    "mouthPressLeft": "MOUTH_PRESS_LEFT",
    "mouthPressRight": "MOUTH_PRESS_RIGHT",
    "mouthPucker": "MOUTH_PUCKER",
    "mouthRight": "MOUTH_RIGHT",
    "mouthRollLower": "MOUTH_ROLL_LOWER",
    "mouthRollUpper": "MOUTH_ROLL_UPPER",
    "mouthShrugLower": "MOUTH_SHRUG_LOWER",
    "mouthShrugUpper": "MOUTH_SHRUG_UPPER",
    "mouthSmileLeft": "MOUTH_SMILE_LEFT",
    "mouthSmileRight": "MOUTH_SMILE_RIGHT",
    "mouthStretchLeft": "MOUTH_STRETCH_LEFT",
    "mouthStretchRight": "MOUTH_STRETCH_RIGHT",
    "mouthUpperUpLeft": "MOUTH_UPPER_UP_LEFT",
    "mouthUpperUpRight": "MOUTH_UPPER_UP_RIGHT",
    "noseSneerLeft": "NOSE_SNEER_LEFT",
    "noseSneerRight": "NOSE_SNEER_RIGHT"
}

SR = None
YN = None


def extract_speech_text_and_timestamps(audio_path):
    """
    Uses OpenAI's whisper to transcribe audio and
    return a list of (text_segment, start, end).
    """
    model = whisper.load_model("large")  # or "medium"/"large" for better accuracy
    result = model.transcribe(audio_path, word_timestamps=True, verbose=True)

    sentence_segments = []
    word_segments = []
    for segment in result["segments"]:
        seg_text = segment["text"]
        start = segment["start"]
        end = segment["end"]
        sw = ((w["word"], w["start"], w["end"]) for w in segment["words"])
        word_segments.extend(sw)
        sentence_segments.append((seg_text, start, end))
    return sentence_segments, word_segments


def upscale_imagex2(image):
    """Upscale small images using OpenCV's DNN-based super resolution."""
    global SR
    if SR is None:
        SR = cv2.dnn_superres.DnnSuperResImpl_create()
        SR.readModel(GET_PATH("EDSR_x2.pb"))
        SR.setModel("edsr", 2)

    return SR.upsample(image)


def detect_faces(image):
    global YN
    if YN is None:
        YN = cv2.FaceDetectorYN.create(
            model=GET_PATH('face_detection_yunet_2023mar.onnx'),
            config='',
            input_size=(640, 640),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
    YN.setInputSize((image.shape[1], image.shape[0]))
    _, faces = YN.detect(image)
    if faces is not None:
        xywh = faces[0]
        x1, y1, x2, y2 = xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]
        return x1, y1, x2, y2
    return None


def pad_and_crop(image, x1, y1, x2, y2):
    """Pads the image if bbox exceeds bounds and crops the face region."""
    h, w, _ = image.shape

    # Calculate padding required
    pad_top = abs(y1) if y1 < 0 else 0
    pad_bottom = abs(y2 - h) if y2 > h else 0
    pad_left = abs(x1) if x1 < 0 else 0
    pad_right = abs(x2 - w) if x2 > w else 0

    # Pad the image if necessary
    if pad_top or pad_bottom or pad_left or pad_right:
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)

    # Adjust bbox after padding
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w + pad_left + pad_right, x2), min(h + pad_top + pad_bottom, y2)

    # Crop the face region
    return np.array(image[y1:y2, x1:x2])


def extract_facial_expressions(video_path):
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=GET_PATH("face_landmarker.task")),
        running_mode=mp_vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        num_faces=1
    )

    with mp_vision.FaceLandmarker.create_from_options(options) as face_landmarker:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expressions = []

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                detections = detect_faces(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if detections is None:
                    pbar.update(1)
                    continue

                x1, y1, x2, y2 = map(int, detections)
                margin_x = int((x2 - x1) * 0.333)
                margin_y = int((y2 - y1) * 0.333)
                x1, y1 = x1 - margin_x, y1 - margin_y
                x2, y2 = x2 + margin_x, y2 + margin_y

                face_crop = pad_and_crop(frame, x1, y1, x2, y2)
                # cv2.imshow("Face", cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(1)

                # min_face_size = 150
                # if face_crop.shape[0] < min_face_size or face_crop.shape[1] < min_face_size:
                #     face_crop = upscale_imagex2(face_crop)

                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop)
                timestamp_ms = int(round(cap.get(cv2.CAP_PROP_POS_MSEC)))
                res = face_landmarker.detect(image)
                blendshapes = res.face_blendshapes

                if blendshapes:
                    expressions.append(
                        (timestamp_ms,
                         {BLENDSHAPE_TO_FURHAT[k.category_name]: k.score for k in blendshapes[0] if
                          k.category_name in BLENDSHAPE_TO_FURHAT})
                    )

                pbar.update(1)

        cap.release()
        return expressions


def test():
    # Example usage:
    vid_path = "./st2_cut.mp4"

    s, w = extract_speech_text_and_timestamps(vid_path)
    print(w)
    # exp = extract_facial_expressions(vid_path)
    # print(f'Extracted {len(exp)} expressions.')
    # print(exp[:5])


if __name__ == "__main__":
    test()
