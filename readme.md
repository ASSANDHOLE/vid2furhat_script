## Automated Comedy Replication with Virtual Furhat Agent

### What This Project Does

This system converts stand-up comedy videos into synchronized performance scripts for the [Furhat](https://docs.furhat.io/getting_started/) virtual agent, automating facial gesture and voice reproduction from raw input. 

### How It Works

- **Video-to-Gesture Extraction:** Facial keypoints are extracted using MediaPipe's Face Landmarker and resampled at `N` FPS to align with speech. These are converted into blendshape coefficients controlling over 40 expressions for the Furhat platform.

- **Speech-to-Audio Generation:** Each line is paired with a synthesized voice using a zero-shot TTS model. A LLaMA-based system mimics the original speaker's voice using short reference clips, producing speech outputs that match the original cadence and tone.
(Or you can use the default TTS module from Furhat SDK.)

- **Script Generation:** Outputs are encoded as Kotlin scripts compatible with the Furhat SDK, supporting direct playback on the robot or in simulation.

- **Automation:** The entire process -- from video ingestion to Furhat-ready output -- is fully automated, making it suitable for high-volume replication tasks.

---

## Requirements

> Excluding TTS module under `./TTS/`
### 1. PyTorch

https://pytorch.org/get-started/locally/

Tested w/ torch 2.6.0 w/ cuda 12.6

### Other libs
```bash
pip install opencv-python mediapipe openai-whisper tqdm matplotlib
```

### Models
Download [https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) and save it to
`./resources` (i.e., `./resources/face_detection_yunet_2023mar.onnx`), it is a lightweight face detection library that runs very efficiently on CPU.
See [OpenCV Zoo](https://github.com/opencv/opencv_zoo).

## Run the code
If you want to use LLaSa, set up the environment according to [TTS/llasa_quant/readme.md](./TTS/llasa_quant/readme.md),
Prepare the target video file and set the `video_path` in `main` function of [main_custom_tts.py](./main_custom_tts.py).
