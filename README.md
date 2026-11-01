# ImageRecognition — YOLOv4 Object Detection

A 2021 computer vision project exploring **custom object detection with YOLOv4 and Darknet**, including unwanted hand-gesture detection as an application.

## Project scope

The repository combines the Darknet framework with supporting Python scripts for the project workflow and a CUDA/OpenCV environment check.

- `darknet-master/`: bundled Darknet framework files. Framework code and its authorship remain separate from the project-specific work.
- `create_list_of_images.py`: image-list preparation script; review and adapt its paths to your environment.
- `test_script.py`: compares CPU and CUDA matrix operations through OpenCV. This is an environment check, not an object-detection accuracy benchmark.

## Environment

For the Python CUDA check, use Python, NumPy, an OpenCV build with CUDA support, and a compatible NVIDIA GPU / CUDA environment. Darknet training or detection additionally requires a compatible framework build, the relevant model configuration, class definitions and weights.

## Run the existing CUDA check

```bash
git clone https://github.com/rootroxox/ImageRecognition.git
cd ImageRecognition
python -m pip install numpy
python test_script.py
```

Configure CUDA-enabled OpenCV before running the script. The script prints elapsed-time lines labelled `CUDA` and `CPU`; timings depend on the machine and environment. No representative timings are claimed here.

## Detection workflow

1. Prepare the labelled images and class definitions for the intended objects or gestures.
2. Review the dataset paths and Darknet / YOLOv4 configuration.
3. Build Darknet for the target environment and use matching model weights.
4. Evaluate false positives, false negatives and inference performance on the intended inputs.

## Technology

YOLOv4 · Darknet · OpenCV · CUDA · Python

## Author

[Ayberk Saygı](https://github.com/rootroxox) · [LinkedIn](https://www.linkedin.com/in/ayberk-s-0640811a4/)
