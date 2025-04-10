# Vision Warp Tools

A computer vision tool that automatically detects and rectifies perspective distortion in images using the Florence model.

## Features

- Automatic rectangle/document detection
- Perspective correction and warping
- Matrix computation for 3D transformations
- User-friendly Gradio interface

## How It Works

Vision Warp Tools uses the Florence detector model to identify rectangular objects in images, then applies perspective transformation to create a frontal view of the detected area.

The tool:
1. Detects rectangular objects in the image
2. Refines the corners using computer vision techniques
3. Computes homography matrices for perspective transformation
4. Outputs the warped (rectified) image

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- PIL (Pillow)
- Transformers
- Gradio

## Usage

1. Clone the repository
2. Install dependencies
3. Run the application: `python app.py`
4. Upload an image through the Gradio interface
5. Specify the Florence model path and detection category (e.g., "rectangle")
6. Click "Auto-detect Corners" to process the image


## Result
![photo18039697937](https://github.com/user-attachments/assets/4270381f-a225-467a-8a83-676dd6f1ee23)
![photo18039698202](https://github.com/user-attachments/assets/1f214916-8fbb-4d98-a3fd-5d9108d180a7)
![photo18039698048](https://github.com/user-attachments/assets/24136f0a-d230-41cc-8251-4b34a0ffa5c7)


## License

MIT
