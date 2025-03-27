import gradio as gr
from PIL import Image, ImageDraw
from src.services.factory import ProcessorFactory

selected_points = []

def draw_markers(image, points):
    if image is None:
        return None
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    for i, (x, y) in enumerate(points):
        draw.ellipse((x-5, y-5, x+5, y+5), fill="red")
        draw.text((x+7, y-7), str(i+1), fill="red")
    return img_copy

def select_point(img, evt: gr.SelectData):
    global selected_points
    if len(selected_points) < 4:
        selected_points.append((evt.index[0], evt.index[1]))
    return draw_markers(img, selected_points)

def reset_points():
    global selected_points
    selected_points = []
    return None

def process_image(image):
    if len(selected_points) != 4:
        return None, "Error: Please select exactly 4 points."
    processor = ProcessorFactory.create_processor()
    warped_img, matrices = processor.process(image, selected_points)
    reset_points()  # clear selected points after processing
    return warped_img, matrices

def auto_detect(image):
    if isinstance(image, str):
        try:
            image = Image.open(image).convert("RGB")
        except Exception as e:
            return image, None, f"Error: Invalid image path - {str(e)}"
    elif not isinstance(image, Image.Image):
        return image, None, "Error: Image must be a PIL Image or valid file path"
    processor = ProcessorFactory.create_processor()
    warped_img, matrices, detected_points = processor.auto_process(image)
    if detected_points is None:
        return image, None, matrices
    global selected_points
    selected_points = [tuple(pt) for pt in detected_points]
    image_marked = draw_markers(image, selected_points)
    return image_marked, warped_img, matrices

def main():
    with gr.Blocks(title="Image Warp & Projection Matrix App") as demo:
        gr.Markdown("## Image Warp & Projection Matrix App\nUpload an image, click 4 points manually or use **Auto-detect Corners**, and then click **Process**.")
        with gr.Row():
            image_input = gr.Image(label="Upload Image", type="pil", interactive=True)
        image_input.select(select_point, inputs=[image_input], outputs=[image_input])
        process_btn = gr.Button("Process")
        auto_btn = gr.Button("Auto-detect Corners")
        reset_btn = gr.Button("Reset Points")
        with gr.Row():
            warped_output = gr.Image(label="Warped Image")
        matrices_output = gr.Textbox(label="Computed Matrices", lines=10)
        process_btn.click(fn=process_image, inputs=[image_input], outputs=[warped_output, matrices_output])
        auto_btn.click(fn=auto_detect, inputs=[image_input], outputs=[image_input, warped_output, matrices_output])
        reset_btn.click(fn=reset_points, inputs=[], outputs=[image_input])
    demo.launch()

if __name__ == "__main__":
    main()
