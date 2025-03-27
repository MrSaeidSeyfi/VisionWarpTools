import gradio as gr
from PIL import Image, ImageDraw
from src.services.factory import ProcessorFactory

def draw_markers(image, points):
    if image is None:
        return None
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    for i, (x, y) in enumerate(points):
        draw.ellipse((x-5, y-5, x+5, y+5), fill="red")
        draw.text((x+7, y-7), str(i+1), fill="red")
    return img_copy

def auto_detect(image, model_path, category):
    if isinstance(image, str):
        try:
            image = Image.open(image).convert("RGB")
        except Exception as e:
            return image, None, f"Error: Invalid image path - {str(e)}"
    elif not isinstance(image, Image.Image):
        return image, None, "Error: Image must be a PIL Image or valid file path"
    processor = ProcessorFactory.create_processor()
    # Now passing the user-specified model_path and category to auto_process
    warped_img, matrices, detected_points = processor.auto_process(image, model_path, category)
    if detected_points is None:
        return image, None, matrices
    image_marked = draw_markers(image, detected_points)
    return image_marked, warped_img, matrices

def main():
    with gr.Blocks(title="App") as demo:
        gr.Markdown("## Vision warp tools \nUpload an image, specify the model path and detection category, then click **Auto-detect Corners**.")
        with gr.Row():
            image_input = gr.Image(label="Upload Image", type="pil", interactive=True)
        with gr.Row():
            model_path_input = gr.Textbox(label="Model Path Folder", value=r"E:\Project\test\Florence2-large\model")
            category_input = gr.Textbox(label="Detection Category", value="rectangle")
        auto_btn = gr.Button("Auto-detect Corners")
        with gr.Row():
            warped_output = gr.Image(label="Warped Image")
        matrices_output = gr.Textbox(label="Computed Matrices", lines=10)
        auto_btn.click(fn=auto_detect, inputs=[image_input, model_path_input, category_input],
                       outputs=[image_input, warped_output, matrices_output])
    demo.launch()

if __name__ == "__main__":
    main()
