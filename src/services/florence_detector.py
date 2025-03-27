import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

class FlorenceDetector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.dtype = torch.float16 if device == 'cuda' else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

    def detect_bounding_box(self, image: Image.Image, category="rectangle"):
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL.Image.Image, got {type(image)}")
        prompt = f"<OPEN_VOCABULARY_DETECTION> {category}"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.dtype)
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,
                    num_beams=8,
                    do_sample=True
                )
        except Exception as e:
            return []
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        try:
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task="<OPEN_VOCABULARY_DETECTION>",
                image_size=(image.width, image.height)
            )
            detection_results = parsed_answer.get("<OPEN_VOCABULARY_DETECTION>", {})
            bboxes = detection_results.get("bboxes", [])
            return bboxes
        except Exception as e:
            return []

    def refine_rectangle(self, image: Image.Image, bbox):
        if isinstance(bbox, list) and len(bbox) > 0 and isinstance(bbox[0], (list, tuple)):
            bbox = bbox[0]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        x_min, y_min, x_max, y_max = map(int, bbox)
        x_min = max(0, min(x_min, image_cv.shape[1] - 1))
        y_min = max(0, min(y_min, image_cv.shape[0] - 1))
        x_max = max(0, min(x_max, image_cv.shape[1] - 1))
        y_max = max(0, min(y_max, image_cv.shape[0] - 1))
        rect = (x_min, y_min, x_max - x_min, y_max - y_min)
        mask = np.zeros(image_cv.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                image_cv,
                mask,
                rect,
                bgd_model,
                fgd_model,
                5,
                cv2.GC_INIT_WITH_RECT
            )
        except Exception as e:
            return None
        binary_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype="float32")
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) != 4:
            return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype="float32")
        refined_pts = approx.reshape(4, 2)
        return refined_pts.astype("float32")
