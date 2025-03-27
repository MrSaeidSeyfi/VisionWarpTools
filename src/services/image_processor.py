import cv2
import numpy as np
from PIL import Image
from src.services.florence_detector import FlorenceDetector

class ImageProcessor:
    def auto_process(self, image, model_path, category):
        if image is None:
            return None, "Error: No image provided.", None
        if isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except Exception as e:
                return None, f"Error: Invalid image path - {str(e)}", None
        elif not isinstance(image, Image.Image):
            return None, "Error: Image must be a PIL Image", None

        detector = FlorenceDetector(model_path=model_path)
        bboxes = detector.detect_bounding_box(image, category=category)
        if not bboxes or len(bboxes) == 0:
            return None, "Error: Could not detect bounding box.", None

        bbox = bboxes[0]
        refined_pts = detector.refine_rectangle(image, bbox)
        if refined_pts is None:
            return None, "Error: Could not refine rectangle.", None

        rect = self.order_points(refined_pts)
        area = cv2.contourArea(rect)
        if area < 1e-5:
            return None, "Error: Degenerate quadrilateral.", None

        width = max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3]))
        height = max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2]))
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        H = cv2.getPerspectiveTransform(rect, dst)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        warped = cv2.warpPerspective(image_cv, H, (int(width), int(height)))
        cv2.imwrite("warped_image.jpg", warped)
        warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        norm_factor = np.linalg.norm(h1)
        if norm_factor < 1e-10:
            matrices_str = f"H:\n{H}\n\nP:\nNot computed"
        else:
            r1 = h1 / norm_factor
            r2 = h2 / norm_factor
            t = h3 / norm_factor
            r3 = np.cross(r1, r2)
            P = np.column_stack((r1, r2, r3, t))
            matrices_str = f"H:\n{H}\n\nP:\n{P}"
        return warped_pil, matrices_str, rect

    def order_points(self, pts):
        pts = np.array(pts, dtype="float32")
        rect = np.zeros((4, 2), dtype="float32")
        centroid = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_pts = pts[np.argsort(angles)]
        top_left_idx = np.argmin(sorted_pts[:, 1] + sorted_pts[:, 0] * 0.001)
        rect[0] = sorted_pts[top_left_idx]
        remaining_pts = np.delete(sorted_pts, top_left_idx, axis=0)
        angles_from_tl = np.arctan2(remaining_pts[:, 1] - rect[0, 1], remaining_pts[:, 0] - rect[0, 0])
        remaining_pts = remaining_pts[np.argsort(angles_from_tl)]
        rect[1], rect[2], rect[3] = remaining_pts[0], remaining_pts[1], remaining_pts[2]
        return rect
