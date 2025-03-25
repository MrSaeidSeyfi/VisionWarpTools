import cv2
import numpy as np
from PIL import Image

class ImageProcessor:
    def process(self, image, points):
        """Warp the image and compute the homography and projection matrices using provided points."""
        if not image or len(points) != 4:
            return None, "Error: Please provide an image and exactly 4 points."
        
        # Convert PIL image to OpenCV format (BGR)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        pts = np.array(points, dtype="float32")
        
        rect = self.order_points(pts)
        width = max(
            np.linalg.norm(rect[0] - rect[1]),
            np.linalg.norm(rect[2] - rect[3])
        )
        height = max(
            np.linalg.norm(rect[0] - rect[3]),
            np.linalg.norm(rect[1] - rect[2])
        )
        
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        H = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image_cv, H, (int(width), int(height)))
        
        # Compute Projection Matrix (approximation assuming identity intrinsic matrix)
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        norm_factor = np.linalg.norm(h1)
        if norm_factor == 0:
            return None, "Error: Invalid homography matrix."
        r1 = h1 / norm_factor
        r2 = h2 / norm_factor
        t = h3 / norm_factor
        r3 = np.cross(r1, r2)
        P = np.column_stack((r1, r2, r3, t))
        
        matrices_str = f"Homography Matrix (H):\n{H}\n\nProjection Matrix (P):\n{P}"
        
        warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        return warped_pil, matrices_str

    def auto_process(self, image):
        """
        Automatically detects a quadrilateral in the image using contour detection,
        then warps the image and computes the homography and projection matrices.
        Returns the warped image, computed matrices, and detected corner points.
        """
        if image is None:
            return None, "Error: No image provided.", None

        # Convert PIL image to OpenCV format (BGR)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, "Error: No contours found.", None
        
        # Assume the largest contour corresponds to the object of interest
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) != 4:
            return None, "Error: Could not find 4 corner points automatically.", None
        
        pts = approx.reshape(4, 2).astype("float32")
        rect = self.order_points(pts)
        
        width = max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3]))
        height = max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2]))
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        H = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image_cv, H, (int(width), int(height)))
        
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        norm_factor = np.linalg.norm(h1)
        if norm_factor == 0:
            return None, "Error: Invalid homography matrix.", None
        r1 = h1 / norm_factor
        r2 = h2 / norm_factor
        t = h3 / norm_factor
        r3 = np.cross(r1, r2)
        P = np.column_stack((r1, r2, r3, t))
        
        matrices_str = f"Homography Matrix (H):\n{H}\n\nProjection Matrix (P):\n{P}"
        warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        
        return warped_pil, matrices_str, rect

    def order_points(self, pts):
        """Orders points as top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
