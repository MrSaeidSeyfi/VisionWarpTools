from src.services.image_processor import ImageProcessor
from src.services.florence_detector import FlorenceDetector

class ProcessorFactory:
    @staticmethod
    def create_processor():
        return ImageProcessor()
    
    @staticmethod
    def create_florence_detector(model_path):
        return FlorenceDetector(model_path)
