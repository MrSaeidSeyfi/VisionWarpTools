from src.services.image_processor import ImageProcessor

class ProcessorFactory:

    @staticmethod
    def create_processor():
        return ImageProcessor()