import cv2

class ImageCapture:
    def __init__(self, camera_id=0, output_dir='captured_images'):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.output_dir = output_dir

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            raise Exception("Failed to capture image")

    def save_image(self, image, filename):
        cv2.imwrite(f"{self.output_dir}/{filename}", image)

    def release(self):
        self.cap.release()

# Example usage
if __name__ == "__main__":
    image_capture = ImageCapture()
    image = image_capture.capture_image()
    image_capture.save_image(image, "test_image.jpg")
    image_capture.release()

