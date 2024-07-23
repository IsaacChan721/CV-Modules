import cv2
import time
import mediapipe as mp

class faceDetector():
    def __init__(self, model_selection = 0, min_detection_confidence=0.5):
        self.model_selection = model_selection #0 for short range, 1 for long range
        self.min_detection_confidence = min_detection_confidence
        
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.face = self.mpFaceDetection.FaceDetection(self.model_selection, 
                                                       self.min_detection_confidence)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face.process(imgRGB)
        if results.detections:
            for id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                H, W, _ = img.shape
                bbox = int(bboxC.xmin*W), int(bboxC.ymin*H), int(bboxC.width*W), int(bboxC.height*H)
                cv2.rectangle(img, bbox, (0, 0, 0), 2)
                cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        
class FPS():
    def __init__(self):
        self.start = time.time()
        self.frames = 0
        self.fps = 0
        
    def update(self):
        self.frames += 1
        if time.time() - self.start >= 1:
            self.fps = self.frames
            self.frames = 0
            self.start = time.time()
        return self.fps
    
def main():
    cap = cv2.VideoCapture(0)
    detector = faceDetector()
    counter = FPS()
    
    while True:
        ret, img = cap.read()
        if ret:
            detector.findFaces(img)
            cv2.putText(img, f"FPS: {str(counter.update())}", (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
            cv2.imshow("Image", img)
            cv2.waitKey(1)

if __name__ == '__main__':
    main()