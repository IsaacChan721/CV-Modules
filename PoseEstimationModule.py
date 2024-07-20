import cv2
import time
import mediapipe as mp

class poseDetector(): 
    def __init__(self, static_image_mode=False, upper_body_only=False, smooth_landmarks=True, enable_segmentation=False, 
                 smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation #generates segmentation mask
        self.smooth_segmentation = smooth_segmentation #filters segmentation masks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.upper_body_only, self.smooth_landmarks, self.enable_segmentation, 
                                     self.smooth_segmentation, self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        lm_pos = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lm_pos.append([id, cx, cy])
        return lm_pos

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
    pose = poseDetector()
    counter = FPS()
    while True:
        ret, img = cap.read()
        if ret:
            img = pose.findPose(img)
            print(pose.findPosition(img))
            cv2.putText(img, f"FPS: {str(counter.update())}", (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
            cv2.imshow("Image", img)
            cv2.waitKey(1) #video
        
if __name__ == "__main__":
    main()
