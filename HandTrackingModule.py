import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode # depends on if its an image or video
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # must be turned to RGB for mediapipe
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                # for drawing circles on certain landmarks of the hand
                # for id, lm in enumerate(handLms.landmark):
                #     h, w, _ = img.shape
                #     cx, cy = int(lm.x*w), int(lm.y*h)
                #     if id == 0: # particular landmark
                #         cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def main():
        cap = cv2.VideoCapture(0)
        detector = handDetector()
        FPS = FPS()
        while True:
            ret, img = cap.read()
            if ret:
                img = detector.findHands(img)
                cv2.putText(img, f"FPS: {str(FPS.update())}", (0,0), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
                cv2.imshow("Image", img)
                cv2.waitKey(1)
                
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