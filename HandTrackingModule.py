import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity = 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode # depends on if its an image or video
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # must be turned to RGB for mediapipe
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lm_pos = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lm_pos.append([id, cx, cy])
        return lm_pos        
    
    def findFingersUp(self, lm_pos):
        fingers = [False]*5
        
        baseHandy = lm_pos[0][2]
        
        distThumby = baseHandy-lm_pos[4][2]
        distIndexy = baseHandy-lm_pos[8][2]
        distMiddley = baseHandy-lm_pos[12][2]
        distRingy = baseHandy-lm_pos[16][2]
        distPinkyy = baseHandy-lm_pos[20][2]
      
        #thumb
        if lm_pos[4][2] < lm_pos[3][2] and distThumby > 100:
            fingers[0] = True
        #index
        if lm_pos[8][2] < lm_pos[7][2] and distIndexy > 175:
            fingers[1] = True
        #middle
        if lm_pos[12][2] < lm_pos[11][2] and distMiddley > 200:
            fingers[2] = True
        #ring
        if lm_pos[16][2] < lm_pos[15][2] and distRingy > 185:
            fingers[3] = True
        #pinky     
        if lm_pos[20][2] < lm_pos[19][2] and distPinkyy > 150:
            fingers[4] = True

        return fingers
    
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
    detector = handDetector()
    counter = FPS()
    while True:
        ret, img = cap.read()
        if ret:
            img = detector.findHands(img)
            if detector.findPosition(img):
                print(detector.findFingersUp(detector.findPosition(img)))
            cv2.putText(img, f"FPS: {str(counter.update())}", (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
            cv2.imshow("Image", img)
            cv2.waitKey(1) # 1 = video, 0 = image

if __name__ == "__main__": 
    main()