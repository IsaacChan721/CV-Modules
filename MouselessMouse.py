import cv2
import time
import HandTrackingModule as htm

def canLeftClick(landmarks): # raising 2 fingers
    lmyMin = 9999
    for landmark in landmarks:
        if lmyMin > landmark[2]:
            lmyMin = landmark[2]
    x1, y1, x2, y2 = landmarks[8][1], landmarks[8][2], landmarks[12][1], landmarks[12][2]
    dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
    if dist <= 40 and lmyMin == landmarks[12][2]:
        return True
    else:
        return False 
    
def canRightClick(landmarks): # rasing 3 fingers
    lmyMin = 9999
    for landmark in landmarks:
        if lmyMin > landmark[2]:
            lmyMin = landmark[2]
    x1, y1, x2, y2 = landmarks[8][1], landmarks[8][2], landmarks[16][1], landmarks[16][2]
    dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
    if dist <= 80 and lmyMin == landmarks[12][2]:
        return True
    else:
        return False 

def canDrag(landmarks): # raising pinky finger
    lmyMin = 9999
    for landmark in landmarks:
        if lmyMin > landmark[2]:
            lmyMin = landmark[2]
    if lmyMin == landmarks[20][2]:
        return True
    else:
        return False

def canScrollUp(landmarks): # thumbs up
    lmyMin = 9999
    for landmark in landmarks:
        if lmyMin > landmark[2]:
            lmyMin = landmark[2]
    if lmyMin == landmarks[4][2]:
        return True
    else:
        return False
    
def canScrollDown(landmarks): # thumbs down
    lmyMax = 0
    for landmark in landmarks:
        if lmyMax < landmark[2]:
            lmyMax = landmark[2]
    if lmyMax == landmarks[4][2]:
        return True
    else:
        return False

def canZoomIn(landmarks): # pinching
    x1, y1, x2, y2 = landmarks[4][1], landmarks[4][2], landmarks[8][1], landmarks[8][2]
    dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
    if dist <= 40:
        return True
    else:
        return False

def canZoomOut(landmarks): # check mark with hands
    x1, y1, x2, y2 = landmarks[4][1], landmarks[4][2], landmarks[8][1], landmarks[8][2]
    dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
    if dist >= 200:
        return True
    else:
        return False

def main():
    cap = cv2.VideoCapture(0)
    detector = htm.handDetector()
    startTime = time.time()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        landmarks = detector.findPosition(img)
        if time.time() - startTime >= 1 and landmarks:
            if canZoomIn(landmarks):
                print("Zoom In")
            elif canZoomOut(landmarks):
                print("Zoom Out")
            elif canLeftClick(landmarks):
                print("Left Click")
            elif canRightClick(landmarks):
                print("Right Click")
            elif canDrag(landmarks):
                print("Drag")
            elif canScrollUp(landmarks):
                print("Scroll Up")
            elif canScrollDown(landmarks):
                print("Scroll Down")
            startTime = time.time()
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
if __name__ == '__main__':
    main()