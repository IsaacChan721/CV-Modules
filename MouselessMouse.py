import cv2
import time
import numpy as np
import HandTrackingModule as htm
import pyautogui as pag

frameW, frameH = 80, 60 # frame width and height
camW, camH = 640, 480 # frame height
screenW, screenH = pag.size() # screen width and height

def minLandmark(list):
    min = 9999
    for i in range(len(list)):
        if min > list[i][2]:
            min = list[i][2]
    return min
            
def maxLandmark(list):
    max = 0
    for i in range(len(list)):
        if max < list[i][2]:
            max = list[i][2]
    return max

def canMoveMouse(landmarks): # moving index finger
    lmyMin = minLandmark(landmarks)
    if lmyMin == landmarks[8][2]:
        return True
    else:
        return False
    
def moveMouse(x, y):
    # map values
    x2 = np.interp(x, (frameW, camW-frameW), (0, screenW))
    y2 = np.interp(y, (frameH, camH-frameH), (0, screenH))
    pag.moveTo(x2,y2)

def canLeftClick(landmarks): # raising 2 fingers
    lmyMin = minLandmark(landmarks)
    x1, y1, x2, y2 = landmarks[8][1], landmarks[8][2], landmarks[12][1], landmarks[12][2]
    dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
    if dist <= 40 and lmyMin == landmarks[12][2]:
        return True
    else:
        return False 
    
def leftClick():
    pag.click(button='left')
    
def canRightClick(landmarks): # rasing 3 fingers
    lmyMin = minLandmark(landmarks)
    x1, y1, x2, y2 = landmarks[8][1], landmarks[8][2], landmarks[16][1], landmarks[16][2]
    dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
    if dist <= 60 and lmyMin == landmarks[12][2]:
        return True
    else:
        return False 

def rightClick():
    pag.click(button='right')

def canDrag(landmarks): # raising pinky finger
    lmyMin = minLandmark(landmarks)
    if lmyMin == landmarks[20][2]:
        return True
    else:
        return False
    
def drag(x, y):
    # map values
    x2 = np.interp(x, (frameW, screenW-frameW), (0, screenW))
    y2 = np.interp(y, (frameH, screenH-frameH), (0, screenH))
    pag.dragTo(x2, y2, button='left')

def canScrollUp(landmarks): # thumbs up
    lmyMin = minLandmark(landmarks)
    if lmyMin == landmarks[4][2]:
        return True
    else:
        return False
    
def scrollUp():
    pag.scroll(1)
    
def canScrollDown(landmarks): # thumbs down
    lmyMax = maxLandmark(landmarks)
    if lmyMax == landmarks[4][2]:
        return True
    else:
        return False
    
def scrollDown():
    pag.scroll(-1)

def main():
    cap = cv2.VideoCapture(0)
    detector = htm.handDetector()
    startTime = time.time()
    while True:
        success, img = cap.read() # image size of (480, 640, 3)
        img = cv2.flip(img, 1)
        img = cv2.rectangle(img, (frameW, frameH), (camW-frameW, camH-frameH), (0, 255, 0), 2)
        if success:
            img = detector.findHands(img)
            landmarks = detector.findPosition(img)
            if time.time() - startTime >= 1 and landmarks:
                if canLeftClick(landmarks):
                    leftClick()
                    print("Left Click")
                elif canRightClick(landmarks):
                    rightClick()
                    print("Right Click")
                elif canDrag(landmarks):
                    drag(landmarks[20][1], landmarks[20][2])
                    print("Drag")
                elif canScrollUp(landmarks):
                    scrollUp()
                    print("Scroll Up")
                elif canScrollDown(landmarks):
                    scrollDown()
                    print("Scroll Down")
                startTime = time.time()
            elif landmarks and canMoveMouse(landmarks):
                moveMouse(landmarks[8][1], landmarks[8][2])
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
if __name__ == '__main__':
    main()
    
print(pag.size())