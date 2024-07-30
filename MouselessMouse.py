import cv2
import time
import numpy as np
import HandTrackingModule as htm
import pyautogui as pag

frameW, frameH = 160, 120 # frame width and height
camW, camH = 640, 480 # frame height
screenW, screenH = pag.size() # screen width and height
smoothening = 5
curlocX, curlocY, prevlocX, prevlocY = [0]*4
# pag.FAILSAFE = False

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

def canMoveMouse(fingersUp): # moving index finger
    if fingersUp[1] and not any(fingersUp[2:]):
        return True
    return False
    
def moveMouse(x, y):
    # map values
    x2 = int(np.interp(x, (frameW, camW-frameW), (0, screenW)))
    y2 = int(np.interp(y, (frameH, camH-frameH), (0, screenH)))
    curlocX = prevlocX + (x2 - prevlocX) / smoothening
    curlocY = prevlocY + (y2 - prevlocY) / smoothening
    pag.moveTo(curlocX, curlocY, button='left')

def canLeftClick(fingersUp): # raising 2 fingers
    if fingersUp[1:3] == [True, True] and not any(fingersUp[3:]) and not fingersUp[0]:
        return True
    return False
    
def leftClick():
    pag.click(button='left')
    
def canRightClick(fingersUp): # rasing 3 fingers
    if fingersUp[1:4] == [True, True, True] and not fingersUp[0] and not fingersUp[4]:
        return True
    return False

def rightClick():
    pag.click(button='right')

def canDrag(fingersUp): # raising 4 fingers
    if all(fingersUp[1:]) and not fingersUp[0]:
        return True
    return False
    
def drag(x, y):
    # map values
    x2 = int(np.interp(x, (frameW, screenW-frameW), (0, screenW)))
    y2 = int(np.interp(y, (frameH, screenH-frameH), (0, screenH)))
    curlocX = prevlocX + (x2 - prevlocX) / smoothening
    curlocY = prevlocY + (y2 - prevlocY) / smoothening
    pag.dragTo(curlocX, curlocY, button='left')

def canScrollUp(fingersUp): # thumbs up
    if fingersUp[0] and not any(fingersUp[1:]):
        return True
    return False
    
def scrollUp():
    pag.scroll(20)
    
def canScrollDown(fingersUp): # all fingers down
    if not any(fingersUp):
        return True
    return False
    
def scrollDown():
    pag.scroll(-20)

def main():
    cap = cv2.VideoCapture(0)
    detector = htm.handDetector()
    startTime = time.time()
    while True:
        success, img = cap.read() # image size of (480, 640, 3)
        img = cv2.flip(img, 1)
        img = cv2.rectangle(img, (frameW, frameH), (camW-frameW, camH-frameH), (0, 255, 0), 2)
        img = detector.findHands(img)
        landmarks = detector.findPosition(img)
        
        if landmarks:
            if time.time() - startTime > 1:
                fingersUp = detector.findFingersUp(landmarks)
                if canLeftClick(fingersUp):
                    leftClick()
                    img = cv2.putText(img, "Left Click", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                elif canRightClick(fingersUp):
                    rightClick()
                    img = cv2.putText(img, "Right Click", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                # elif canDrag(fingersUp):
                #     drag(landmarks[20][1], landmarks[20][2])
                #     img = cv2.putText(img, "Drag", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                #     prevlocX, prevlocY = curlocX, curlocY
                # elif canScrollUp(fingersUp):
                #     scrollUp()
                #     img = cv2.putText(img, "Scroll Up", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                # elif canScrollDown(fingersUp):
                #     scrollDown()
                #     img = cv2.putText(img, "Scroll Down", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                startTime = time.time()
            elif canMoveMouse(fingersUp):
                moveMouse(landmarks[8][1], landmarks[8][2])
                prevlocX, prevlocY = curlocX, curlocY
        
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
if __name__ == '__main__':
    main()