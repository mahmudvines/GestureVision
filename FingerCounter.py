import cv2
import time
import os
import HandTrackingModule as htm

# Camera resolution
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)  # use 0 for default webcam, try 1 if you have multiple
cap.set(3, wCam)
cap.set(4, hCam)

# Load overlay images
folderPath = "FingerImages"
myList = os.listdir(folderPath)
print("Overlay files:", myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(os.path.join(folderPath, imPath))
    overlayList.append(image)

print("Loaded overlays:", len(overlayList))

pTime = 0
detector = htm.handDetector(detectionCon=0.75, trackCon=0.75)

# Landmark indices for finger tips
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success or img is None:
        print("⚠️ Failed to grab frame from camera")
        continue

    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)

    totalFingers = 0
    if len(lmList) != 0:
        fingers = []

        # Thumb logic (basic right-hand orientation)
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for i in range(1, 5):
            if lmList[tipIds[i]][2] < lmList[tipIds[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        print("Detected fingers:", totalFingers)

        # Overlay image safely resized
        if 1 <= totalFingers <= len(overlayList):
            overlay = overlayList[totalFingers - 1]
            if overlay is not None:
                overlay_resized = cv2.resize(overlay, (200, 200))  # resize to fit
                h, w = overlay_resized.shape[:2]
                img[0:h, 0:w] = overlay_resized

        # Draw rectangle + number
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375),
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Finger Counter", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()