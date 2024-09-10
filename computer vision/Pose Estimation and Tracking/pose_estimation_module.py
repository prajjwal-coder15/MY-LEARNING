import cv2 
import mediapipe 
import time


class PoseDetector:
    def __init__(self,mode = False, upBody=False, smooth=True, 
                detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth 
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mediapipe.solutions.drawing_utils
        self.mpPose = mediapipe.solutions.pose
        self.pose = self.mpPose.Pose( static_image_mode=self.mode,
                                        model_complexity=1,
                                        smooth_landmarks=self.smooth,
                                        enable_segmentation=False,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon
                                    )



    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw: 
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks)
        return img


   
    def getPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy),  2, (255,0,0), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture('1.mp4')
    pTime = 0
    detector = PoseDetector()
    while True:
        success,img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img, draw = False)
        print(lmList[14])
        cv2.circle(img, (lmList[14][1],lmList[14][2]), 5, (0, 0, 255), cv2.FILLED)


        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv2.imshow("chomu",img)
        if cv2.waitKey(1) == ord("q"):
            break



if __name__ == "__main__":
    main()        