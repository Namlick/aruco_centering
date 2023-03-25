import cv2
import imutils

vs = cv2.VideoCapture(0)

while True: 
    ret,frame = vs.read()
    frame = imutils.resize(frame, 500)
  
    cv2.putText(frame, Test, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
  
    cv2.imshow("Frame", frame)
  
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop
