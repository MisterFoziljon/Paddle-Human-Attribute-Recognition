from Detector import HumanDetector
from AttributeRecognizer import HumanAttributeRecognizer
import cv2
import time

human_model = HumanDetector("models/human/")
atribut_model = HumanAttributeRecognizer("models/high_precision/")

video = cv2.VideoCapture("video.mp4")

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

#out = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (width//2,height//2))
FPS = 0
while video.isOpened():
    
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width//2,height//2), interpolation = cv2.INTER_LINEAR)
        
    people_boxes = human_model([frame])
    
    for person_box in people_boxes:
        xmin, ymin, xmax, ymax = person_box
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (17, 212, 105), 3)
        
        human = frame[ymin:ymax,xmin:xmax]
        result = atribut_model([human])[0]

        y = 15
        for key, value in result.items():
            cv2.putText(frame, f"{key}:{value}", (xmin+5,ymin+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y += 15
        
    
    
    
    cv2.putText(frame, f"FPS: {fps}", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
    #out.write(frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
#out.release()
cv2.destroyAllWindows() 