from sympy.abc import alpha
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from utilsfixed import score,detect_down,detect_up,in_hoop_region,clean_hoop_pos,clean_ball_pos,get_device


class ShotDetector:
    def __init__(self):
        #load the YOLO model
        self.overlay_text="waiting.."
        self.model=YOLO("D://repos//Basketball_App//BasketballAIApp//Trainings//kagglebest.pt")

        #Uncomment this line to accelerate inference .Note that this may cause  errors in some setups
        #self.model.half()

        self.class_names=["basketball","rim"]
        self.device=get_device()

        #uncomment line below to use webcam
        #self.cap=cv2.VideoCapture(0)

        #use video
        self.cap=cv2.VideoCapture("D://repos//Basketball_App//BasketballAIApp//clips//training7.mp4")

        self.ball_pos=[] #array of tuples ((x_pos,y_pos)),frame_count,width,height,conf)
        self.hoop_pos=[] #array of tuples ((x_pos,y_pos),frame_count,width,height,conf)

        self.frame_count=0
        self.frame=None

        self.makes=0
        self.attempts=0

        #used to detect shots (upper and lower region)
        self.up=False
        self.down=False
        self.up_frame=0
        self.down_frame=0

        #used for green and red colors after make/miss
        self.fade_frames=20
        self.fade_counter=0
        self.overlay_color=(0,0,0)

        self.run()
    def run(self):
        while(True):
            ret, self.frame = self.cap.read()
            if not ret:
                #end of video or an error occured
                print("Error")
                break
            results=self.model(self.frame,stream=True,device=self.device)
            print("okundu")
            for r in results:
                boxes=r.boxes
                for box in boxes:
                    #bounding box
                    x1,y1,x2,y2=box.xyxy[0]
                    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                    w,h=x2-x1,y2-y1

                    #confidence
                    conf=math.ceil((box.conf[0]*100))/100

                    #Class name
                    cls=int(box.cls[0])
                    current_class=self.class_names[cls]
                    center=(int(x1+w/2),int(y1+h/2))


                    #only create ball points if high confidence or near hoop
                    if(conf>0.3 or (in_hoop_region(center,self.hoop_pos) and conf>0.15)) and current_class=="basketball":
                        self.ball_pos.append((center,self.frame_count,w,h,conf))
                        cvzone.cornerRect(self.frame,(x1,y1 ,w,h))

                    #Create hoop points if high confidence
                    if(conf>0.5 and current_class=="rim"):
                        self.hoop_pos.append((center,self.frame_count,w,h,conf))
                        cvzone.cornerRect(self.frame,(x1,y1,w,h))


            self.clean_motion()
            self.shot_detection()
            self.display_score()
            self.frame_count+=1

            cv2.imshow("Frame",self.frame)

            #Close if "q" is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def clean_motion(self):
        #clean and display ball motion
        self.ball_pos=clean_ball_pos(self.ball_pos,self.frame_count)
        for i in range(0,len(self.ball_pos)):
            cv2.circle(self.frame,self.ball_pos[i][0],2,(0,0,255),2)

        #clean hoop motion and display current hoop center
        if len(self.hoop_pos)>1:
            self.hoop_pos=clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame,self.hoop_pos[-1][0],2,(128,128,0),2)

    def shot_detection(self):
        if len(self.hoop_pos)>0 and len(self.ball_pos)>0:
            #detecting when ball is in up and down area ball can only be in down area after it is in up
            if not self.up:
                self.up=detect_up(self.ball_pos,self.hoop_pos)
                if self.up:
                    self.up_frame=self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down=detect_down(self.ball_pos,self.hoop_pos)
                if self.down:
                    self.down_frame=self.ball_pos[-1][1]

            #if ball goes from up area to down area in that order,increase attemt and reset
            if self.frame_count%10==0:
                if self.up and self.down and self.up_frame<self.down_frame:
                    self.attempts+=1
                    self.up=False
                    self.down=False

                    #if it is a make , put a green screen and display "Score"
                    if score(self.ball_pos,self.hoop_pos):
                        self.makes+=1
                        self.overlay_color=(0,255,0) # green for score
                        self.overlay_text="Score"
                        self.fade_counter=self.fade_frames
                    else:
                        self.overlay_color=(255,0,0) #red for miss
                        self.overlay_text="Miss"
                        self.fade_counter=self.fade_frames
    def display_score(self):
        #add text
        text=str(self.makes)+"/"+str(self.attempts)
        cv2.putText(self.frame,text,(50,125),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),6)
        cv2.putText(self.frame,text,(50,125),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),3)

        #add overlay text for shot result if it exists
        if hasattr(self,"overlay_text"):
            #calculate text size to position it at the right top corner
            (text_width,text_height),_=cv2.getTextSize(self.overlay_text,cv2.FONT_HERSHEY_SIMPLEX,3,6)
            text_x=self.frame.shape[1]-text_width-40 #rigght alignment with some margin
            text_y=100 # top margin

            #display overlay text with color (overlay color)
            cv2.putText(self.frame,self.overlay_text,(text_x,text_y),cv2.FONT_HERSHEY_SIMPLEX,3,self.overlay_color,6)
            #cv2.putText(self.frame,self.overlay_text,(text_x,text_y),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),3)

        #gradually fade out color after shot
        if self.fade_counter>0:
            alpha=0.2*(self.fade_counter/self.fade_frames)
            self.frame=cv2.addWeighted(self.frame,1-alpha,np.full_like(self.frame,self.overlay_color),alpha,0)
            self.fade_counter-=1


if __name__=="__main__":
    ShotDetector()