import cv2,time, pandas
from datetime import datetime

first_frame=None
status_list=[0,0]
times = [None,None]
df=pandas.DataFrame(columns=["Start","End"])

video=cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status=0 #no motion
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame=gray
        continue
#I will use methode absdiff to make an absolute difference between the first_frame(which is to be still) and all
#following frame in order to see if something changed(movement in the range)
    delta_frame=cv2.absdiff(first_frame,gray)
#with the threshold binary method (returns a tuple) i transform the pixels calculated from difference
#every frame with difference value higher than 30 will become white
    threshold_frame=cv2.threshold(delta_frame, 30, 255,cv2.THRESH_BINARY)[1]
#in order to smoothen the image we go through a dilate process
    threshold_frame=cv2.dilate(threshold_frame, None, iterations=2)

#we find contours of the moving object and place it in a tuple
    (cnts,_) = cv2.findContours(threshold_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# we will use only relevant portion of the contour
    for contour in cnts:
        if cv2.contourArea(contour) > 15000:
            continue
        status+=1 #movement has been found
    #we create a tuple with four coordinates that we later assign to a rectangle
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),3)
    status_list.append(status)#all the movements are stored in a status_list

    status_list=status_list[-2:]

    if status_list[-1]>=1 and status_list[-2]==0:
        try:
            times.append(datetime.now())
        except (TypeError):
            pass
    if status_list[-1]==0 and status_list[-2]>=1:
        try:
            times.append(datetime.now())
        except (TypeError):
            pass
#this ComputerVision method opens the window giving title(first value) using input(second value)
    cv2.imshow("Capturing",gray)
    cv2.imshow('difference',delta_frame)
    cv2.imshow("Threshold Frame",threshold_frame)
    cv2.imshow('Color Frame',frame)


    key= cv2.waitKey(1)
    if key == ord('q'):
        if status>=1:
            times.append(datetime.now())
        break


print(status_list)
print (times)

for i in range(0,len(times),2):#iterate with a step of 2 to create a csv file
    df = df.append({"Start":[times[i],status],"End":[times[i+1],status]},ignore_index=True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()
