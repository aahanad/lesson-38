import cv2, os
name=input("please enter your name")
path="C:/Users/aahan/OneDrive - The Overlake School/Open cv/datasets"
if not os.path.isdir(path+"/"+name):
    os.mkdir(path+"/"+ name)
facexml="lesson31-main\haarcascade_frontalface_default.xml"
face=cv2.CascadeClassifier(facexml)
webcam=cv2.VideoCapture(0)
count=0
while count < 30:
    ret,img=webcam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.5,4)
    print (faces)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),3)
        person=gray[y:y+h,x:x+w]
        person_resize=cv2.resize(person,(130,100))
        cv2.imwrite(path+"/"+name+"/"+str(count)+".png",person_resize)
    count=count+1
    cv2.imshow("screen",img)

    k=cv2.waitKey(10)
    if k==27:
        break
