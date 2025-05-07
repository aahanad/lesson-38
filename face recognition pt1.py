import cv2, os
import numpy as np
path="C:/Users/aahan/OneDrive - The Overlake School/Open cv/datasets"
(images,lables,names,id)=([],[],{},0)
for (subdirs,dirs,files)in os.walk(path):
    for subdir in dirs:
        names[id]=subdir
        img_path= os.path.join(path,subdir)
        for img in os.listdir(img_path):
            file=img_path+"/"+img
            label=id
            images.append(cv2.imread(file,0))
            lables.append(label)
        id=id+1
print(names)
(images,lables)=[np.array(lis) for lis in [images,lables]]
recogniser=cv2.face.LBPHFaceRecognizer_create()
recogniser.train(images,lables)
facexml="lesson31-main\haarcascade_frontalface_default.xml"
face=cv2.CascadeClassifier(facexml)
webcam=cv2.VideoCapture(0)

while True:
    ret,img=webcam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.5,4)
    print (faces)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),3)
        person=gray[y:y+h,x:x+w]
        person_resize=cv2.resize(person,(130,100))
        prediction=recogniser.predict(person_resize)
        print(prediction[1])
        if prediction[1]>60:
            cv2.putText(img,names[prediction[0]],(x-10,y-20),cv2.FONT_HERSHEY_PLAIN,1,(60,60,75))
    cv2.imshow("screen",img)

    k=cv2.waitKey(10)
    if k==27:
        break
    #Record the photos of family/friends
# 
