from os.path import curdir

import cv2
import numpy as np
import face_recognition
import os
from datetime import  datetime

path = 'imagesAttendance'
images = []
classNames = []
myList =os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb
        encode = face_recognition.face_encodings(img)[0] #find the encodings
        encodList.append(encode)
    return encodList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList =f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime("%d/%m/%Y %H:%M:%S")
            f.write(f'\n{name},{dtstring}')




encodeListKnown = findEncodings(images)
print('Encoding Complete.')

cap =cv2.VideoCapture(0) # has our id

while True: # to get each fream at real time
    success, img = cap.read()
    imgs = cv2.resize(img, (0,0), None, 0.25, 0.25) # we will reduce the size of our image to speed the process, cause this all happen at realtime
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facesCurFrame= face_recognition.face_locations(imgs)[0] #we might get multiple faces for that, we have to get the location, and send to the encoding function
    encodeCurFrame = face_recognition.face_encodings(imgs, facesCurFrame)[0]

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame): #one by one it will grab the face location and the encode face from the face current frame and encode current fream respectively, zip: we need them in the same loop
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
           # print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y2*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0,255, 0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            markAttendance(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)





# faceloc = face_recognition.face_locations(imgelon)[0] ## only sending the first element of this
# encodelon = face_recognition.face_encodings(imgelon)[0]
# cv2.rectangle(imgelon, (faceloc[3], faceloc[0]),(faceloc[1], faceloc[2]),(255,0,255),2)
#
# facelocTest = face_recognition.face_locations(imgTest)[0] ## only sending the first element of this
# encodTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest, (facelocTest[3], facelocTest[0]),(facelocTest[1], facelocTest[2]),(255,0,255),2)
#
# results = face_recognition.compare_faces([encodelon], encodTest)
# faceDis = face_recognition.face_distance([encodelon], encodTest)
