import cv2
import numpy as np
import face_recognition

## step 1/ importing images
imgelon = face_recognition.load_image_file("imageBasic\Elon.png")
imgelon = cv2.cvtColor(imgelon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file("imageBasic\elonTest.png")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

## step2/ finding the faces in the image and finding their encoding/location
faceloc = face_recognition.face_locations(imgelon)[0] ## only sending the first element of this
encodelon = face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon, (faceloc[3], faceloc[0]),(faceloc[1], faceloc[2]),(255,0,255),2)

facelocTest = face_recognition.face_locations(imgTest)[0] ## only sending the first element of this
encodTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (facelocTest[3], facelocTest[0]),(facelocTest[1], facelocTest[2]),(255,0,255),2)

## step3 / finding the faces and the distance between them.

results = face_recognition.compare_faces([encodelon], encodTest)
faceDis = face_recognition.face_distance([encodelon], encodTest)
print(results)
print(faceDis)
cv2.putText(imgTest, f'{results}, {round(faceDis[0],2)}',(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2 ,)

cv2.imshow("Elon Musk", imgelon)
cv2.imshow("Elon Test", imgTest)
cv2.waitKey(0)



