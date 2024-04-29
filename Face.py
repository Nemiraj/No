from unicodedata import name
import cv2

# Initialize LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained model
recognizer.read('trainer/trainer.yml')

# Path to the Haar cascade
cascadePath = "C:\\Users\\91797\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\cv2\\data/haarcascade_frontalface_default.xml"

# Initialize the face cascade
faceCascade = cv2.CascadeClassifier(cascadePath)

# Define font type
font = cv2.FONT_HERSHEY_SIMPLEX

# Number of persons you want to Recognize
id = 10

# Names, leave first two empty because the counter starts from 0
names = ['','','nj','']

# Initialize the camera
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640) # set video FrameWidht
cam.set(4, 480) # set video FrameHeight

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read() # Read the frames
    
    # Convert image to grayscale
    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = faceCascade.detectMultiScale(converted_image,
                                          scaleFactor = 1.2,
                                          minNeighbors = 5,
                                          minSize = (int(minW), int(minH)))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Predict on every single image
        id, accuracy = recognizer.predict(converted_image[y:y+h, x:x+w])
        
        # Check if accuracy is less than 100 ==> "0" is a perfect match 
        if accuracy < 100:
            id = names[id]
            accuracy = "  {0}%".format(round(100 - accuracy))
        else:
            id = "unknown"
            accuracy = "  {0}%".format(round(100 - accuracy))
        
        cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(accuracy), (x+5, y+h-5), font, 1, (255, 255, 0), 1)  
    
    cv2.imshow('camera', img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Cleanup
print("Thanks for using this program, have a good day.")
cam.release()
cv2.destroyAllWindows()
