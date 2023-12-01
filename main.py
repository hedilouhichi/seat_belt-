# This is a sample Python script.
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from keras.models import load_model
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk
from datetime import datetime
import time
# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# Load the saved model
model = load_model('C:/Users/ASUS/Downloads/my_model.h5')
# Define the camera object
cap = cv2.VideoCapture(0)

# Define the GUI window
root = Tk()
root.title("Belt no Belt  Classifier")

# Define the label to display the predicted posture
posture_label = Label(root, text="", font=("Arial", 30))
posture_label.pack(pady=20)


def predict_posture(start_time,x):
    # Capture the image
    ret, bgr_img = cap.read()
    label_position = (bgr_img.shape[1]-150, bgr_img.shape[0]-50) # bottom right corner
    cv2.putText(bgr_img, x, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('image', bgr_img)
    # Preprocess the image
    # dsize
    start_time1 = time.time()
    if ((((int)(start_time1)-(int)(start_time)) % 5 == 0) ):
        print(start_time1+1)
        dsize = (100,100)
        #resize image
        resized_image = cv2.resize(bgr_img,dsize)


        from tensorflow.keras.preprocessing.image import  img_to_array

        # copnvert to array
        input_arr1 = img_to_array(resized_image)
        # convert as a batch for the model
        input_arr1 = np.array([input_arr1])  # Convert single image to a batch.
        # and finally predict
        prediction = model.predict(input_arr1)
        prediction=np.squeeze(np.where(prediction == np.max(np.squeeze(prediction)), 1, 0))
        print(prediction)
        if prediction[0]==1:
            posture_name="belt"
        else:
            posture_name="no belt"

        print(posture_name)
        # Add label to the image


        return posture_name
    if x == "":
        return ""
    return x



# Run the GUI window
# Call the function to start predicting postures
start_time = time.time()
x=""
while True:
    x=predict_posture(start_time,x)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
