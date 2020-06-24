from tkinter import *  
from PIL import ImageTk,Image  
from tkinter import filedialog
import tensorflow.keras as keras
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

window = Tk()  
window.title("Doggi")
window.iconphoto(False, PhotoImage(file=r"C:\Users\gabri\Desktop\Studia_VI_semestr\BIAI\DogRecognition_v2\reksio.png"))
global img       
global filename
canvas = Canvas(window, width = 300, height = 300)  
canvas.pack()  


global text
text = StringVar()

path = os.getcwd()
print(path)
label = Label( window, textvariable = text )
label.pack()
#############################################################################
myImg_HEIGHT = 150
myImg_WIDTH = 150
path = r"C:\Users\gabri\Desktop\Studia_VI_semestr\BIAI\DogRecognition_v2\dogs"
train_dir = os.path.join(path, 'train')
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                   )


train_data_gen = image_gen_train.flow_from_directory(batch_size=64,
                                                          directory=train_dir,
                                                          shuffle=True,
                                                          target_size=(myImg_HEIGHT, myImg_WIDTH),
                                                          class_mode='categorical')

filenames = train_data_gen.filenames
global labels
labels = []
for filename in filenames:
    newLabel = filename.split('\\')[0]
    isNew = True
    for label in labels:
        if label == newLabel:
            isNew = False
            break
    if isNew:
        labels.append(newLabel)
        
myImg_HEIGHT = 150
myImg_WIDTH = 150
pathf = r"C:\Users\gabri\Desktop\Studia_VI_semestr\BIAI\DogRecognition_v2\dog1.jpg"
global model
model = tf.keras.models.load_model('model')

batch_holder = np.zeros((1, myImg_HEIGHT, myImg_WIDTH, 3))


###########################################################################
def loadPicture():
    global filename
    window.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    global img
    img = Image.open(window.filename)
    img = ImageTk.PhotoImage(img.resize((300, 300), Image.ANTIALIAS))    
    canvas.create_image(20, 20, anchor=NW, image=img) 
    
    myImg = image.load_img(window.filename, target_size=(myImg_HEIGHT,myImg_WIDTH))
    myImg = image.img_to_array(myImg)
    myImg = np.expand_dims(myImg, axis=0)
    myImg = myImg.reshape(1, myImg_WIDTH, myImg_HEIGHT, 3)   
    result = model.predict(myImg, batch_size=1)
    index = 0
    for i in result:
        for j in i:
            if j == 1:
                break
            index += 1
    global text
    text.set(labels[index])

button = Button(window, text='Wybierz zdjęcie', width=25, command=loadPicture) 
button.pack()

window.mainloop() 