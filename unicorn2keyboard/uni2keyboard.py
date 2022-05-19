import sys
# setting path
sys.path.append('./../')

import socket
import numpy as np
import tensorflow as tf
import keyboard
from deep_learning.custom_utils import CustomUtils

###############: udp packets reception
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

###############: load trained model
model = tf.keras.models.load_model("attempt1_model/")

###############: variables
frame = np.zeros((1,640,2))     # where the info is stored
keys= {0:'c',1:'q',2:'d',3:''}       # class to key binding

###############: main loop
while(True):
    print('prout')
    data, addr = sock.recvfrom(68)              # receive data
    np.roll(frame[0],-2)                        # roll matrix, and
    #frame[0,-1] = np.array([[data[1],data[3]]]) # put data at the top of the frame
    frame = CustomUtils.scale(frame)            # normalize
    prediction = model.predict(frame)[0]        # predict thought from frame
    move = np.argmax(prediction)                # get most probable class
                                                # press corresponding key
    if(move == 3):
        keyboard.release('c, q, d')
    else:
        keyboard.press_and_release(keys[move])      


###############