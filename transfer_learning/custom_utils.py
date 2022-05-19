from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mne

class CustomUtils:
    @staticmethod
    def get_data_from_edf(data,run_nb,p_frames,p_classes,channels=[1,5]):
        """
            data: un fichier de données importé avec mne
            p_frames: liste de listes pour chaque classe
            channels: les éléctrodes choisies
            ==> remplit la liste p_frames.
        """
        raw_data = data.get_data()
        n = 0
        ################################## what do the labels mean
        labels = []
        if(run_nb in [3, 4, 7, 8, 11, 12]):
            labels = {'T0':0,'T1':1,'T2':2}
        else:
            labels = {'T0':0,'T1':3,'T2':4}

        ################################## 
        onsets,events,durations = [],[],[]

        for a in data.annotations:
            onsets.append(a['onset']*160)
            durations.append(a['duration']*160)
            events.append(a['description'])

        ################################## for every frame in the file

        for event_nb in range(len(events)):

            wstart = int(onsets[event_nb])  
            wstop = int(onsets[event_nb]+640)
            #print(wstart,wstop)

            frame = np.zeros((len(channels),640),dtype=np.float32)

            for i in range(len(channels)):
                frame[i,0:640]= raw_data[channels[i],wstart:wstop]
            
            frame = frame.T 
            theclass = labels[events[event_nb]]
            
            p_frames.append(frame)
            p_classes.append(theclass-1)

    @staticmethod
    def get_data_from_all_edf(filenames_list):

        bframes,bclasses = [],[]
        for file_name in filenames_list: 

            data = mne.io.read_raw_edf(file_name,verbose=False)
            run_nb = int(file_name[-6:-4])
            session_nb = int(file_name[-9:-7])
            
            if run_nb in [4,6,8,10,12,14]:
                CustomUtils.get_data_from_edf(data,run_nb,bframes,bclasses)
                
        return bframes,bclasses

    @staticmethod
    def get_data_from_csv(filename,frames,classes,channels,SAMP_RATE,FRAME_WIDTH,FRAME_STEP,classes_names=None):
        '''
            filename (path of the file to get data from)
            frames (list to store frames in)
            classes (list to store classes in)
            channels (list of channels to keep)
        '''
        
        data = pd.read_csv(filename)[channels]  
        
        if(classes_names==None):
            classes_names = {'Avancer':0,'tourner_droite':1,'tourner_gauche':2,'Rien':3}
        
        frame_start = 1*SAMP_RATE   # sample to start in
        frame_end = frame_start + FRAME_WIDTH

        while(frame_end < data.shape[0]):
            
            frame = np.array(data[frame_start:frame_end])
            frames.append(frame)
            class_name = filename.split("/")[-2]

            classes.append(classes_names[class_name])
            
            frame_start += FRAME_STEP
            frame_end = frame_start + FRAME_WIDTH

    @staticmethod
    def viz(frame,ymin,ymax):
        t = [i for i in range(len(frame))]
        for channel in frame:
            plt.plot(channel)

        plt.ylim(ymin,ymax)
        plt.show()

    @staticmethod
    def scale(x): # reshape to scale, then put back in original form
        reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x_scaled = minmax_scale(reshaped_x, axis=1)
        return x_scaled.reshape(x_scaled.shape[0], int(x_scaled.shape[1]/2), 2).astype(np.float64)