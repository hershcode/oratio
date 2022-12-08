import os
import numpy as np
import tensorflow as tf
import malaya_speech
import noisereduce as nr
import librosa 
import numpy as np
import pickle 
import yaml
from google.cloud import aiplatform 

def load_config_file():
    
    print(os.getcwd())
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config 

class Oratio:
    
    def __init__(self):
        self.config = load_config_file() 
        self.load_model()
    
    def prepare_data(self, filename):

        sr=16000
        vad = malaya_speech.vad.webrtc()
        samples, sample_rate = librosa.load(filename, sr=16000)
        samples = nr.reduce_noise(y=samples, sr=16000, stationary=True)
        y_ = malaya_speech.resample(samples, sr, 16000)
        y_ = malaya_speech.astype.float_to_int(y_)
        frames = malaya_speech.generator.frames(samples, 30, sr)
        frames_ = list(malaya_speech.generator.frames(y_, 30, 16000, append_ending_trail=False))
        frames_webrtc = [(frames[no], vad(frame)) for no, frame in enumerate(frames_)]
        y_ = malaya_speech.combine.without_silent(frames_webrtc)
        zero = np.zeros(((sr+4000)-y_.shape[0]))
        signal = np.concatenate((y_, zero))
    
        return signal
    
    def extract_mfcc(self, array):

        mfcc_feat = librosa.feature.mfcc(y=array, sr=16000, n_mfcc=13)
        mfccs = np.array([mfcc_feat.flatten()])

        return mfccs 
    
    def predict(self, filename):
        
        signal = self.prepare_data(filename=filename)
        mfcc_input = self.extract_mfcc(array=signal)
        prediction = self.model.predict([mfcc_input])
        if self.le:
            index = np.argmax(prediction[0])
            prediction = self.le.inverse_transform([index])[0]
        else:
            prediction = prediction.predictions[0][0]

        return prediction 


    def load_model(self):

        if self.config['model_path'][-2:]:
            self.model = tf.keras.models.load_model('models/oratio_model.h5')
            self.le = pickle.load(open("models/le.pkl",'rb'))
        else:
            self.model = aiplatform.Endpoint(
                endpoint_name=self.config['model_path']
            )
            print("here")
            print(self.model)
            self.le = None 
