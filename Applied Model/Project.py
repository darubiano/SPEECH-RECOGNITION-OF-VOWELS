# -*- coding: utf-8 -*-
"""
Andrés Santiago Arias Páez
David Andrés Rubiano Venegas
"""
import os

# instalar librerias
os.system('pip install -r requirements.txt')
import pyaudio
import wave
from PIL import Image
import numpy as np
from python_speech_features import mfcc
from python_speech_features import logfbank
from sklearn.externals import joblib
import scipy.io.wavfile as wav
import scipy
"""
GRABAR AUDIO 
"""
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "Audios/vowel.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Record for 3 seconds")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("End")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print("Processing")

os.system('pre_processing_and_MFCC.exe')

f = open ('MFCCs.txt','r')
mfccs = f.read()
mfccs=mfccs.split(',')
f.close()

clf = joblib.load('Trained_Model_SVM_linear_25%_Test_75%_Train_MFCCs.pkl')
prediction = clf.predict([mfccs])

print('Predicted Vowel : ',prediction)

if prediction=='0':
    vocal_imagen = Image.open('Vowels\_A.png')
    vocal_imagen.show()
elif prediction=='1':
    vocal_imagen = Image.open('Vowels\_E.png')
    vocal_imagen.show()
elif prediction=='2':
    vocal_imagen = Image.open('Vowels\_I.png')
    vocal_imagen.show()
elif prediction=='3':
    vocal_imagen = Image.open('Vowels\_O.png')
    vocal_imagen.show()
else:
    vocal_imagen = Image.open("Vowels\_U.png")
    vocal_imagen.show()













