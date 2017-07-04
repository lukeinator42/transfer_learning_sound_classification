import csv
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import time

count=-1
with open('metadata/UrbanSound8K.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        count+=1

        if count == 0:
            continue

        if not os.path.exists('spectrograms/' + row[7]):
            os.makedirs('spectrograms/' + row[7])

        y, sr = librosa.load("audio/fold" + str(row[5])+ "/" + str(row[0]))

        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

        log_S = librosa.logamplitude(S, ref_power=np.max)

        fig = plt.figure(figsize=(12,4))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

        plt.savefig('spectrograms/' + row[7] + '/' + row[0] + '.png')
        plt.close()

        #print count
        print '{0}\r'.format(count),
