import webrtcvad
import pyaudio
import wave
import os, sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import tensorflow as tf
from time import time

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

label_file = "retrained_labels.txt"
input_height = 224
input_width = 224
input_mean = 0
input_std = 255
input_layer = "input"
output_layer = "final_result"
graph = load_graph("retrained_graph.pb")

vad = webrtcvad.Vad()
vad.set_mode(2)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 32000
CHUNK = 960
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "file.wav"

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)



with tf.Session(graph=graph) as sess:
    # Feed the image_data as input to the graph and get first prediction

    while True:

        frames = []
        frameCount = 0

        while frameCount<5:
            data = stream.read(CHUNK)

            if vad.is_speech(data, RATE):
                frameCount+=1;
            else:
                frameCount = 0;
            # print frameCount


        print "recording..."
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print "finished recording"
        start = time()
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # change this as you see fit
        audio_path = 'file.wav'
        image_path = 'tmp/tmp.jpg'

        y, sr = librosa.load(audio_path)

        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.logamplitude(S, ref_power=np.max)

        # Make a new figure
        fig = plt.figure(figsize=(12,4))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

        # Make the figure layout compact

        #plt.show()
        plt.savefig(image_path)
        plt.close()

        t = read_tensor_from_image_file(image_path,
                                        input_height=input_height,
                                        input_width=input_width,
                                        input_mean=input_mean,
                                        input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name);
        output_operation = graph.get_operation_by_name(output_name);

        results = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]: t})

        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)

        #for i in top_k:
        print(labels[top_k[0]], results[top_k[0]])

        print "took " + str(time()-start)

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
