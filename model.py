import tensorflow as tf 
import numpy as np 
# import os
import cv2
# import glob

class Classifier():

    def load_graph(self, loc):
        with tf.gfile.GFile(loc, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name=self.name)
        return graph 
    
    def __init__(self, loc, name=None):
        self.label = ['male', 'female']
        self.name = name
        self.graph = self.load_graph(loc)
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.output = self.graph.get_tensor_by_name(self.name + '/Classifier_1/Softmax:0')
        self.input = self.graph.get_tensor_by_name(self.name + '/input_C:0') #(?,96,96,1)

    def predict(self, x):
        output = self.sess.run(self.output, feed_dict={self.input: x})
        output = np.argmax(output, axis=1)
        return output

def get_data(s):
    ids = s.split(',')
    data = []
    for id in ids:
        path = './faces/parent-{}.jpg'.format(id)
        data.append(cv2.imread(path, 0))

    data = np.array(data).reshape(-1,96,96,1)/255
    return data



# model = Classifier('./model/bknet.pb', 'SA')

# paths = glob.glob('faces/*')
# imgs = []
# for path in paths:
#     imgs.append(cv2.imread(path, 0))

# x = np.load('male_50.npy')
# out = model.predict(x)
# print(out)
# print(out.shape)
# print(np.sum(out==np.ones(50)))