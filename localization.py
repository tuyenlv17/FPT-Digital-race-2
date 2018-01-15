
# coding: utf-8

# In[33]:



# coding: utf-8

# In[ ]:


from __future__ import division


# In[ ]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
import cv2
import os
from skimage import io as ski_io
from skimage.viewer import ImageViewer
from skimage import color as ski_color
from skimage import io as ski_io
import time
from nolearn.lasagne import BatchIterator
from skimage import transform as ski_transform
NUM_CLASS=10
def one_hot(labels, n_class = NUM_CLASS):
    total_record = len(labels)
    y = np.zeros((total_record, n_class))
    y[np.arange(total_record), labels] = 1
    return y


from collections import namedtuple

Parameters = namedtuple('Parameters', [
        # Data parameters
        'num_classes', 'image_size', 
        # Training parameters
        'batch_size', 'max_epochs', 'log_epoch', 'print_epoch',
        # Optimisations
        'learning_rate_decay', 'learning_rate',
        'l2_reg_enabled', 'l2_lambda', 
        'early_stopping_enabled', 'early_stopping_patience', 
        'resume_training', 
        # Layers architecture
        'conv1_k', 'conv1_d', 'conv1_p', 
        'conv2_k', 'conv2_d', 'conv2_p', 
        'conv3_k', 'conv3_d', 'conv3_p', 
        'fc4_size', 'fc4_p'
    ])

import os

class Paths(object):
    """
    Provides easy access to common paths we use for persisting 
    the data associated with model training.
    """
    
    def __init__(self, params):
        """
        Initialises a new `Paths` instance and creates corresponding folders if needed.

        Parameters
        ----------
        params  : Parameters
                  Structure (`namedtuple`) containing model parameters.
        """
        self.model_name = self.get_model_name(params)
        self.var_scope = self.get_variables_scope(params)
        self.root_path = os.getcwd() + "/models/" + self.model_name + "/"
        self.model_path = self.get_model_path()
        self.train_history_path = self.get_train_history_path()
        self.learning_curves_path = self.get_learning_curves_path()
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)

    def get_model_name(self, params):
        """
        Generates a model name with some of the crucial model parameters encoded into the name.

        Parameters
        ----------
        params  : Parameters
                  Structure (`namedtuple`) containing model parameters.
                  
        Returns
        -------
        Model name.
        """
        # We will encode model settings in its name: architecture, optimisations applied, etc.
        print ("call get model name")
        model_name = "k{}d{}p{}_k{}d{}p{}_k{}d{}p{}_fc{}p{}".format(
            params.conv1_k, params.conv1_d, params.conv1_p, 
            params.conv2_k, params.conv2_d, params.conv2_p, 
            params.conv3_k, params.conv3_d, params.conv3_p, 
            params.fc4_size, params.fc4_p
        )
        model_name += "_lrdec" if params.learning_rate_decay else "_no-lrdec"
        model_name += "_l2" if params.l2_reg_enabled else "_no-l2"
        return model_name

    def get_variables_scope(self, params):
        """
        Generates a model variable scope with some of the crucial model parameters encoded.

        Parameters
        ----------
        params  : Parameters
                  Structure (`namedtuple`) containing model parameters.
                  
        Returns
        -------
        Variables scope name.
        """
        # We will encode model settings in its name: architecture, optimisations applied, etc.
        var_scope = "k{}d{}_k{}d{}_k{}d{}_fc{}_fc0".format(
            params.conv1_k, params.conv1_d,
            params.conv2_k, params.conv2_d,
            params.conv3_k, params.conv3_d, 
            params.fc4_size
        )
        return var_scope

    def get_model_path(self):
        """
        Generates path to the model file.
   
        Returns
        -------
        Model file path.
        """
        return self.root_path + "model.ckpt"

    def get_train_history_path(self):
        """
        Generates path to the train history file.
   
        Returns
        -------
        Train history file path.
        """
        return self.root_path + "train_history"
    
    def get_learning_curves_path(self):
        """
        Generates path to the learning curves graph file.
   
        Returns
        -------
        Learning curves file path.
        """
        return self.root_path + "learning_curves.png"


# In[ ]:


def fully_connected(input, size):
    """
    Performs a single fully connected layer pass, e.g. returns `input * weights + bias`.
    """
    weights = tf.get_variable( 'weights', 
        shape = [input.get_shape()[1], size],
        initializer = tf.contrib.layers.xavier_initializer()
      )
    biases = tf.get_variable( 'biases',
        shape = [size],
        initializer = tf.constant_initializer(0.0)
      )
    return tf.matmul(input, weights) + biases

def fully_connected_relu(input, size):
    return tf.nn.relu(fully_connected(input, size))

def conv_relu(input, kernel_size, depth):
    """
    Performs a single convolution layer pass.
    """
    weights = tf.get_variable( 'weights', 
        shape = [kernel_size, kernel_size, input.get_shape()[3], depth],
        initializer = tf.contrib.layers.xavier_initializer()
      )
    biases = tf.get_variable( 'biases',
        shape = [depth],
        initializer = tf.constant_initializer(0.0)
      )
    conv = tf.nn.conv2d(input, weights,
        strides = [1, 1, 1, 1], padding = 'SAME')
    return tf.nn.relu(conv + biases)

def pool(input, size):
    """
    Performs a max pooling layer pass.
    """
    return tf.nn.max_pool(
        input, 
        ksize = [1, size, size, 1], 
        strides = [1, size, size, 1], 
        padding = 'SAME'
    )

def model_pass(input, params, is_training):
    # Convolutions

    with tf.variable_scope('conv1'):
        conv1 = conv_relu(input, kernel_size = params.conv1_k, depth = params.conv1_d) 
    with tf.variable_scope('pool1'): 
        pool1 = pool(conv1, size = 2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob = params.conv1_p), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size = params.conv2_k, depth = params.conv2_d)
    with tf.variable_scope('pool2'):
        pool2 = pool(conv2, size = 2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob = params.conv2_p), lambda: pool2)
    with tf.variable_scope('conv3'):
        conv3 = conv_relu(pool2, kernel_size = params.conv3_k, depth = params.conv3_d)
    with tf.variable_scope('pool3'):
        pool3 = pool(conv3, size = 2)
        pool3 = tf.cond(is_training, lambda: tf.nn.dropout(pool3, keep_prob = params.conv3_p), lambda: pool3)
    
    # Fully connected
    
    # 1st stage output
    pool1 = pool(pool1, size = 4)
    shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])
    
    # 2nd stage output
    pool2 = pool(pool2, size = 2)
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])    
    
    # 3rd stage output
    shape = pool3.get_shape().as_list()
    pool3 = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])
    
    flattened = tf.concat([pool1, pool2, pool3], 1)
    
    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(flattened, size = params.fc4_size)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob = params.fc4_p), lambda: fc4)
    with tf.variable_scope('out'):
        logits = fully_connected(fc4, size = params.num_classes)
    return logits


parameters = Parameters(
    # Data parameters
    num_classes = NUM_CLASS,
    image_size = (32, 32),
    # Training parameters
    batch_size = 256,
    max_epochs = 100,
    log_epoch = 1,
    print_epoch = 1,
    # Optimisations
    learning_rate_decay = False,
    learning_rate = 0.0001,
    l2_reg_enabled = True,
    l2_lambda = 0.0001,
    early_stopping_enabled = True,
    early_stopping_patience = 100,
    resume_training = True,
    # Layers architecture
    conv1_k = 5, conv1_d = 32, conv1_p = 0.9,
    conv2_k = 5, conv2_d = 64, conv2_p = 0.8,
    conv3_k = 5, conv3_d = 128, conv3_p = 0.7,
    fc4_size = 1024, fc4_p = 0.5
)


def get_top_k_predictions(params, X, k = 5):
    
    # Initialisation routines: generate variable scope, create logger, note start time.
    paths = Paths(params)
    
    # Build the graph
    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
        tf_x = tf.placeholder(tf.float32, shape = (None, params.image_size[0], params.image_size[1], 1))
        is_training = tf.constant(False)
        with tf.variable_scope(paths.var_scope):
            predictions = tf.nn.softmax(model_pass(tf_x, params, is_training))
            top_k_predictions = tf.nn.top_k(predictions, k)

    with tf.Session(graph = graph) as session:
        session.run(tf.global_variables_initializer())
        tf.train.Saver().restore(session, paths.model_path)
        [p] = session.run([top_k_predictions], feed_dict = {
                tf_x : X
            }
        )
        return np.array(p)


# In[34]:


import cv2
import numpy as np


# In[35]:


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


# In[36]:


map_class = ['stop', 'turn_left', 'turn_right', 'prob_left', 'prob_right', 'no_entry', 'speed_limit', 'min_speed', 'end_min', 'unknown']
map_answer = []


# In[38]:


import cv2
import numpy as np
cap = cv2.VideoCapture('./input-sample/MVI_1049.avi')

# Initialisation routines: generate variable scope, create logger, note start time.
paths = Paths(parameters)

# Build the graph
graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
    tf_x = tf.placeholder(tf.float32, shape = (None, parameters.image_size[0], parameters.image_size[1], 1))
    is_training = tf.constant(False)
    with tf.variable_scope(paths.var_scope):
        predictions = tf.nn.softmax(model_pass(tf_x, parameters, is_training))
        top_k_predictions = tf.nn.top_k(predictions, 5)

with tf.Session(graph = graph) as session:
    session.run(tf.global_variables_initializer())
    tf.train.Saver().restore(session, paths.model_path)        
    while (1):

        _, frame = cap.read()
#         frame = cv2.resize(frame, (640, 360), interpolation = cv2.INTER_LINEAR)

        blur = cv2.medianBlur(frame.copy(), 5)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([100, 70, 40])
        upper_blue = np.array([130, 255, 255])

        lower_red = np.array([0, 70, 50])
        upper_red = np.array([10, 255, 255]) 
        r_mask1 = cv2.inRange(hsv, lower_red, upper_red) 
        r_mask2 = cv2.inRange(hsv, np.array([100, 10, 170]), np.array([180, 230, 255])) 
        r_mask3 = cv2.inRange(hsv, np.array([157, 60, 11]), np.array([180, 255, 255])) 
        red_mask = r_mask1 | r_mask2 | r_mask3
        

        # Threshold the HSV image to get only blue colors
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = blue_mask + red_mask

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)
        mask = cv2.erode(mask.copy(), None, iterations=2)
        mask2 = cv2.dilate(mask.copy(), None, iterations=2)

        cnts = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        framedraw = frame.copy()
        if len(cnts) > 0:
            for cnt in cnts:
                x, y, w, h = cv2.boundingRect(cnt)
#                 Vi tri cac bien            
                if w < 20 or h < 20:
                    cv2.rectangle(framedraw, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    continue
                roi = frame[y:y+h, x:x+w]
                # extract image and feed to cnn
                roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)  
                roi = ski_transform.resize(roi, (32, 32))
                roi = ski_color.rgb2gray(roi)
                roi = roi.reshape(roi.shape + (1,))        
                tf_X = np.array([roi])
                cv2.rectangle(framedraw, (x, y), (x + w, y + h), (255,0,0), 2)
                [prediction] = session.run([top_k_predictions], feed_dict = {
                        tf_x : tf_X
                    }
                )
                prob_class =prediction[0][0][0] 
                class_ord = prediction[1][0][0]
                class_name = map_class[class_ord]
                cv2.putText(framedraw, class_name, (x+w, y + int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,color=(0,0,255))
                print (prob_class)
#         cv2.imshow('frame', framedraw)

#         k = cv2.waitKey(5) & 0xFF
#         if k == 27:
#             break

cv2.destroyAllWindows()

