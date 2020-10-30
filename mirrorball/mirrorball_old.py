# Import and configure modules
"""

#importing library

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import numpy as np
from PIL import Image
import time
import functools

import tensorflow as tf

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

"""# Visualize the input

Funtion to load image from the given path/website
"""

def load_img(path_to_img):
  #maximum image size is set to be 512*512
  max_dim = 512 
  img = Image.open(path_to_img)

  #scaling the image by dividing it by the maximum value of size 
  long = max(img.size) 
  scale = max_dim/long
  #resizing the image and converting to an array format
  img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS) #downsampling filter
  img = kp_image.img_to_array(img)
  
  #We need to broadcast the image array such that it has a batch dimension, hence we add one dimension
  img = np.expand_dims(img, axis=0)
  return img

"""Function to print the image from the variable into which it is loaded"""

def imshow(img, title=None):
  #Remove the batch dimension, therefore remove one dimension by using squeeze
  out = np.squeeze(img, axis=0)

  #Display and plotting the image
  out = out.astype('uint8')
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out)

"""#Preprocess and Deprocess

Function to preprocess using vgg19 preprocess input layer
"""

def load_and_process_img(path_to_img):
  #We do preprocessing as expected for a VGG training process
  #Input: A floating point numpy.array or a tf.Tensor, 3D or 4D with 3 color channels, with values in the range [0, 255].
  img = load_img(path_to_img)
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img
  #Output : The images are converted from RGB to BGR, 
  #then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.

def deprocess_img(processed_img):
  #input to deprocess image must be an image of dimension [1, height, width, channel] or [height, width, channel]
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  #perform the inverse of the preprocessing step
  #As our optimized image may take its values from −∞ and  ∞, so we must clip to maintain our values within 0-255 range
  #VGG networks are trained on images with each channel normalized by mean [103.939, 116.779, 123.68] and the channels BGR
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x

"""# Define content and style representations

To get the content and style from the respective images, we use the mid-layers of the vgg19 model.


The deeper we dive in the neural network layer, the higher order of features are extracted. 


For example, the first layer might extract the horizontal and vertical lines present. 


Vgg19 is pretrained to classify images


The middle layers are necessary to define the representation of content and style from our images. 


For an input image, we will try to match the corresponding style and content target representations at these intermediate layers. 


#### Why intermediate layers?

For a neural network to perform image classification, it must understand the image.


So we take a raw image as input pixels and build an internal representation through transformations that turn the raw image pixels into a complex understanding of the features present within the image. 


Thus, somewhere between where the raw image is fed in and the classification label is output, the model serves as a complex feature extractor; hence by accessing middle layers, we’re able to describe the content and style of input images.
"""

#Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

#Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

"""##Function to create our model with access to intermediate layers. 
  
  This function will load the VGG19 model and access the intermediate layers. 
  These layers will then be used to create a new model that will take input image
  and return the outputs from these intermediate layers from the VGG model. 
  
Returns:
Returns a keras model that takes image inputs and outputs the style and content intermediate layers.

Here we load our pretrained image classification network, then we grab the layers of interest as we said in the previous snippet. 
  After this we define a models by setting the models input to an image and the outputs to the outputs style and content layers,
  so we the input as the image and the output is the content and style intermediate layers 

Load our model. We load pretrained VGG, trained on imagenet data
"""

def get_model():

  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  #Get output layers corresponding to style and content layers
  style_outputs = [vgg.get_layer(name).output for name in style_layers]
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs

  #Build model
  return models.Model(vgg.input, model_outputs)

"""We’ll pass both the desired content image and our base input image to the neural network

This will return the intermediate layer outputs (from the layers defined above) from our model. 

Then we simply take the euclidean distance between the two intermediate representations of those images.
"""

def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
  # We make the image channels first 
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
  #Expects two images of dimension h, w, c
  # height, width, num filters of each layer
  # We scale the loss at a given layer by the size of the feature map and the number of filters
  height, width, channels = base_style.get_shape().as_list()
  gram_style = gram_matrix(base_style)
  
  return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)

"""##Function to load and preprocess both the content and style images from their path. 

Then it will feed them through the network to obtain the outputs of the intermediate layers. 
  
Input:
    
model: The model that we are using.

content_path: The path to the content image.

style_path: The path to the style image
    
Output: 

returns the style features and the content features.
"""

def get_feature_representations(model, content_path, style_path):

  # Load our images in 
  content_image = load_and_process_img(content_path)
  style_image = load_and_process_img(style_path)
  
  # batch compute content and style features
  style_outputs = model(style_image)
  content_outputs = model(content_image)
  
  
  # Get the style and content feature representations from our model  
  style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
  content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
  return style_features, content_features

"""## Function to compute the loss total loss.
  
Input: 

model: The model that will give us access to the intermediate layers

loss_weights: The weights of each contribution of each loss function. 
      (style weight, content weight, and total variation weight)

init_image: Our initial base image. This image is what we are updating with 
      our optimization process. We apply the gradients wrt the loss we are 
      calculating to this image.

gram_style_features: Precomputed gram matrices corresponding to the 
      defined style layers of interest.
    
content_features: Precomputed outputs from defined content layers of 
      interest.
      
Output:

returns the total loss, style loss, content loss, and total variational loss
"""

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):

  style_weight, content_weight = loss_weights
  
  # Feed our init image through our model. This will give us the content and 
  # style representations at our desired layers. Since we're using eager
  # our model is callable just like any other function!
  model_outputs = model(init_image)
  
  style_output_features = model_outputs[:num_style_layers]
  content_output_features = model_outputs[num_style_layers:]
  
  style_score = 0
  content_score = 0

  # Accumulate style losses from all layers
  # Here, we equally weight each contribution of each loss layer
  weight_per_style_layer = 1.0 / float(num_style_layers)
  for target_style, comb_style in zip(gram_style_features, style_output_features):
    style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
  # Accumulate content losses from all layers 
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features, content_output_features):
    content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
  
  style_score *= style_weight
  content_score *= content_weight

  # Get total loss
  loss = style_score + content_score 
  return loss, style_score, content_score

"""### Function to compute the gradients on the input image"""

def compute_grads(cfg):
  with tf.GradientTape() as tape: 
    all_loss = compute_loss(**cfg)
  # Compute gradients wrt input image
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss

"""### Final function that combines all the function to run model and print output at every 100 iterations"""

import IPython.display

def run_style_transfer(content_path, 
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3, 
                       style_weight=1e-2): 
  # We don't need to (or want to) train any layers of our model, so we set their
  # trainable to false. 
  model = get_model() 
  for layer in model.layers:
    layer.trainable = False
  
  # Get the style and content feature representations (from our specified intermediate layers) 
  style_features, content_features = get_feature_representations(model, content_path, style_path)
  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
  
  # Set initial image
  init_image = load_and_process_img(content_path)
  init_image = tf.Variable(init_image, dtype=tf.float32)
  # Create our optimizer
  opt = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1,name = 'Adam')

  # For displaying intermediate images 
  iter_count = 1
  
  # Store our best result
  best_loss, best_img = float('inf'), None
  
  # Create a nice config 
  loss_weights = (style_weight, content_weight)
  cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
  }
    
  # For displaying
  num_rows = 2
  num_cols = 5
  display_interval = num_iterations/(num_rows*num_cols)
  start_time = time.time()
  global_start = time.time()
  
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means   
  
  imgs = []
  for i in range(num_iterations):
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    end_time = time.time() 
    
    if loss < best_loss:
      # Update best loss and best image from total loss. 
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())

    if i % display_interval== 0:
      start_time = time.time()
      
      # Use the .numpy() method to get the concrete numpy array
      plot_img = init_image.numpy()
      plot_img = deprocess_img(plot_img)
      imgs.append(plot_img)
      IPython.display.clear_output(wait=True)
      IPython.display.display_png(Image.fromarray(plot_img))
     
  IPython.display.clear_output(wait=True)
  plt.figure(figsize=(14,4))
  for i,img in enumerate(imgs):
      plt.subplot(num_rows,num_cols,i+1)
      plt.imshow(img)
      plt.xticks([])
      plt.yticks([])
      
  return best_img, best_loss

"""###Function to print the content and style images and final output image"""

def show_results(best_img, content_path, style_path, show_large_final=True):
  
  """ deprocessing the output image in order to remove the processing that was applied to it before"""
  plt.figure(figsize=(10, 5))
  content = load_img(content_path) 
  style = load_img(style_path)

  plt.subplot(1, 2, 1)
  imshow(content, 'Content Image')

  plt.subplot(1, 2, 2)
  imshow(style, 'Style Image')

  if show_large_final: 
    plt.figure(figsize=(10, 10))

    plt.imshow(best_img)
    plt.title('Output Image')
    plt.show()
