
class sc_filter:
  @classmethod
  def __init__(self):
    import matplotlib.pyplot as plt
    self.plt = plt

    import matplotlib as mpl
    self.mpl = mpl
    
    self.mpl.rcParams['figure.figsize'] = (10,10)
    self.mpl.rcParams['axes.grid'] = False

    import numpy as np
    self.np = np

    from PIL import Image
    self.Image = Image

    import time
    self.time = time

    import functools
    self.functools = functools

    import tensorflow as tf
    self.tf = tf


    from tensorflow.python.keras.preprocessing import image as kp_image
    self.kp_image = kp_image

    from tensorflow.python.keras import models
    self.models = models 
     
    from tensorflow.python.keras import losses
    self.losses = losses

    from tensorflow.python.keras import layers
    self.layers = layers

    from tensorflow.python.keras import backend as K
    self.K = K

    import IPython.display
    self.IPython_display = IPython.display

# """Content layer where will pull our feature maps"""
    self.content_layers = ['block5_conv2'] 

# """Style layer we are interested in"""
    self.style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1'
                    ]

    self.num_content_layers = len(self.content_layers)
    self.num_style_layers = len(self.style_layers)

  @classmethod
  def load_img(self,path_to_img):
    #maximum image size is set to be 512*512
    max_dim = 512
    img = self.Image.open(path_to_img)

    #scaling the image by dividing it by the maximum value of size 
    long = max(img.size) 
    scale = max_dim/long
    #resizing the image and converting to an array format
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), self.Image.ANTIALIAS)# """downsampling filter""" 
    img = self.kp_image.img_to_array(img)
    
    #We need to broadcast the image array such that it has a batch dimension, hence we add one dimension
    img = self.np.expand_dims(img, axis=0)
    return img

  @classmethod
  def imshow(self,img, title=None):
    #Remove the batch dimension, therefore remove one dimension by using squeeze
    out = self.np.squeeze(img, axis=0)

    #Display and plotting the image
    out = out.astype('uint8')
    self.plt.imshow(out)
    if title is not None:
      self.plt.title(title)
      self.plt.imshow(out)

  @classmethod
  def load_and_process_img(self,path_to_img):
    #We do preprocessing as expected for a VGG training process
    #Input: A floating point numpy.array or a tf.Tensor, 3D or 4D with 3 color channels, with values in the range [0, 255].
    img = sc_filter.load_img(path_to_img)
    img = self.tf.keras.applications.vgg19.preprocess_input(img)
    return img
    #Output : The images are converted from RGB to BGR, 
    #then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.

  @classmethod
  def deprocess_img(self,processed_img):
    #input to deprocess image must be an image of dimension [1, height, width, channel] or [height, width, channel]
    x = processed_img.copy()
    if len(x.shape) == 4:
      x = self.np.squeeze(x, 0)
    assert len(x.shape) == 3
    if len(x.shape) != 3:
      raise ValueError("Invalid input to deprocessing image")
    
    # """perform the inverse of the preprocessing step"""
    # """As our optimized image may take its values from −∞ and  ∞, so we must clip to maintain our values within 0-255 range"""
    # """VGG networks are trained on images with each channel normalized by mean [103.939, 116.779, 123.68] and the channels BGR"""
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = self.np.clip(x, 0, 255).astype('uint8')
    return x
  
  @classmethod
  def get_model(self):

    vgg = self.tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    # """Get output layers corresponding to style and content layers """
    style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
    content_outputs = [vgg.get_layer(name).output for name in self.content_layers]
    model_outputs = style_outputs + content_outputs

    # """Build model""" 
    return self.models.Model(vgg.input, model_outputs)

  @classmethod
  def get_content_loss(self,base_content, target):
    return self.tf.reduce_mean(self.tf.square(base_content - target))
  
  @classmethod
  def gram_matrix(self,input_tensor):
    # We make the image channels first 
    channels = int(input_tensor.shape[-1])
    a = self.tf.reshape(input_tensor, [-1, channels])
    n = self.tf.shape(a)[0]
    gram = self.tf.matmul(a, a, transpose_a=True)
    return gram / self.tf.cast(n, self.tf.float32)

  @classmethod
  def get_style_loss(self,base_style, gram_target):
    # """Expects two images of dimension h, w, c"""
    # height, width, num filters of each layer
    # We scale the loss at a given layer by the size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = sc_filter.gram_matrix(base_style)
    
    return self.tf.reduce_mean(self.tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)

  @classmethod
  def get_feature_representations(self,model, content_path, style_path):

    # Load our images in 
    content_image = sc_filter.load_and_process_img(content_path)
    style_image = sc_filter.load_and_process_img(style_path)
    
    # batch compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)
    
    
    # Get the style and content feature representations from our model  
    style_features = [style_layer[0] for style_layer in style_outputs[:self.num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[self.num_style_layers:]]
    return style_features, content_features

  @classmethod
  def compute_loss(self,model, loss_weights, init_image, gram_style_features, content_features):

    style_weight, content_weight = loss_weights
    
    # Feed our init image through our model. This will give us the content and 
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_image)
    
    style_output_features = model_outputs[:self.num_style_layers]
    content_output_features = model_outputs[self.num_style_layers:]
    
    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(self.num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
      style_score += weight_per_style_layer * sc_filter.get_style_loss(comb_style[0], target_style)
      
    # Accumulate content losses from all layers 
    weight_per_content_layer = 1.0 / float(self.num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
      content_score += weight_per_content_layer* sc_filter.get_content_loss(comb_content[0], target_content)
    
    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score 
    return loss, style_score, content_score
  
  @classmethod
  def compute_grads(self,cfg):
    with self.tf.GradientTape() as tape: 
      all_loss = sc_filter.compute_loss(**cfg)
    # Compute gradients wrt input image
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


  @classmethod
  def run_style_transfer(self,content_path, 
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3, 
                       style_weight=1e-2): 
    # We don't need to (or want to) train any layers of our model, so we set their
    # trainable to false. 
    model = sc_filter.get_model() 
    for layer in model.layers:
      layer.trainable = False
    
    # Get the style and content feature representations (from our specified intermediate layers) 
    style_features, content_features = sc_filter.get_feature_representations(model, content_path, style_path)
    gram_style_features = [sc_filter.gram_matrix(style_feature) for style_feature in style_features]
    
    # Set initial image
    init_image = sc_filter.load_and_process_img(content_path)
    init_image = self.tf.Variable(init_image, dtype=self.tf.float32)
    # Create our optimizer
    opt = self.tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1,name = 'Adam')

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
    start_time = self.time.time()
    global_start = self.time.time()
    
    norm_means = self.np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means   
    
    imgs = []
    for i in range(num_iterations):
      grads, all_loss = sc_filter.compute_grads(cfg)
      loss, style_score, content_score = all_loss
      opt.apply_gradients([(grads, init_image)])
      clipped = self.tf.clip_by_value(init_image, min_vals, max_vals)
      init_image.assign(clipped)
      end_time = self.time.time() 
      
      if loss < best_loss:
        # Update best loss and best image from total loss. 
        best_loss = loss
        best_img = sc_filter.deprocess_img(init_image.numpy())

      if i % display_interval== 0:
        start_time = self.time.time()
        
        # Use the .numpy() method to get the concrete numpy array
        plot_img = init_image.numpy()
        plot_img = sc_filter.deprocess_img(plot_img)
        imgs.append(plot_img)
        self.IPython_display.clear_output(wait=True)
        self.IPython_display.display_png(self.Image.fromarray(plot_img))
        print('Iteration: {}'.format(i))        
        print('Total loss: {:.4e}, ' 
              'style loss: {:.4e}, '
              'content loss: {:.4e}, '
              'time: {:.4f}s'.format(loss, style_score, content_score, self.time.time() - start_time))
    print('Total time: {:.4f}s'.format(self.time.time() - global_start))
    self.IPython_display.clear_output(wait=True)
    self.plt.figure(figsize=(14,4))
    for i,img in enumerate(imgs):
        self.plt.subplot(num_rows,num_cols,i+1)
        self.plt.imshow(img)
        self.plt.xticks([])
        self.plt.yticks([])
        
    return best_img, best_loss

  @classmethod
  def show_results(self,best_img, content_path, style_path, show_large_final=True):
  
    # """ deprocessing the output image in order to remove the processing that was applied to it before"""
    self.plt.figure(figsize=(10, 5))
    content = sc_filter.load_img(content_path) 
    style = sc_filter.load_img(style_path)

    self.plt.subplot(1, 2, 1)
    sc_filter.imshow(content, 'Content Image')

    self.plt.subplot(1, 2, 2)
    sc_filter.imshow(style, 'Style Image')

    if show_large_final: 
      self.plt.figure(figsize=(10, 10))

      self.plt.imshow(best_img)
      self.plt.title('Output Image')
      self.plt.show()
