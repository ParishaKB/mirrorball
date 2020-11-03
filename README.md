# Mirrorball

Mirrorball is an artistic package which aims at combining two images using the power of neural networks. The package picks up the style from one image and adds it on the content layer, creating new images having the style and looks with primitive design of the primary content image. The package can combine a variety of images ranging from Portraits to Landscapes, and from natural paintings to modern art.  The package can also be used to check the authenticity of images, by checking the error while running the program. 
The functions that a user accesses here are: 

* run_style_transfer (content_path, 
                                    style_path, 
                                    num_iterations = 1000, 
                                    content_weight = 1000, 
                                    style_weight = 0.01)
                                    
content_path: Input the path of your primary/content image

style_path: Input the path of the image whose style you want to apply on the primary/content image

num_iterations: Number of iterations you want the model to run. Default value is 1000.

content_weight: How much originality of the primary/content image do you wish to retain. Default value is 1000

style_weight: How much of the style do you wish to take from the style image. Default value is 0.01

The function gives 2 outputs: best image, and least loss. You need to add 2 variables before running this function to store the outputs.

*	show_results (best_img, 
                           content_path, 
                           style_path, 
                           show_large_final=True)
                           
best_img: The best image formed by the neural network, as stored from the above function.

content_path: Input the path of your primary/content image

style_path: Input the path of the image whose style you want to apply on the primary/content image

show_large_final: To display the best_img. Default value is True.

*	return_image(best_img)

best_img: The best image formed by the neural network, as stored from the above function.

The function converts the numpy array to image and also saves it as a .jpg file, which the user can download for further use.

# Prerequisites

File path of image in .jpeg, .png, .jpg format. The image background shouldn’t be transparent. 

### Dependencies

The libraries required to function smoothly are all open source.
  
* Matplotlib: Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
  
* Tensorflow: TensorFlow is a Python library for fast numerical computing created and released by Google. It is a foundation library that can be used to create Deep Learning models directly or by using wrapper libraries that simplify the process built on top of TensorFlow
  
* Numpy: NumPy is a library for the Python programming language, adding support for large,            multi-dimensional arrays and matrices, along with a large collection of high-level mathematical  functions to operate on these arrays.
  
* Pandas: Pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.
  
* Image: Python Imaging Library is a free and open-source additional library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
  
* Keras: Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result as fast as possible is key to doing good research.

### Installation

Use the following commands for jupyter notebooks (if libraries already exists, need not be re-installed)

```sh
$ pip3 install matplotlib
```

```sh
$ pip3 install tensorflow 
```

```sh
$ pip3 install numpy
```

```sh
$ pip3 install pandas
```
```sh
$ pip3 install image
```
```sh
$pip3 install keras
```
```sh
$pip3 install mirrorball
```
  
Replace the $ pip3 by !pip for google colab users.
  
### Development

Developed by: "Parisha Bhatia, Soham Sharangpani, Shreyansh Bardia, Ujwal Shah, Aniket Modi, Gaurav Ankalagi"

# License
MIT

Visit  GitHub repository “mirrorball” for more details 
