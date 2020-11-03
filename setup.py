
import setuptools
from setuptools import setup


def readme():
    with open('README.md',encoding="utf8") as f:
        return f.read()

setup(name='mirrorball',
      version='0.2.0',
      description='A python package to use neural networks to combine the content of an image with the style of another image',
      long_description= readme(),
      url='https://github.com/ParishaKB/mirrorball',
      author='Parisha Bhatia, Soham Sharangpani, Shreyansh Bardia, Ujwal Shah,Aniket Modi,Gaurav Ankalagi',
      license='MIT',
      include_package_data=True,
      package_data={
          'mirrorball': ['mirrorball']
      },
      packages=setuptools.find_packages(exclude=['tests']),
      zip_safe=False,
      install_requires=[ 
          "librosa>=0.8.0",
          "matplotlib>=3.2.2"
          "tensorflow>=2.3.0",
          "numpy>=1.18.5",
          "pandas>=1.1.3",
          "Image>=1.1",
          "keras>=2.4.3",
          "importlib-resources>=3.2.0"
      ]
      )
