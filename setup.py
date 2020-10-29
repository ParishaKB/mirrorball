
import setuptools
from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='mirrorball',
      version='0.1.2',
      description='A python package to use neural networks to combine the content of an image with the style of another image',
      url='https://github.com/ParishaKB/MirrorballPackage',
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
          "matplotlib>=3.3.2"
          "tensorflow>=2.2.0",
          "numpy>=1.19.1",
          "pandas>=1.0.4",
          "Image>=8.0.0",
          "keras>=2.3.1",
          "time>=1.2.0",
          "functools>=3.5.1",
          "importlib-resources>=3.2.0"
      ]
      )
