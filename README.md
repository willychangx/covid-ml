# COVID19 Classifier

## Google Colab
It is easiest to lauch this code using Google Colab. To do so, go to eg.ipynb and click "Open in Colab."

**NOTE:** It is recommended to use GPU when running (Runtime -> Change runtime type -> GPU).


## Recommendations
It is recommended to run this project using a virtual environment like Anaconda. Below are references to install it on your respective OS.

If you are running on your local machine, make sure to have the Covid, Healthy, and Others folder with images (freshly unzipped) in the same directory as the code.

***Windows***  
[Installing on Windows](https://docs.anaconda.com/anaconda/install/windows/)

***Linux***  
[Installing on Linux](https://docs.anaconda.com/anaconda/install/linux/)

***MacOS***  
[Installing on macOS](https://docs.anaconda.com/anaconda/install/mac-os/)

## Requirements
The project requires the following dependencies:
- numpy
- matplotlib
- torch, torchvision
- tqdm
- opencv

The project runs on Python 3.7.X.

You can check your dependencies by running:  
`python3 checkdeps.py`

To install, run:  
`pip install -r requirements.txt`

**NOTE:** You have to have pip3 and Python3. See below for instructions on how to install on Windows and Linux.

## Install Python 3.7.X
***Windows***  
Go to the [official Python page](https://www.python.org/downloads/) and download the appropriate version. Pip3 should come installed with it.

***Linux***  
Python 3 should be installed. If it isn't, run the following commands:  
`sudo apt-get update`  
`sudo apt-get install python3.7`

After you have Python 3, you can install pip/pip3 by running:  
`sudo apt install python3-pip`

## Source Code
