## Installation

### Requirements:
- PyTorch >= 1.0. Installation instructions can be found in https://pytorch.org/get-started/locally/.
- torchvision==0.2.1
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- (optional) OpenCV for the webcam demo

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name SCAN python=3.7
conda activate SCAN

# this installs the right pip and dependencies for the fresh python
conda install ipython

# FCOS and coco api dependencies
pip install ninja yacs cython matplotlib tqdm

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
# More selections can be found at https://github.com/CityU-AIM-Group/SIGMA/blob/main/INSTALL.md
conda install -c pytorch pytorch=1.3.0 torchvision==0.2.1 cudatoolkit=9.0 (cudatoolkit 10.0, 10.1 and 11.0 are ok!)


export INSTALL_DIR=$PWD

# install pycocotools. Please make sure you have installed cython.
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/CityU-AIM-Group/SCAN.git
cd SCAN

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it

python setup.py build develop

# If you meet the python version problem, 
# pls check the scipy verison (we use 1.6.0),
# since the automatically installed 1.8+ version 
# may not support python 3.7.


unset INSTALL_DIR

