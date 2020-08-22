# Facial Patches Segmentation 

## Overview

This framework will enable you to detect facial parts separately. facial parts including: eyes, eyebrows, nose, mouth and jawline.


## Dependencies

``` 
Install imutils
Install dlib library : link => https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/
Shape Predictor File : shape_predictor_68_face_landmarks.dat file 
video file :vk.mp4
Install Opencv (cv2) package
Install numpy package
Install pickle package
Shape Predictor File Download Link : https://drive.google.com/file/d/17ULgxYgx9lVpu_SW8qyBoEyjw7qFhG8j/view?usp=sharing
```

## Usage
```
<!--extract : COVID Project.zip-->
<!--Navigate to the extracted folder-->
<!--On Commad Line Terminal -->
----------------------------------------------------------------------------------
<!--Optional Section - For Virtual Environment-->
export WORKON_HOME=~/.virtualenvs
VIRTUALENVWRAPPER_PYTHON='/usr/bin/python3'
source /usr/local/bin/virtualenvwrapper.sh
source env/bin/activate
----------------------------------------------------------------------------------
[COMMAND 1] python3 code.py shape_predictor_68_face_landmarks.dat vk.mp4
[COMMAND 2] python3 main_file parts_file 
```
## Results

RUN [COMMAND 1] To get pickle files : main_file and parts_file
RUN [COMMAND 2] To visualise the videos 


#### Credits:
Adrian Rosebrock - Face Detection
Marques Brownlee - Video Credits
