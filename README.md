# Semantic image segmentation for traffic scene understanding
 
**General:**
<br>
This repo contains a comparison between 3 encoder-decoder networks (FCNN, SegNet and UNet) for image segmentation using traffic scenes from the CamVid dataset. The challenge lies in categorizing each pixel within a traffic scene image to one of 32 categories. The encoder is pre-trained using VGG16 weights. Training and test metrics such as binary cross entropy loss, IoU and pixel accuracy are computed and averaged across an entire epoch.

**Training:**
<br>
To train and replicate results open up terminal and cd into the semantic-segmentation-traffic-scenes folder using : ```cd semantic-segmentation-traffic-scenes/```, then call the respective ```modelname_train.py``` from terminal, for example as followed: ```python unet_train.py```. Currently 3 models can be trained by calling ```modelname_train.py```, that is ```fcnn_train.py``` , ```segnet_train.py``` and ```unet_train.py``` Periodical updates regarding training and test progress should appear printed in the console each epoch.

**Results:**
<br>
To obtain evaluation results and corresponding figures such as loss, IoU, pixel accuarcy and predictions on images from the test set, run ```python plots.py``` from terminal. Figures will be saved in the ```/figs``` folder.
