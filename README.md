# Semantic Segmentation for Traffic Scene Understanding
 
**General:**
<br>
This repo contains a comparison between 3 encoder-decoder networks (FCNN, SegNet and UNet) for image segmentation using traffic scenes from the CamVid dataset. The challenge lies in categorizing each pixel within a traffic scene image to one of 32 categories. The encoder is pre-trained using VGG16 weights. Training and test metrics such as binary cross entropy loss, IoU and pixel accuracy are computed and averaged across an entire epoch.

<p align="center">
  <img width="300" height="300" src=/figs/unet_segmentation.jpg?raw=true "Training Environment">
	<figcaption>Fig.1 - Segmentation demo.</figcaption>
</p>

**Training:**
<br>
To train and replicate results open up terminal and cd into the semantic-segmentation-traffic-scenes folder using : ```cd semantic-segmentation-traffic-scenes/```, then call the respective ```modelname_train.py``` from terminal, for example as followed: ```python unet_train.py```. 3 models are supported and can be trained by calling ```modelname_train.py```, that is ```fcnn_train.py``` , ```segnet_train.py``` and ```unet_train.py```. Periodical updates regarding training and test are printed in console.

**Results:**
<br>
To obtain evaluation results and corresponding figures such as loss, IoU, pixel accuarcy and predictions on images from the test set, run ```python plots.py``` from terminal. Figures will be saved in the ```/figs``` folder and should look as followed:


<p align="center">
  <img width="700" height="300" src=/figs/test_vs_test_acc.png?raw=true "Training Environment">
	<figcaption>Fig.2 - Test and training accuracy as a function of epochs.</figcaption>
</p>

<p align="center">
  <img width="700" height="300" src=/figs/test_vs_test_iou.png?raw=true "Training Environment">
	<figcaption>Fig.3 - Model comparison: Test and training intersection over unions as a function of epochs.</figcaption>
</p>


<p align="center">
  <img width="700" height="300" src=/figs/test_vs_test_loss.png?raw=true "Training Environment">
	<figcaption>Fig.4 - Model comparison: Test and training loss as a function of epochs.</figcaption>
</p>


