This project shows how to approach localization probelm in order to solve detection problems

The backbone layer is resnet-101 in pytorch.

The red dots (🔴) are the actual points and the green dots(🟢) are the predicted ones.

![](https://github.com/achyut-srivastava/Projects/edit/master/Computer%20Vision/Object%20Localization/object localization.jpg)

Loss function which I have used is: L1 loss because it doesn't penalizes the error too much.

Optimizer: I have used SGD

The dataset is available here as well as on kaggle the link is here: https://www.kaggle.com/drgilermo/face-images-with-marked-landmark-points

Motivation: yolov4 and SSD object detection algorithm.
