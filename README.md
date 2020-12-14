# Faster_RCNN
Real time Object Detection Framework (Implemented using PyTorch)

## What is Faster R-CNN?
Faster RCNN is a real time object detection framework which can detect bounding boxes for objects in the scene and classifies them according to the type of object present in the bounding box. 

Faster R-CNN has two networks: region proposal network (RPN) for generating region proposals and a Box Header network using these proposals to detect bounding boxes and classify objects. A simplified version of the RPN can be found in my repository: [Region_Proposal_Network](https://github.com/karanpandya12/Region_Proposal_Network.git). For the Faster RCNN, a ResNet-50 network with a Feature Pyramid Network was trained separately on the dataset and used as a backbone.

Lets take a look at a schematic representation of the network. <br>
<center>
  <img src = "/Images/faster_rcnn.jpeg">
</center>

## Dataset
A subset of the COCO dataset was used containing data of 3 classess namely, Vehicles, People and Animals. Here are some example images from the dataset: <br>
<img src = "/Images/dataset_1.png" height = 300> <img src = "/Images/dataset_2.png" height = 300>
<img src = "/Images/dataset_3.png" height = 300> <img src = "/Images/dataset_4.png" height = 300>

## Results
In the following images, the top 3 bounding boxes for each image are displayed after performing Non-Maximum Suppression. The different colors of the boxes correspond to different classes. <br>
- Red - Vehicle
- Blue - Person
- Green - Animal <br>
<img src = "/Images/result_1.png" height = 300> <img src = "/Images/result_2.png" height = 300>
<img src = "/Images/result_3.png" height = 300> <img src = "/Images/result_4.png" height = 300>
