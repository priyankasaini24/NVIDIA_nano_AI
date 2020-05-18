# Covid 19 Alert system

This Alert system monitors and  counts number of people passing by or standing in an area. It also Alert with the sound when camera detects more than maximum number of people defined. This project can be one of the solution to fight Covid-19 by maintaining social distancing, avoiding people gatherings and can be useful for Authorities to monitor areas as a Surveillance system.

See project - [![](http://img.youtube.com/vi/bEtI82LdjTo/0.jpg)](http://www.youtube.com/watch?v=bEtI82LdjTo "Covid-19 Alert System using NVIDIA Jetson Nano")

This system is based on __NVIDIA Jetson Nano__ and uses Computer vision, Deep Neural Networks to detect humans. Here, pre- trained COCO model *ssdlite_mobilenet_v2_coco* is used. You can use other models also from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) website.  

## Requirements

NVIDIA Jetson nano developer kit

JetPack 4.3
> Insall jetpack 4.3 from [JetPack 4.3 Developer](https://developer.nvidia.com/jetpack-4_3_DP)

OpenCV 4.1.0
> The pre installed OpenCV in Jetson Nano doesn't come with Gstreamer and CUDA.  So, I'd recommend to install Opencv + Gstramer from this website [OpenCV 4 + CUDA on Jetson Nano](https://www.jetsonhacks.com/2019/11/22/opencv-4-cuda-on-jetson-nano/).

Tensorflow 2.0.0
> Install tensorflow from [Download Tensorflow for Jetson nano](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)


## Files inside this Repository

1- _ssdlite_mobilenet_v2_coco_2018_05_09_   - pre-trained tensorflow model

2- _Beep-mp3.mp3_ -  Alert sound file

3- _people_counter_alert.py_ - Alert system code
