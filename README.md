# Covid 19 Alert system

This Alert system monitors and  counts number of people passing by or standing in an area. It also alert with the sound when camera detects more than maximum number of people defined. This project can be one of the solutions to fight Covid-19 by maintaining social distancing, avoiding people gatherings and can be useful for Authorities to monitor areas from a Surveillance system.

See project - [![](http://img.youtube.com/vi/bEtI82LdjTo/0.jpg)](http://www.youtube.com/watch?v=bEtI82LdjTo "Covid-19 Alert System using NVIDIA Jetson Nano")

This system is based on __NVIDIA Jetson Nano__ and uses Computer vision, Deep Neural Networks to detect humans. Here, pre- trained COCO model *ssdlite_mobilenet_v2_coco* is used.
## Requirements

NVIDIA Jetson nano

JetPack 4.3

OpenCV 4.1.0

Tensorflow 2.0.0


## Files inside this Repository

1- _ssdlite_mobilenet_v2_coco_2018_05_09_   - pre-trained tensorflow model

2- _Beep-mp3.mp3_ -  Alert sound file

3- _people_counter_alert.py_ - Alert system code
