# voice-cmd

Project based on TensroFlow micro_speech example. Written for Arduino Nano 33 BLE Sense. The microcontroller acts 
as a peripheral device and after starting it begins to wait until a central device, for example the "nRF Connect" 
smartphone app, is connected to it. When connected, the microcontroller begins inference with a specified period 
and in case of voice command detection notifies central device via VoiceCmdService. BLE functionality is written 
using mbed OS and its API's.

## Model

In this project currently is used model trained with TensorFlow training script with "on, off"
words. You can follow their examples to make your own architecture or use own dataset.

## TODO

**_Nothing_**
