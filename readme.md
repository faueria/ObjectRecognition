# Object Recognition

## info
This was created by Nihal Sandadi
I used Visual Studio Community Edition 2022 with c++ 17 and Windows 11
I am using 2 travel days

## How to run:
you need a webcam which is pointed downwards to a white background and some example objects. 
C++ 17 and resnet18-v2-7.onnx
Before you run, make sure to add open cv to your project and make sure you have resnet18-v2-7.onnx, modify the file paths in the code to match your own file system.
build and run the file without any additional arguments.

## some basic controls:
g - Grayscale thresholding

c - Custom color thresholding

m - Toggle morphological cleaning

r - Toggle region analysis

f - Toggle feature computation

t - Toggle training/classification mode

q - Quit program

### training objects
Press t to enter training mode
Place object in camera view
Press n and enter label (e.g., "wrench")
Repeat for multiple orientations
Press s to save training data

### to see the classification
Exit training mode (t)
Present objects to camera
View classification results:
Top: classic features (4D)
Bottom: CNN embeddings (150K+ features)

## Customization
3 thresholding options:
K-means sampling fraction
HSV channel weights
Blur parameters