# Human-Background Segmentation

https://github.com/rohit-krish/Background-Segmentation/assets/100354219/2bec9057-2e55-4f72-8aae-7b65b21bf141

## Realtime background segmentation deep-learning model.

The process is segmenting a person(or persons) in an image from its background. Here we use the concept of semantic segmentation to predict the label of every pixel (dense prediction) in an image. This technique is widely used in computer vision applications like background replacement, or other applications.

**The project is created with idea for background replacement service for video streaming applications.**

Here we limit ourselves to binary classes (person or background) and use only plain portrait-selfie images to make the model more accurate we can do transfer-learning approch on the trained model.

We experimented with the following architectures for implementing a real-time portrait segmentation on web-servers.
- DeepLabV3
- SegNet
- SegResNet
- Unet

There are of course exist other architectures we haven't tried yet.

__And Found DeepLabV3 model more accurate than other architectures__

## Dataset
- https://www.kaggle.com/datasets/rohit369/human-background-segmentation

## Trained Models
| Model                   | Link                                                                                  |
|-------------------------|---------------------------------------------------------------------------------------|
| DeeplabV3               | https://drive.google.com/file/d/1--MKJToJsM2YLjG2nz7BRgkyQ0lMYJzi/view?usp=share_link |
| SegNet(ResNet backbone) | https://drive.google.com/file/d/1pR0O9ZZyEP-_aUSR_MiJn-2gMVZ9bqhj/view?usp=share_link |
| U-Net                   | https://drive.google.com/file/d/1lOZoCspE6soV73Oadini_KEF8lYL-LD9/view?usp=share_link |


## Dependencies

| Libraries      | Version  |
|--------------- |--------- |
| torch          | 2.0.0    |
| torchvision    | 0.15.1   |
| albumentations | 1.3.0    |
| opencv-python  | 4.7.0.72 |
| numpy          | 1.23.5   |
| matplotlib     | 3.6.3    |
| Pillo          | 9.4.0    |
| tqdm           | 4.65.0   |


# convert to tflite

# how to use?
