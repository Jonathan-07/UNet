# “Adapting the UNet Neural Network for Generation of Superresolution Biological Images”
## 4th year MPhys research project


I developed a machine learning tool for upscaling the resolution of biological images, thereby developing a less damaging and faster method of producing high resolution biological images. The UNet CNN was adapted via changes to model architecture and optimisation, repurposing it from a segmentation classifier to a superresolution model that provided a four-fold increase in resolution to high accuracy. Throughout the project I developed my deep neural network Python coding skills in PyTorch to a level required for academic research – giving me experience in end-to-end implementation of deep CNNs for a desired use specified by a brief.


### Dissertation abstract:
High resolution microscopy techniques are associated with increased photon damage, potentially damaging biological samples. This project developed a machine learning tool for upscaling the resolution of confocal spinning-disk images to match the detail found in ISM imaging, thereby developing a less damaging and faster method of producing high resolution biological images. The UNet convolutional neural network was adapted via changes to model architecture and optimisation to be repurposed from a state-of-the-art segmentation classifier to a superresolution model. A custom ‘Dataset’ class was also developed to split images in smaller tiles, allowing operation of the adapted UNet model on computers with limited GPU memory. The fully trained model provides a four-fold increase in resolution, corresponding to a doubling of both image dimensions. Images upscaled by the optimally trained model were found to have significant similarity to high resolution ISM images of the same samples, attaining accuracy of 93.9% according to a custom defined metric. Already pre-trained to high accuracy on the biological structures found in mouse kidneys, the model can be re-trained in approximately 7 hours in order to provide this high accuracy for other visually different biological data samples.

### Example of results
![Screenshot (183)](https://user-images.githubusercontent.com/57955969/124358076-88f3c380-dc16-11eb-911a-d29cb2bee65b.png)
From left to right: Confocal (lower res) image of mouse kidney sample --> superresolution fourfold resolution upscaled image
produced by adapted UNet of the same sample --> ISM (higher res) image of the same sample.
