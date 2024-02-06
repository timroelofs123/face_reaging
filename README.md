# Re-aging of faces
Re-aging is increasingly used in the film industry and commercial applications like TikTok and FaceApp. 
This repo presents an open-source re-aging method.


## Method
This repo replicates the Face Re-Aging Network (FRAN) architecture presented in Disney Research's 
["Production-Ready Face Re-Aging for Visual Effects"](https://studios.disneyresearch.com/2022/11/30/production-ready-face-re-aging-for-visual-effects/) paper, 
which is a relatively simple U-Net-based architecture.

Following the aforementioned Disney Research paper, the dataset used for training consists of artificially re-aged images from [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset/) generated using [SAM](https://yuval-alaluf.github.io/SAM/). 
However, compared to SAM, this model's output preserves the identity of the pictured person much better, and is more robust. 

## Pre-trained model
The trained model can be downloaded from HuggingFace **LINK**. This model can be tested with the two Gradio demos. 

In the first demo, one can input an image with a source age (age of the person pictured) and a target age. This demo can be accessed on HuggingFace here **LINK**, 
or be run locally by downloading the model and running the `scripts/gradio_demo_img.py` script.

In the second demo, one does not have to specify a target age: Instead, a video will be shown where we cycle through the target age between 10 - 95. 
This demo can be accessed on HuggingFace here **LINK**, 
or be run locally by downloading the model and running the `scripts/gradio_demo_vid.py` script.


## Model re-training
The training script is available in this repo, together with the training parameters used.
In order to train the model from scratch using the available files, one would need to put the training data in `data/processed/train`. 
The training dataset should consist of folders where each folder contains images of one person, where the filename indicates the age, 
e.g. `person1/10.jpg` is _person1_ at age 10 and `person1/20.jpg` is the same person at age 20.


