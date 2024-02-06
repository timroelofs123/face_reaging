# Re-aging of faces
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>

This repo presents an open-source re-aging method.

Re-aging is increasingly used in the film industry and commercial applications like TikTok and FaceApp. With this, there is 
also an increased interest in the application of Generative AI to this use-case. Such a method is presented here, largely based on Disney Research's
["Production-Ready Face Re-Aging for Visual Effects"](https://studios.disneyresearch.com/2022/11/30/production-ready-face-re-aging-for-visual-effects/) paper, 
the model and code of which remain proprietary. 

The method only requires an image (or video frame) 
and the (approximate) age of the person to generate the same image of the person looking older or younger. 

<img src="assets/docs/ex4.gif" width="600">



## Method
This repo replicates the Face Re-Aging Network (FRAN) architecture from the aforementioned paper, 
which is a relatively simple U-Net-based architecture.

Following the paper, the dataset used for training consists of artificially re-aged images from [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset/) generated using [SAM](https://yuval-alaluf.github.io/SAM/). 
However, compared to SAM, this model's output preserves the identity of the pictured person much better, and is more robust. 
In the example below it is visible how, compared to SAM, the model is able to preserve the identity of the subject, details like the background, glass, and earring, while still providing realistic aging.

<table>
    <tr>
        <th>Input image</th>
        <th>Our model output</th>
        <th>SAM output</th>
    </tr>
    <tr>
        <td><img src="assets/docs/ex5_img.png" width="200"></td>
        <td><img src="assets/docs/ex5.gif" width="200"></td>
        <td><img src="assets/docs/sam_ex.gif" width="200"></td>
    </tr>
</table>


## Pre-trained model
The trained model can be downloaded from HuggingFace **LINK**. This model can be tested with the two Gradio demos. 

In the first demo, one can input an image with a source age (age of the person pictured) and a target age. This demo can be accessed on HuggingFace here **LINK**, 
or be run locally by downloading the model and running the `scripts/gradio_demo_img.py` script.

In the second demo, one does not have to specify a target age: Instead, a video will be shown where we cycle through the target age between 10 - 95. 
This demo can be accessed on HuggingFace here **LINK**, 
or be run locally by downloading the model and running the `scripts/gradio_demo_vid.py` script.

<table>
    <tr>
        <td><img src="assets/docs/ex1.gif" width="200"></td>
        <td><img src="assets/docs/ex2.gif" width="200"></td>
        <td><img src="assets/docs/ex3.gif" width="200"></td>
    </tr>
</table>


## Model re-training
The training script is available in this repo, together with the training parameters used.
In order to train the model from scratch using the available files, one would need to put the training data in `data/processed/train`. 
The training dataset should consist of folders where each folder contains images of one person, where the filename indicates the age, 
e.g. `person1/10.jpg` is _person1_ at age 10 and `person1/20.jpg` is the same person at age 20.


