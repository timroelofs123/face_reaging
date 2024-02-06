import gradio as gr
import torch
import argparse

import sys
sys.path.append(".")

from model.models import UNet
from scripts.test_functions import process_image


def run(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet_model = UNet().to(device)
    unet_model.load_state_dict(torch.load(model_path, map_location=device))
    unet_model.eval()

    def block(image, source_age, target_age):
        return process_image(unet_model, image, video=False, source_age=source_age,
                             target_age=target_age, window_size=512, stride=256)

    demo = gr.Interface(
        fn=block,
        inputs=[
            gr.Image(type="pil"),
            gr.Slider(10, 90, value=20, step=1, label="Current age", info="Choose your current age"),
            gr.Slider(10, 90, value=80, step=1, label="Target age", info="Choose the age you want to become")
        ],
        outputs="image",
        examples=[
            ['assets/gradio_example_images/1.png', 20, 80],
            ['assets/gradio_example_images/2.png', 75, 40],
            ['assets/gradio_example_images/3.png', 30, 70],
            ['assets/gradio_example_images/4.png', 22, 60],
            ['assets/gradio_example_images/5.png', 28, 75],
            ['assets/gradio_example_images/6.png', 35, 15]
        ],
    )

    demo.launch()


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Testing script - Image demo")
    parser.add_argument("--model_path", type=str, default="best_unet_model.pth", help="Path to the model")

    # Parse command-line arguments
    args = parser.parse_args()

    run(args.model_path)
