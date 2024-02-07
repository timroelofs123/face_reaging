import gradio as gr
import torch
import argparse

import sys
sys.path.append(".")

from model.models import UNet
from scripts.test_functions import process_image, process_video

# default settings
window_size = 512
stride = 256
steps = 18
frame_count = 0

def run(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet_model = UNet().to(device)
    unet_model.load_state_dict(torch.load(model_path, map_location=device))
    unet_model.eval()

    def block_img(image, source_age, target_age):
        return process_image(unet_model, image, video=False, source_age=source_age,
                             target_age=target_age, window_size=window_size, stride=stride)

    def block_img_vid(image, source_age):
        return process_image(unet_model, image, video=True, source_age=source_age,
                             target_age=0, window_size=window_size, stride=stride, steps=steps)

    def block_vid(video_path, source_age, target_age):
        return process_video(unet_model, video_path, source_age, target_age,
                             window_size=window_size, stride=stride, frame_count=frame_count)

    demo_img = gr.Interface(
        fn=block_img,
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
        description="Input an image of a person and age them from the source age to the target age."
    )

    demo_img_vid = gr.Interface(
        fn=block_img_vid,
        inputs=[
            gr.Image(type="pil"),
            gr.Slider(10, 90, value=20, step=1, label="Current age", info="Choose your current age"),
        ],
        outputs=gr.Video(),
        examples=[
            ['assets/gradio_example_images/1.png', 20],
            ['assets/gradio_example_images/2.png', 75],
            ['assets/gradio_example_images/3.png', 30],
            ['assets/gradio_example_images/4.png', 22],
            ['assets/gradio_example_images/5.png', 28],
            ['assets/gradio_example_images/6.png', 35]
        ],
        description="Input an image of a person and a video will be returned of the person at different ages."
    )

    demo_vid = gr.Interface(
        fn=block_vid,
        inputs=[
            gr.Video(),
            gr.Slider(10, 90, value=20, step=1, label="Current age", info="Choose your current age"),
            gr.Slider(10, 90, value=80, step=1, label="Target age", info="Choose the age you want to become")
        ],
        outputs=gr.Video(),
        examples=[
            ['assets/gradio_example_images/orig.mp4', 35, 60],
        ],
        description="Input a video of a person, and it will be aged frame-by-frame."
    )

    demo = gr.TabbedInterface([demo_img, demo_img_vid, demo_vid],
                              tab_names=['Image inference demo', 'Image animation demo', 'Video inference demo'],
                              title="Face Re-Aging Demo",
                              )

    demo.launch()


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Testing script - Image demo")
    parser.add_argument("--model_path", type=str, default="best_unet_model.pth", help="Path to the model")

    # Parse command-line arguments
    args = parser.parse_args()

    run(args.model_path)
