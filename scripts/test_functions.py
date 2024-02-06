import face_recognition
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.io import write_video
import tempfile

mask_file = torch.from_numpy(np.array(Image.open('assets/mask1024.jpg').convert('L'))) / 255


def sliding_window_tensor(input_tensor, window_size, stride, your_model, mask=mask_file):
    """
    Apply aging operation on input tensor using a sliding-window method. This operation is done on the GPU, if available.
    """

    input_tensor = input_tensor.to(next(your_model.parameters()).device)
    mask = mask.to(next(your_model.parameters()).device)

    n, c, h, w = input_tensor.size()
    output_tensor = torch.zeros((n, 3, h, w), dtype=input_tensor.dtype, device=input_tensor.device)

    count_tensor = torch.zeros((n, 3, h, w), dtype=torch.float32, device=input_tensor.device)

    add = 2 if window_size % stride != 0 else 1

    for y in range(0, h - window_size + add, stride):
        for x in range(0, w - window_size + add, stride):
            window = input_tensor[:, :, y:y + window_size, x:x + window_size]

            # Apply the same preprocessing as during training
            input_variable = Variable(window, requires_grad=False)  # Assuming GPU is available

            # Forward pass
            with torch.no_grad():
                output = your_model(input_variable)

            output_tensor[:, :, y:y + window_size, x:x + window_size] += output
            count_tensor[:, :, y:y + window_size, x:x + window_size] += 1

    count_tensor = torch.clamp(count_tensor, min=1.0)

    # Average the overlapping regions
    output_tensor /= count_tensor

    # Apply mask
    output_tensor *= mask

    return output_tensor.cpu()


def process_image(your_model, image, video, source_age, target_age=0,
                  window_size=512, stride=256, steps=18):
    """
    Aging the person in the image.
    If video=False, we age as from source_age to target_age, and return an image.
    If video=True, we age from source_age to a range of target ages, and return this as the path to a video.
    """
    if video:
        target_age = 0
    input_size = (1024, 1024)

    # image = face_recognition.load_image_file(filename)
    image = np.array(image)
    fl = face_recognition.face_locations(image)[0]

    # calculate margins
    margin_y_t = int((fl[2] - fl[0]) * .63 * .85)  # larger as the forehead is often cut off
    margin_y_b = int((fl[2] - fl[0]) * .37 * .85)
    margin_x = int((fl[1] - fl[3]) // (2 / .85))
    margin_y_t += 2 * margin_x - margin_y_t - margin_y_b  # make sure square is preserved

    l_y = max([fl[0] - margin_y_t, 0])
    r_y = min([fl[2] + margin_y_b, image.shape[0]])
    l_x = max([fl[3] - margin_x, 0])
    r_x = min([fl[1] + margin_x, image.shape[1]])

    # crop image
    cropped_image = image[l_y:r_y, l_x:r_x, :]

    # Resizing
    orig_size = cropped_image.shape[:2]

    cropped_image = transforms.ToTensor()(cropped_image)

    cropped_image_resized = transforms.Resize(input_size, interpolation=Image.BILINEAR, antialias=True)(cropped_image)

    source_age_channel = torch.full_like(cropped_image_resized[:1, :, :], source_age / 100)
    target_age_channel = torch.full_like(cropped_image_resized[:1, :, :], target_age / 100)
    input_tensor = torch.cat([cropped_image_resized, source_age_channel, target_age_channel], dim=0).unsqueeze(0)

    image = transforms.ToTensor()(image)

    if video:
        # aging in steps
        interval = 1 / steps
        aged_cropped_images = torch.zeros((steps, 3, input_size[1], input_size[0]))
        for i in range(0, steps):
            input_tensor[:, -1, :, :] += interval

            # performing actions on image
            aged_cropped_images[i, ...] = sliding_window_tensor(input_tensor, window_size, stride, your_model)

        # resize back to original size
        aged_cropped_images_resized = transforms.Resize(orig_size, interpolation=Image.BILINEAR, antialias=True)(
            aged_cropped_images)

        # re-apply
        image = image.repeat(steps, 1, 1, 1)

        image[:, :, l_y:r_y, l_x:r_x] += aged_cropped_images_resized
        image = torch.clamp(image, 0, 1)
        image = (image * 255).to(torch.uint8)

        output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

        write_video(output_file.name, image.permute(0, 2, 3, 1), 2)

        return output_file.name

    else:
        # performing actions on image
        aged_cropped_image = sliding_window_tensor(input_tensor, window_size, stride, your_model)

        # resize back to original size
        aged_cropped_image_resized = transforms.Resize(orig_size, interpolation=Image.BILINEAR, antialias=True)(
            aged_cropped_image)

        # re-apply
        image[:, l_y:r_y, l_x:r_x] += aged_cropped_image_resized.squeeze(0)
        image = torch.clamp(image, 0, 1)

        return transforms.functional.to_pil_image(image)
