import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse

import sys
sys.path.append(".")

from model.models import UNet, PatchGANDiscriminator
from model.losses import GeneratorLoss, DiscriminatorLoss
from utils.dataloader import CustomDataset, transform


def train_model(root_dir, start_epoch, num_epochs, load_model_g, load_model_d, num_workers,
                val_freq, batch_size, accum_iter, lr, lr_d, wandb_tracking, desc):
    if wandb_tracking:
        import wandb

        wandb.init(project="FRAN",
                   # track hyperparameters and run metadata
                   config={
                       "lr": lr,
                       "lr_d": lr_d,
                       "dataset": root_dir,
                       "epochs": num_epochs,
                       "batch_size": batch_size,
                       "description": desc
                   }
                   )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if torch.cuda.device_count() > 0:
        print(f"{torch.cuda.device_count()} GPU(s)")
        if torch.cuda.device_count() > 1:
            print("multi-GPU training is currently not supported.")

    # Create instances of the dataset and split into scripts and validation sets
    dataset = CustomDataset(root_dir=root_dir, transform=transform)

    # Assuming you want to use 80% of the data for scripts and 20% for validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for scripts and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Create instances of the U-Net, discriminator, and loss models
    unet_model = UNet()
    discriminator_model = PatchGANDiscriminator(input_channels=4)

    if load_model_g:
        unet_model.load_state_dict(torch.load(load_model_g, map_location=device))
        print(f'loaded {load_model_g} for unet_model')
    if load_model_d:
        discriminator_model.load_state_dict(torch.load(load_model_d, map_location=device))
        print(f'loaded {load_model_d} for discriminator_model')

    unet_model = unet_model.to(device)
    discriminator_model = discriminator_model.to(device)

    # if multiGPU:
    #    unet_model = nn.DataParallel(unet_model)
    #    discriminator_model = nn.DataParallel(discriminator_model)

    # Create loss instances
    generator_loss_func = GeneratorLoss(discriminator_model, l1_weight=1.0, perceptual_weight=1.0,
                                        adversarial_weight=0.05, device=device)
    discriminator_loss_func = DiscriminatorLoss(discriminator_model)

    # Create instances of the Adam optimizer
    optimizer_g = optim.Adam(unet_model.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator_model.parameters(), lr=lr_d)

    # Training and validation loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch - 1, num_epochs):
        # Training
        unet_model.train()
        discriminator_model.train()
        batch_idx = 0
        for batch in train_dataloader:
            batch_idx += 1
            source_images, target_images = batch

            # if not multiGPU:
            # if multi GPU, nn.DataParallel will already put the batches on the right devices.
            # Otherwise, we do it manually
            source_images = source_images.to(device)
            target_images = target_images.to(device)

            # Zero gradients
            # optimizer_g.zero_grad()
            # optimizer_d.zero_grad()

            # Forward pass
            output_images = unet_model(source_images)
            # if multiGPU:
            #     output_device = output_images.get_device()
            #     source_images, target_images = source_images.to(output_device), target_images.to(output_device)
            output_images += source_images[:, :3, :, :]

            # Discriminator pass
            discriminator_loss = discriminator_loss_func(output_images.detach(), target_images, source_images)
            # discriminator_loss /= accum_iter
            discriminator_loss.backward()

            if (batch_idx % accum_iter == 0) or (batch_idx == len(train_dataloader)):
                optimizer_d.step()
                optimizer_d.zero_grad()

            # Generator pass
            # Calculate the loss
            generator_loss, l1_loss, per_loss, adv_loss = generator_loss_func(output_images, target_images,
                                                                              source_images)
            generator_loss, l1_loss, per_loss, adv_loss = [i / accum_iter for i in
                                                           [generator_loss, l1_loss, per_loss, adv_loss]]
            generator_loss.backward()

            if (batch_idx % accum_iter == 0) or (batch_idx == len(train_dataloader)):
                optimizer_g.step()
                optimizer_g.zero_grad()

            # Print scripts information (if needed)
            print(
                f'Training Epoch [{epoch + 1}/{num_epochs}], Gen Loss: {generator_loss.item()}, L1: {l1_loss.item()}, P: {per_loss.item()}, A: {adv_loss.item()}, Dis Loss: {discriminator_loss.item()}')
            if wandb_tracking:
                wandb.log({
                    'Training Epoch': epoch + 1,
                    'Gen Loss': generator_loss.item(),
                    'L1': l1_loss.item(),
                    'P': per_loss.item(),
                    'A': adv_loss.item(),
                    'Dis Loss': discriminator_loss.item()
                })

        torch.save(unet_model.state_dict(), 'recent_unet_model.pth')
        torch.save(discriminator_model.state_dict(), 'recent_discriminator_model.pth')

        # Validation
        if epoch % val_freq == 0:
            unet_model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_source_images, val_target_images = val_batch

                    # if not multiGPU:
                    # if multi GPU, nn.DataParallel will already put the batches on the right devices.
                    # Otherwise, we do it manually
                    val_source_images = val_source_images.to(device)
                    val_target_images = val_target_images.to(device)

                    # Forward pass
                    val_output_images = unet_model(val_source_images)

                    # if multiGPU:
                    #     output_device = val_output_images.get_device()
                    #     val_source_images, val_target_images = val_source_images.to(output_device), \
                    #         val_target_images.to(output_device)

                    # Calculate the loss
                    generator_loss, _, _, _ = generator_loss_func(val_output_images, val_target_images,
                                                                  val_source_images)
                    total_val_loss += generator_loss.item()

            average_val_loss = total_val_loss / len(val_dataloader)

            # Print validation information
            print(f'Validation Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_val_loss}')
            if wandb_tracking:
                wandb.log({
                    'Training Epoch': epoch + 1,
                    'Val Loss': average_val_loss,
                })

            # Save the model with the best validation loss
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                torch.save(unet_model.state_dict(), 'best_unet_model.pth')
                torch.save(discriminator_model.state_dict(), 'best_discriminator_model.pth')

    if wandb_tracking:
        wandb.finish()


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--root_dir", type=str, default='data/processed/train',
                        help="Path to the training data. Note the format: To use the dataloader, the directory should be filled with folders containing image files of various ages, where the file name is the age.")
    parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch, if scripts is resumed")
    parser.add_argument("--num_epochs", type=int, default=2000, help="End epoch")
    parser.add_argument("--load_model_g", type=str, default='',
                        help="Path to pretrained generator model. Leave blank to train from scratch")
    parser.add_argument("--load_model_d", type=str, default='',
                        help="Path to pretrained discriminator model. Leave blank to train from scratch")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size")
    parser.add_argument("--accum_iter", type=int, default=3, help="Number of batches after which weights are updated")
    parser.add_argument("--val_freq", type=int, default=1, help="Validation frequency (epochs)")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate for generator")
    parser.add_argument("--lr_d", type=float, default=0.00001, help="Learning rate for discriminator")
    parser.add_argument("--wandb_tracking", help="A binary (True/False) argument for using WandB tracking or not")
    parser.add_argument("--desc", type=str, default='', help="Description for WandB")

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the scripts function with parsed arguments
    train_model(args.root_dir, args.start_epoch, args.num_epochs, args.load_model_g, args.load_model_d,
                args.num_workers, args.val_freq, args.batch_size, args.accum_iter, args.lr, args.lr_d,
                args.wandb_tracking, args.desc)
