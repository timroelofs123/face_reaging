import torch
import torch.nn as nn
import lpips  # LPIPS library for perceptual loss

class GeneratorLoss(nn.Module):
    def __init__(self, discriminator_model, l1_weight=1.0, perceptual_weight=1.0, adversarial_weight=0.05,
                 device="cpu"):
        super(GeneratorLoss, self).__init__()
        self.discriminator_model = discriminator_model
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.criterion_l1 = nn.L1Loss()
        self.criterion_adversarial = nn.BCEWithLogitsLoss()
        self.criterion_perceptual = lpips.LPIPS(net='vgg').to(device)

    def forward(self, output, target, source):
        # L1 loss

        l1_loss = self.criterion_l1(output, target)

        # Perceptual loss
        perceptual_loss = torch.mean(self.criterion_perceptual(output, target))

        # Adversarial loss
        fake_input = torch.cat([output, source[:, 4:5, :, :]], dim=1)
        fake_prediction = self.discriminator_model(fake_input)

        adversarial_loss = self.criterion_adversarial(fake_prediction, torch.ones_like(fake_prediction))

        # Combine losses
        generator_loss = self.l1_weight * l1_loss + self.perceptual_weight * perceptual_loss + \
                         self.adversarial_weight * adversarial_loss

        return generator_loss, l1_loss, perceptual_loss, adversarial_loss

class DiscriminatorLoss(nn.Module):
    def __init__(self, discriminator_model, fake_weight=1.0, real_weight=2.0, mock_weight=.5):
        super(DiscriminatorLoss, self).__init__()
        self.discriminator_model = discriminator_model
        self.criterion_adversarial = nn.BCEWithLogitsLoss()
        self.fake_weight = fake_weight
        self.real_weight = real_weight
        self.mock_weight = mock_weight

    def forward(self, output, target, source):
        # Adversarial loss
        fake_input = torch.cat([output, source[:, 4:5, :, :]], dim=1)  # prediction img with target age
        real_input = torch.cat([target, source[:, 4:5, :, :]], dim=1)  # target img with target age

        mock_input1 = torch.cat([source[:, :3, :, :], source[:, 4:5, :, :]], dim=1)  # source img with target age
        mock_input2 = torch.cat([target, source[:, 3:4, :, :]], dim=1)  # target img with source age
        mock_input3 = torch.cat([output, source[:, 3:4, :, :]], dim=1)  # prediction img with source age
        mock_input4 = torch.cat([target, source[:, 3:4, :, :]], dim=1)  # target img with target age

        fake_pred, real_pred = self.discriminator_model(fake_input), self.discriminator_model(real_input)
        mock_pred1, mock_pred2, mock_pred3, mock_pred4 = (self.discriminator_model(mock_input1),
                                                          self.discriminator_model(mock_input2),
                                                          self.discriminator_model(mock_input3),
                                                          self.discriminator_model(mock_input4))

        discriminator_loss = (self.fake_weight * self.criterion_adversarial(fake_pred, torch.zeros_like(fake_pred)) +
                              self.real_weight * self.criterion_adversarial(real_pred, torch.ones_like(real_pred)) +
                              self.mock_weight * self.criterion_adversarial(mock_pred1, torch.zeros_like(mock_pred1)) +
                              self.mock_weight * self.criterion_adversarial(mock_pred2, torch.zeros_like(mock_pred2)) +
                              self.mock_weight * self.criterion_adversarial(mock_pred3, torch.zeros_like(mock_pred3)) +
                              self.mock_weight * self.criterion_adversarial(mock_pred4, torch.zeros_like(mock_pred4))
                              )

        return discriminator_loss
