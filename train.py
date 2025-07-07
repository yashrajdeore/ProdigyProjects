

import torch
from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = UNetGenerator().to(device)
discriminator = PatchGANDiscriminator().to(device)

# Dummy input test
input_image = torch.randn((1, 3, 256, 256)).to(device)
target_image = torch.randn((1, 3, 256, 256)).to(device)

# Forward pass
fake_image = generator(input_image)
disc_out = discriminator(torch.cat((input_image, fake_image), 1))

print("Generator output shape:", fake_image.shape)
print("Discriminator output shape:", disc_out.shape)
