"""
Try-On GAN Model - Generator and Discriminator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with instance normalization"""
    def __init__(self, channels: int):
        super(ResBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.relu(out)
        return out


class TryOnGenerator(nn.Module):
    """
    Generator for try-on synthesis
    
    Takes warped cloth, body segmentation, and pose as input
    Outputs realistic try-on image
    """
    def __init__(self, input_channels: int = 22, output_channels: int = 3, 
                 ngf: int = 64, num_res_blocks: int = 6):
        """
        Args:
            input_channels: warped cloth (3) + segmentation (1) + pose (18) = 22
            output_channels: RGB image (3)
            ngf: Number of generator filters in first layer
            num_res_blocks: Number of residual blocks
        """
        super(TryOnGenerator, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (B, 22, H, W)
            nn.Conv2d(input_channels, ngf, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            
            # Downsample 1
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            
            # Downsample 2
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResBlock(ngf * 4))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsample 1
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            
            # Upsample 2
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            
            # Output
            nn.Conv2d(ngf, output_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )
    
    def forward(self, warped_cloth, segmentation, pose_heatmap):
        """
        Args:
            warped_cloth: (B, 3, H, W)
            segmentation: (B, 1, H, W)
            pose_heatmap: (B, 18, H, W)
        
        Returns:
            generated_image: (B, 3, H, W)
        """
        # Concatenate all inputs
        x = torch.cat([warped_cloth, segmentation, pose_heatmap], dim=1)
        
        # Encode
        x = self.encoder(x)
        
        # Residual blocks
        x = self.res_blocks(x)
        
        # Decode
        output = self.decoder(x)
        
        return output


class UNetGenerator(nn.Module):
    """
    U-Net style generator with skip connections
    Better preserves details
    """
    def __init__(self, input_channels: int = 22, output_channels: int = 3, ngf: int = 64):
        super(UNetGenerator, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, ngf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, warped_cloth, segmentation, pose_heatmap):
        """
        Args:
            warped_cloth: (B, 3, H, W)
            segmentation: (B, 1, H, W)
            pose_heatmap: (B, 18, H, W)
        
        Returns:
            generated_image: (B, 3, H, W)
        """
        # Concatenate inputs
        x = torch.cat([warped_cloth, segmentation, pose_heatmap], dim=1)
        
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with skip connections
        d1 = self.dec1(b)
        d1 = torch.cat([d1, e4], dim=1)
        
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        
        output = self.final(d4)
        
        return output


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator
    Classifies whether overlapping image patches are real or fake
    """
    def __init__(self, input_channels: int = 6, ndf: int = 64, n_layers: int = 3):
        """
        Args:
            input_channels: person image (3) + generated/real (3) = 6
            ndf: Number of discriminator filters in first layer
            n_layers: Number of layers in discriminator
        """
        super(PatchGANDiscriminator, self).__init__()
        
        layers = []
        
        # First layer (no normalization)
        layers.append(nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Additional layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                                   kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Penultimate layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                               kernel_size=4, stride=1, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layer
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, person_image, generated_or_real):
        """
        Args:
            person_image: (B, 3, H, W) original person image
            generated_or_real: (B, 3, H, W) generated or real try-on image
        
        Returns:
            output: (B, 1, H', W') patch-wise predictions
        """
        x = torch.cat([person_image, generated_or_real], dim=1)
        output = self.model(x)
        return output


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for better detail discrimination
    """
    def __init__(self, input_channels: int = 6, ndf: int = 64, num_scales: int = 3):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.num_scales = num_scales
        
        # Create discriminator for each scale
        self.discriminators = nn.ModuleList()
        for _ in range(num_scales):
            self.discriminators.append(PatchGANDiscriminator(input_channels, ndf))
        
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, person_image, generated_or_real):
        """
        Args:
            person_image: (B, 3, H, W)
            generated_or_real: (B, 3, H, W)
        
        Returns:
            outputs: List of discriminator outputs at different scales
        """
        outputs = []
        
        person_downsampled = person_image
        gen_downsampled = generated_or_real
        
        for i in range(self.num_scales):
            outputs.append(self.discriminators[i](person_downsampled, gen_downsampled))
            
            if i != self.num_scales - 1:
                person_downsampled = self.downsample(person_downsampled)
                gen_downsampled = self.downsample(gen_downsampled)
        
        return outputs


def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def get_tryon_models(generator_type: str = 'unet', 
                     discriminator_type: str = 'patchgan',
                     input_channels: int = 22) -> tuple:
    """
    Factory function to get try-on GAN models
    
    Args:
        generator_type: 'resnet' or 'unet'
        discriminator_type: 'patchgan' or 'multiscale'
        input_channels: Number of input channels for generator
    
    Returns:
        generator, discriminator
    """
    # Generator
    if generator_type == 'resnet':
        generator = TryOnGenerator(input_channels=input_channels)
    elif generator_type == 'unet':
        generator = UNetGenerator(input_channels=input_channels)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")
    
    # Discriminator
    if discriminator_type == 'patchgan':
        discriminator = PatchGANDiscriminator(input_channels=6)
    elif discriminator_type == 'multiscale':
        discriminator = MultiScaleDiscriminator(input_channels=6)
    else:
        raise ValueError(f"Unknown discriminator type: {discriminator_type}")
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    return generator, discriminator


if __name__ == '__main__':
    # Test try-on GAN models
    print("Testing Try-On GAN Models...")
    
    batch_size = 2
    warped_cloth = torch.randn(batch_size, 3, 256, 192)
    segmentation = torch.randn(batch_size, 1, 256, 192)
    pose = torch.randn(batch_size, 18, 256, 192)
    person = torch.randn(batch_size, 3, 256, 192)
    
    # Test ResNet generator
    print("\nResNet Generator:")
    gen_resnet = TryOnGenerator(input_channels=22)
    output = gen_resnet(warped_cloth, segmentation, pose)
    print(f"  Output shape: {output.shape}")
    num_params = sum(p.numel() for p in gen_resnet.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    # Test U-Net generator
    print("\nU-Net Generator:")
    gen_unet = UNetGenerator(input_channels=22)
    output = gen_unet(warped_cloth, segmentation, pose)
    print(f"  Output shape: {output.shape}")
    num_params = sum(p.numel() for p in gen_unet.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    # Test PatchGAN discriminator
    print("\nPatchGAN Discriminator:")
    disc = PatchGANDiscriminator(input_channels=6)
    disc_output = disc(person, output)
    print(f"  Output shape: {disc_output.shape}")
    num_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    # Test MultiScale discriminator
    print("\nMultiScale Discriminator:")
    disc_ms = MultiScaleDiscriminator(input_channels=6)
    disc_outputs = disc_ms(person, output)
    print(f"  Number of scales: {len(disc_outputs)}")
    for i, out in enumerate(disc_outputs):
        print(f"  Scale {i} output shape: {out.shape}")
    num_params = sum(p.numel() for p in disc_ms.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    
    print("\n✅ Try-On GAN models ready!")
