import torch
import torch.nn as nn
from torchvision.models import vgg13_bn, vgg16_bn
import torch.optim as optim
test_model = True

__all__ = ['vgg13bn_unet', 'vgg16bn_unet']


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels, kernel_size, stride):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=kernel_size, stride=stride
    )


class VGGNET(nn.Module):
    """Unet with VGG-13 (with BN), VGG-16 (with BN) encoder.
    """

    def __init__(self, encoder, *, pretrained=False, nclasses): #out_channels=2):
        super().__init__()
        self.nclasses = nclasses
        
        self.encoder = encoder(pretrained=pretrained).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])

        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = double_conv(512, 1024)

        self.up_conv6 = up_conv(1024, 512, kernel_size = 2, stride = 2)
        self.conv6 = double_conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256, kernel_size = 2, stride = 2)
        self.conv7 = double_conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128, kernel_size = 2, stride = 2)
        self.conv8 = double_conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64, kernel_size = 2, stride = 2)
        self.conv9 = double_conv(64 + 128, 64)
        self.up_conv10 = up_conv(64, 32, kernel_size = 2, stride = 2)
        self.conv10 = double_conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, nclasses, kernel_size=1) # out_channels, kernel

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x



def vgg13bn_unet(nclasses , pretrained = False):    
    return VGGNET(vgg13_bn, pretrained=pretrained, nclasses=nclasses)


def vgg16bn_unet(nclasses, pretrained = False):    
    return VGGNET(vgg16_bn, pretrained=pretrained, nclasses=nclasses)

if __name__ == "__main__":
    if test_model == True:
    
        batch_size, nclasses, h, w = 10, 20, 160, 160
        
        # test output size
        
        unet_model = vgg16bn_unet(nclasses = nclasses, pretrained=True)
        input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
        output = unet_model(input)
        assert output.size() == torch.Size([batch_size, nclasses, h, w])
        
        print("Passed size check")
        
        criterion = nn.BCELoss()
        optimizer = optim.SGD(unet_model.parameters(), lr=1e-3, momentum=0.9)
        input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
        y = torch.autograd.Variable(torch.randn(batch_size, nclasses, h, w), requires_grad=False)
        for iter in range(10):
            optimizer.zero_grad()
            out = unet_model(input)
            # print(out)
            out = torch.sigmoid(out)
            loss = criterion(out, y)
            loss.backward()
            # print(loss)
            print("iter{}, loss {}".format(iter, loss.item()))
            optimizer.step()