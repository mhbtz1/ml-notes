import torch
from torch import nn
import numpy as np

# Note: In CNNs, the number of filters used in a single convlayer is equal to the number of output channels when it is applied to some input feature map
# This is because convolutions are done along the spatial dimension of a feature map

torch.manual_seed(2000)

conv_layer = nn.Conv1d(
    in_channels=4,
    out_channels=2,
    kernel_size=3,
    padding="valid",
    groups=2,
    bias=False,
    dtype=torch.float32
)

for param in conv_layer.parameters():
    param.requires_grad = False

print(f"Conv1d weight tensor shape: {conv_layer.weight.size()}") 
# dims above represent (input_channel_size, output_channel_size, kernel_dims)
# for a nn.Conv1d, the convolution operation is over a 1d tensor, kernel_dims is only going to be 1 number

feature_map = torch.randn(1, 4, 32, dtype=torch.float32)

first_kernel = conv_layer.weight[0][0].numpy()
second_kernel = conv_layer.weight[0][1].numpy()
third_kernel = conv_layer.weight[1][0].numpy()
fourth_kernel = conv_layer.weight[1][1].numpy()
# basically in grouped convolution, we apply the same kernel to some channels, sum their result,
# and use their result as the output channel's value

output1 = np.correlate(feature_map.numpy()[0][0].squeeze(), first_kernel)
output2 = np.correlate(feature_map.numpy()[0][1].squeeze(), second_kernel)
output3 = np.correlate(feature_map.numpy()[0][2].squeeze(), third_kernel)
output4 = np.correlate(feature_map.numpy()[0][3].squeeze(), fourth_kernel)

print(f"Output from numpy correlation ops on channels 0-1: {output1 + output2}")
print(f"Output from pytorch grouped convolution in output channel 0: {conv_layer(feature_map)[0][0]}")

print(f"Output from numpy correlation ops on channels 2-3: {output3 + output4}")
print(f"Output from pytorch grouped convolution in output channel 1: {conv_layer(feature_map)[0][1]}")



