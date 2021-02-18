import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from PIL import Image
from torchvision import transforms

def plot_curve_w_linefit(datapoints, title):
    plt.plot(np.arange(len(datapoints)), datapoints)
    
    b, m = polyfit(np.arange(len(datapoints)), datapoints, 1)
    plt.plot(np.arange(len(datapoints)), 
            b + m * np.arange(len(datapoints)))
    plt.title(title)
    plt.show()
    return b, m

def depreprocess_img(tensor_image, imagenet=True):
    if imagenet:
        # Invert imagenet normalization
        n_inv = transforms.Normalize([-0.485/0.229, -0.546/0.224, -0.406/0.225], 
                             [1/0.229, 1/0.224, 1/0.225])
        tensor_image = n_inv(tensor_image)
    else:
        tensor_image *= 255.
    arr = tensor_image.permute(1, 2, 0).cpu().numpy()
    arr *= 255.0
    arr[arr < 0.0] = 0.0
    arr[arr > 255.0] = 255.0
    arr = arr.astype(np.uint8)
    return Image.fromarray(arr)