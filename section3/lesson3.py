import sys, os
import numpy as np
from PIL import Image
module_dir = os.path.join(os.getcwd(), 'deep_learning_from_scratch_master')
sys.path.append(module_dir)
from dataset.mnist import load_mnist

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label) # 5

print(img.shape) # (784,)
img = img.reshape(28, 28)
print(img.shape) # (28, 28)

img_show(img)