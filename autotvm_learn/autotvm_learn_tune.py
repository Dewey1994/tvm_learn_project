import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner,GATuner,RandomTuner,GridSearchTuner
from tvm import autotvm
import tvm
import tvm.relay as relay
import onnx
import numpy as np
from tvm.contrib.download import download_testdata
from PIL import Image
from tvm.contrib import graph_executor
from scipy.special import softmax


# model_url = (
#     "https://github.com/onnx/models/raw/main/"
#     "vision/classification/resnet/model/"
#     "resnet50-v2-7.onnx"
# )
#
# model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
model_path = '../tvmc_learn/resnet50-v2-7.onnx'
onnx_model = onnx.load(model_path)

# Seed numpy's RNG to get consistent results
np.random.seed(0)


img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# Resize it to 224x224
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# Our input image is in HWC layout while ONNX expects CHW input, so convert the array
img_data = np.transpose(img_data, (2, 0, 1))

# Normalize according to the ImageNet input specification
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

# Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
img_data = np.expand_dims(norm_img_data, axis=0)

target='llvm'

# The input name may vary across model types. You can use a tool
# like Netron to check input names
input_name = "data"
shape_dict = {input_name: img_data.shape}

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

