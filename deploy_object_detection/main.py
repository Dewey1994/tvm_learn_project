import tvm
from tvm import relay
from tvm import relay
from tvm.runtime.vm import VirtualMachine
from tvm.contrib.download import download_testdata

import numpy as np
import cv2
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torchvision

in_size = 600

input_shape = (1, 3, in_size, in_size)


def do_trace(model, inp):
    model_trace = torch.jit.trace(model, inp)
    model_trace.eval()
    return model_trace


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])


# model_func = torchvision.models.detection.retinanet_resnet50_fpn
model_func = torch.hub.load('zhiqwang/yolov5-rt-stack', 'yolov5s',pretrained=True)
model = TraceWrapper(model_func)

model.eval()
inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=(1, 3, in_size, in_size)))

with torch.no_grad():
    out = model(inp)
    script_module = do_trace(model, inp)


img_url = (
    "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg"
# "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
)
img_path = download_testdata(img_url, "street_small.jpg", module="data")

img = cv2.imread(img_path).astype("float32")
img1 = cv2.imread(img_path)
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img1 = cv2.resize(img1, (in_size, in_size))
img = cv2.resize(img, (in_size, in_size))
# call imshow() using plt object
plt.imshow(img1)
plt.show()


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.transpose(img / 255.0, [2, 0, 1])
img = np.expand_dims(img, axis=0)


input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(script_module, shape_list)


# Add "-libs=mkl" to get best performance on x86 target.
# For x86 machine supports AVX512, the complete target is
# "llvm -mcpu=skylake-avx512 -libs=mkl"
target = "llvm"

with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
    vm_exec = relay.vm.compile(mod, target=target, params=params)


dev = tvm.cpu()
vm = VirtualMachine(vm_exec, dev)
vm.set_input("main", **{input_name: img})
tvm_res = vm.run()


score_threshold = 0.6
boxes = tvm_res[0].numpy().tolist()
valid_boxes = []
for i, score in enumerate(tvm_res[1].numpy().tolist()):
    if score > score_threshold:
        valid_boxes.append(boxes[i])
    else:
        break

print("Get {} valid boxes".format(len(valid_boxes)))