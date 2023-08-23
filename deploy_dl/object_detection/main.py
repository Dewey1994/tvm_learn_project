import random
import time

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


model_func = torchvision.models.detection.maskrcnn_resnet50_fpn
# model_func = torch.hub.load('zhiqwang/yolov5-rt-stack', 'yolov5s',pretrained=True)
model = TraceWrapper(model_func)
model = TraceWrapper(model_func(pretrained=True))

model.eval()
inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=(1, 3, in_size, in_size)))

with torch.no_grad():
    out = model(inp)
    script_module = do_trace(model, inp)

start = time.time()
repeats = 50
for i in range(repeats):
    np_out = model(inp)
end = time.time()
print("torch Runtime: %f ms." % (1000 * ((end - start) / repeats)))

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


start = time.time()
repeats = 50
for i in range(repeats):
    tvm_res = vm.run()
end = time.time()
print("tvm Runtime: %f ms." % (1000 * ((end - start) / repeats)))

label_dict={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
           9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
           16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
           25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
           35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
           41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
           48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
           56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
           64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
           75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
           82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
           90: 'toothbrush'}

score_threshold = 0.9
boxes = tvm_res[0].numpy().tolist()
scores = tvm_res[1].numpy().tolist()
labels = tvm_res[2].numpy().tolist()
valid_boxes = []
valid_scores = []
valid_labels = []
for i, score in enumerate(tvm_res[1].numpy().tolist()):
    if score > score_threshold:
        valid_boxes.append([int(i) for i in boxes[i]])
        valid_scores.append(scores[i])
        valid_labels.append(label_dict[labels[i]])

    else:
        break



def draw_label_type(draw_img,bbox,label,label_color):
    labelSize = cv2.getTextSize(label + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    if bbox[1] - labelSize[1] - 3 < 0:
        cv2.rectangle(draw_img,
                      (bbox[0], bbox[1] + 2),
                      (bbox[0] + labelSize[0], bbox[1] + labelSize[1] + 3),
                      color=label_color,
                      thickness=-1
                      )
        cv2.putText(draw_img, label,
                    (bbox[0], bbox[1] + labelSize + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1
                    )
    else:
        cv2.rectangle(draw_img,
                      (bbox[0], bbox[1] - labelSize[1] - 3),
                      (bbox[0] + labelSize[0], bbox[1] - 3),
                      color=label_color,
                      thickness=-1
                      )
        cv2.putText(draw_img, label,
                    (bbox[0], bbox[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1
                    )
    return draw_img

def get_different_color_for_label(labels):
    mp = dict()
    for i in labels:
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        mp[i]=(r,g,b)
    return mp

def postprocess(img,boxes,labels):
    st_label = set(labels)
    different_labels = get_different_color_for_label(st_label)
    for i in range(len(boxes)):
        img = draw_label_type(img,boxes[i],labels[i],different_labels[labels[i]])
        img = cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), color=different_labels[labels[i]], thickness=2)
    return img

img_res = postprocess(img1,valid_boxes,valid_labels)
plt.imshow(img_res)
plt.show()
print("Get {} valid boxes".format(len(valid_boxes)))


