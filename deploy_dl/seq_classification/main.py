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
from transformers import AutoModel, AutoTokenizer, BertForMaskedLM
import onnx
# PyTorch imports
import torch



###############################
# change your config here
n_trails = 2000  # higher is better.
n_early_stopping = 600  # higher is better.
set_seqlen_myself = False  # if set to be true, the model will use the seq_len you set below
seq_len = 512  # only take effect when set_seqlen_myself = True
target = "llvm"
##############################

tokenizer = AutoTokenizer.from_pretrained("/Users/dewey/bert-base-uncased")
device = torch.device("cpu")

# Tokenizing input text
if set_seqlen_myself:
    input_ids = list(np.random.randint(0, 25000, seq_len))
    input_ids[0] = 102
    input_ids[-1] = 103
    atten_mask = list(np.ones(seq_len, dtype=int))
    token_type_ids = list(np.zeros(seq_len, dtype=int))
else:
    sentence_a = "Who was Jim Henson ?"
    sentence_b = "Jim Henson was a puppeteer."
    tokenized_text = tokenizer(sentence_a, sentence_b, padding='max_length')  # will expand to 512 length
    input_ids = tokenized_text['input_ids']
    atten_mask = tokenized_text['attention_mask']
    token_type_ids = tokenized_text['token_type_ids']

seq_len = len(input_ids)

# Creating a dummy input
input_ids_tensor = torch.tensor([input_ids])
atten_mask_tensors = torch.tensor([atten_mask])
token_type_ids_tensors = torch.tensor([token_type_ids])

model = BertForMaskedLM.from_pretrained("/Users/dewey/bert-base-uncased")
res = model(input_ids=input_ids_tensor,attention_mask=atten_mask_tensors,token_type_ids=token_type_ids_tensors)



import timeit

timing_number = 10
timing_repeat = 10
unoptimized = (
    np.array(timeit.Timer(lambda: model(input_ids=input_ids_tensor,attention_mask=atten_mask_tensors,token_type_ids=token_type_ids_tensors)).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}

print(unoptimized)


dummy_input = [input_ids_tensor, atten_mask_tensors, token_type_ids_tensors]


# If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
# model = AutoModel.from_pretrained("/Users/dewey/bert-base-uncased", onnx=True)
onnx_model = onnx.load('/Users/dewey/bert-base-uncased/model.onnx')
# The model needs to be in evaluation mode
# model.eval()
#
# # Creating the trace
# traced_model = torch.jit.trace(model, dummy_input)
# # traced_model = torch.jit.script(model, dummy_input)
# traced_model.eval()
# script_module = traced_model
#
input_infos = [("input_ids", input_ids_tensor.shape), ("attention_mask", atten_mask_tensors.shape),
               ("token_type_ids", token_type_ids_tensors.shape)]
input_dict = {"input_ids":input_ids_tensor.shape,"attention_mask":atten_mask_tensors.shape,"token_type_ids":token_type_ids_tensors.shape}
mod, params = relay.frontend.from_onnx(onnx_model, input_dict)


# Add "-libs=mkl" to get best performance on x86 target.
# For x86 machine supports AVX512, the complete target is
# "llvm -mcpu=skylake-avx512 -libs=mkl"
target = "llvm"

with tvm.transform.PassContext(opt_level=3):
    vm_exec = relay.vm.compile(mod, target=target, params=params)


dev = tvm.cpu()
vm = VirtualMachine(vm_exec, dev)
vm.set_input("main", **{'input_ids': input_ids_tensor.numpy(),'attention_mask':atten_mask_tensors.numpy(),
                        'token_type_ids':token_type_ids_tensors.numpy()})

import timeit

timing_number = 10
timing_repeat = 10
unoptimized = (
    np.array(timeit.Timer(lambda: vm.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}

print(unoptimized)


