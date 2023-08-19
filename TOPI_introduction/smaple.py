import tvm
import tvm.testing
from tvm import te
from tvm import topi
import numpy as np


n = te.var("n")
m = te.var("m")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), "k")
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
s = te.create_schedule(B.op)

print(tvm.lower(s, [A], simple_mode=True))

C = topi.sum(A, axis=1)
ts = te.create_schedule(C.op)
print(tvm.lower(ts, [A], simple_mode=True))
print(ts.stages)
