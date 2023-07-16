alias tvmc='python -m tvm.driver.tvmc'


#-----直接编译和运行resnet-50-----
tvmc compile \
--target "llvm" \
--input-shapes "data:[1,3,224,224]" \
--output resnet50-v2-7-tvm.tar \
resnet50-v2-7.onnx

tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz \
resnet50-v2-7-tvm.tar

#-----auto-tune和编译resnet-50-----
tvmc tune \
--target "llvm" \
--output resnet50-v2-7-autotuner_records.json \
resnet50-v2-7.onnx

tvmc compile \
--target "llvm" \
--tuning-records resnet50-v2-7-autotuner_records.json  \
--output resnet50-v2-7-tvm_autotuned.tar \
resnet50-v2-7.onnx

# [Task  1/25]  Current/Best:  103.85/ 326.97 GFLOPS | Progress: (40/40) | 39.39 s Done.
# [Task  2/25]  Current/Best:   21.16/ 210.91 GFLOPS | Progress: (40/40) | 31.37 s Done.
# [Task  3/25]  Current/Best:  120.88/ 302.07 GFLOPS | Progress: (40/40) | 32.83 s Done.
# [Task  4/25]  Current/Best:  146.72/ 277.11 GFLOPS | Progress: (40/40) | 31.61 s Done.
# [Task  5/25]  Current/Best:  130.65/ 248.43 GFLOPS | Progress: (40/40) | 32.35 s Done.
# [Task  6/25]  Current/Best:  190.06/ 273.91 GFLOPS | Progress: (40/40) | 29.77 s Done.
# [Task  7/25]  Current/Best:   77.96/ 259.97 GFLOPS | Progress: (40/40) | 34.91 s Done.
# [Task  8/25]  Current/Best:   98.48/ 316.33 GFLOPS | Progress: (40/40) | 36.18 s Done.
# [Task  9/25]  Current/Best:  150.97/ 261.66 GFLOPS | Progress: (40/40) | 33.86 s Done.
# [Task 10/25]  Current/Best:   51.84/ 271.38 GFLOPS | Progress: (40/40) | 32.21 s Done.
# [Task 11/25]  Current/Best:  107.75/ 279.97 GFLOPS | Progress: (40/40) | 32.50 s Done.
# [Task 12/25]  Current/Best:   45.84/ 279.36 GFLOPS | Progress: (40/40) | 27.90 s Done.
# [Task 13/25]  Current/Best:   13.13/ 290.12 GFLOPS | Progress: (40/40) | 30.27 s Done.
# [Task 14/25]  Current/Best:   80.45/ 286.30 GFLOPS | Progress: (40/40) | 34.34 s Done.
# [Task 15/25]  Current/Best:  148.05/ 269.00 GFLOPS | Progress: (40/40) | 33.02 s Done.
# [Task 16/25]  Current/Best:   19.15/ 272.32 GFLOPS | Progress: (40/40) | 25.57 s Done.
# [Task 17/25]  Current/Best:  185.72/ 246.04 GFLOPS | Progress: (40/40) | 26.07 s Done.
# [Task 18/25]  Current/Best:   99.76/ 277.22 GFLOPS | Progress: (40/40) | 27.72 s Done.
# [Task 19/25]  Current/Best:  122.83/ 302.17 GFLOPS | Progress: (40/40) | 28.13 s Done.
# [Task 20/25]  Current/Best:   76.28/ 297.57 GFLOPS | Progress: (40/40) | 33.14 s Done.
# [Task 21/25]  Current/Best:  105.38/ 277.91 GFLOPS | Progress: (40/40) | 33.50 s Done.
# [Task 22/25]  Current/Best:   79.07/ 290.75 GFLOPS | Progress: (40/40) | 31.35 s Done.
# [Task 23/25]  Current/Best:   24.74/ 318.75 GFLOPS | Progress: (40/40) | 30.10 s Done.
# [Task 25/25]  Current/Best:    6.31/  56.30 GFLOPS | Progress: (40/40) | 53.52 s Done.


#-----验证输出-----
tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz \
resnet50-v2-7-tvm_autotuned.tar

#-----性能比较-----

tvmc run\
--inputs imagenet_cat.npz\
--output predictions.npz\
--print-time\
--repeat 100\
resnet50-v2-7-tvm_autotuned.tar

# Execution time summary:
# mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
# 30.2568      30.1431       34.4270      28.8275       0.9967

tvmc run\
--inputs imagenet_cat.npz\
--output predictions.npz\
--print-time\
--repeat 100\
resnet50-v2-7-tvm.tar

# Execution time summary:
# mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
# 45.6399      45.6161       50.4800      43.6862       1.0352