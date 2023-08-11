from tvm.driver import tvmc

model = tvmc.load('my_model.onnx') #Step 1: Load

def compile_run():
    # pytorch_model需要指定shape_dict，因为tvm无法搜索它
    # model = tvmc.load('my_model.onnx', shape_dict={'input1' : [1, 2, 3, 4], 'input2' : [1, 2, 3, 4]}) #Step 1: Load + shape_dict

    package = tvmc.compile(model, target="llvm")  # Step 2: Compile

    result = tvmc.run(package, device="cpu")  # Step 3: Run

    print(result)

def tune_run():
    tune_record = tvmc.tune(model, target="llvm")  # Step 1.5: Optional Tune
    tvmc.compile(model, target="llvm", tuning_records = "records.log") #Step 2: Compile


if __name__ == '__main__':
    tune_run()