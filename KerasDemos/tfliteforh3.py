# https://zhuanlan.zhihu.com/p/458001661
# https://www.jianshu.com/p/2be50510670d
# 此處使用tflite來進行預測

from tflite_runtime.interpreter import Interpreter

args_model = 'KerasRnn.tflite'
interpreter = Interpreter(args_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# print('input_details：\n ', input_details)
# print('output_details：\n ', output_details)
print("hello")