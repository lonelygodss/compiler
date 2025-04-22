#%%
import onnx
import netron
from onnxsim import simplify

# 加载ONNX模型
onnx_model = onnx.load("gpt2_FFNlayer.onnx")

# 检查模型格式是否正确
onnx.checker.check_model(onnx_model)

#%%

model_simp, check = simplify(onnx_model)

assert check, "Simplified ONNX model could not be validated"
# 打印模型信息
#print(onnx.helper.printable_graph(onnx_model.graph))
# %%
onnx.save_model(model_simp, "gpt2_FFNsimplified.onnx")
# %%
