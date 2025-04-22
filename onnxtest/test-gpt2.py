#%%
import numpy as np
import torch.onnx


#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

#%%

for name, module in model.named_modules():
    print(name, "->", module)
print("===========================")
# 查看参数名
for name, param in model.named_parameters():
    print(name, param.shape)

#%%
target_module = None
target_name = "transformer.h.0.mlp"  # 你想要的层名称
for name, module in model.named_modules():
    if target_name in name:
        target_module = module
        print(f"找到目标模块: {name}")
        break



# 查看具体层的参数
#print(model.state_dict()["transformer.h.0.attn.c_proj.weight"])
#%%
prompt = "hello world, my name is"
input_ids = tokenizer(prompt, return_tensors="pt"). input_ids
#print(input_ids)
outputs = model(input_ids)
print(outputs.logits.shape)
# %%
dummy_input = torch.randn(1, 1, 768)  # 假设输入的形状是 (batch_size, sequence_length, hidden_size)
#%%
torch.onnx.export(target_module,               # model being run
                  dummy_input,                         # model input (or a tuple for multiple inputs)
                  "gpt2_FFNlayer.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=15,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  export_modules_as_functions=True,
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size',1 : 'sequence'},    # variable length axes
                                'output' : {0 : 'batch_size',1 : 'sequence'}})
