#%%
import numpy as np
import torch.onnx
import torch


#%%
import sys
sys.path.append('/Users/xiongzijian/transformers/DeepSeek-VL')
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

#%%
# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).eval()

#%%
conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>Describe each stage of this image.",
        "images": ["/Users/xiongzijian/transformers/DeepSeek-VL/images/training_pipelines.jpg"]
    },
    {
        "role": "Assistant",
        "content": ""
    }
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True
).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

#%%
# run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)

#%%

for name, module in vl_gpt.named_modules():
    print(name, "->", module)
print("===========================")
# 查看参数名
# for name, param in vl_gpt.named_parameters():
#     print(name, param.shape)

#%%
target_module = None
target_name = "language_model.model.layers.0.mlp"  # 你想要的层名称
for name, module in vl_gpt.named_modules():
    if target_name in name:
        target_module = module
        print(f"找到目标模块: {name}")
        break



# 查看具体层的参数
#print(model.state_dict()["transformer.h.0.attn.c_proj.weight"])
# %%
dummy_input = torch.randn(1, 1, 4096)  # 假设输入的形状是 (batch_size, sequence_length, hidden_size)
#%%
torch.onnx.export(target_module,               # model being run
                  dummy_input,                         # model input (or a tuple for multiple inputs)
                  "deepseekv1_FFNlayer.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=15,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  export_modules_as_functions=True,
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                #   dynamic_axes={'input' : {0 : 'batch_size',1 : 'sequence'},    # variable length axes
                #                 'output' : {0 : 'batch_size',1 : 'sequence'}}
                                 )

# %%
