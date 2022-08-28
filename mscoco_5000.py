import pickle
with open('list_prompts_clean.pkl', 'rb') as f:
    prompts = pickle.load(f)
access_token = "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"
# from huggingface_hub import notebook_login
# notebook_login()
print(prompts[0])
import torch
# print(torch.cuda.device_count())
# torch.cuda.set_device(2)
# print(torch.cuda.current_device())
from diffusers import StableDiffusionPipeline
from torch import autocast
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",use_auth_token=access_token)  
print("Model downloaded")
pipe = pipe.to("cuda:2")

f = open('test.txt','r')
lines = f.read()
lines = lines.split("\n")
lines[0]

cnt = 0
for i in range(len(prompts)):
    cnt+=1
    prompt = prompts[i] 
    file_name = lines[i]
    with autocast("cuda"):
        image = pipe(prompt)["sample"][0]
        f_name = file_name.split(".")[0]+".png"
        image.save(f_name)
        print(cnt)
        # break

