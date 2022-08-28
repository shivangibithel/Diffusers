from transformers import AutoFeatureExtractor, SwinModel
from transformers import AutoFeatureExtractor, AutoModel
from transformers import ViTFeatureExtractor, ViTModel
import torch
from PIL import Image
import requests
import os
import shutil
import numpy as np

src = "./images_5k"

filenames = open("./test.txt").read().split('\n')

# feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
# model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")

feature_extractor = AutoFeatureExtractor.from_pretrained("flyswot/convnext-tiny-224_flyswot")
model = AutoModel.from_pretrained("flyswot/convnext-tiny-224_flyswot")

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

test_pooler_output=[]
cnt = 0
for name in filenames:
    if name:  # to skip empty lines
        name = name.split(".")[0]
        fullpath = os.path.join(src, name + '.png')
        print('Generating features for:', fullpath)
        cnt +=1
        print(cnt)
        image = Image.open(fullpath)
        image = image.convert('RGB')
        inputs = feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        pooler_output = outputs.pooler_output
        pooler_output = pooler_output.cpu().detach().numpy()
        test_pooler_output.append(pooler_output[0])

test_pooler_output = np.array(test_pooler_output)
# np.save("test_pooler_output_image_swin_mscoco_diffuser_images.npy",test_pooler_output)
# np.save("test_pooler_output_image_covnext_mscoco_diffuser_images.npy",test_pooler_output)
np.save("test_pooler_output_image_vit_mscoco_diffuser_images.npy",test_pooler_output)
