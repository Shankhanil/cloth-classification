from torchvision.models import efficientnet_v2_s #, EfficientNet_V2_M_Weights
from torchvision import transforms
from PIL import Image
from torch.nn import Sequential, Dropout, Linear
import torch
import numpy as np

import time
start_time = time.time()

classification_model = efficientnet_v2_s()

classification_model.classifier = Sequential(
        Dropout(p=0.4, inplace=True),
        Linear(in_features=1280, out_features=3, bias=True)
    )



checkpoint = torch.load('checkpoints/epoch_19.pt')
classification_model.load_state_dict(checkpoint['model_state_dict'])
classification_model.eval()

tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

image1 = 'data/Polo_Tshirt/2258.jpg'
image_transformed = tfms(Image.open(image1)).unsqueeze(0)

with torch.no_grad():
        outputs = classification_model(image_transformed)
        pred_class = np.argmax(torch.softmax(outputs[0], dim=-1).detach().numpy())
        pred = np.max(torch.softmax(outputs[0], dim=-1).detach().numpy())

print(pred_class, pred)



image = 'data/T-shirt/0189955076.jpg'
image_transformed = tfms(Image.open(image)).unsqueeze(0)

with torch.no_grad():
        outputs = classification_model(image_transformed)
        pred_class = np.argmax(torch.softmax(outputs[0], dim=-1).detach().numpy())
        pred = np.max(torch.softmax(outputs[0], dim=-1).detach().numpy())

print(pred_class, pred)

print(f'Total time taken = {time.time() - start_time}')