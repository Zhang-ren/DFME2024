import torch
import sys
sys.path.append("../Model/")

from MobileViT import models


# def load_mobilevit_weights(model_path = "../Model/MobileViT/xxsmodel_best.pth.tar"):
  
def load_mobilevit_weights(mode = 'XXS'):
  # Create an instance of the MobileViT model
  if mode == 'S':
    net = models.MobileViT_S(num_classes = 7)
    state_dict = torch.load("../Model/MobileViT/model_best.pth.tar", map_location=torch.device('cpu'))['state_dict']
  else:
    net = models.MobileViT_XXS(num_classes = 7)
    state_dict = torch.load("../Model/MobileViT/xxsmodel_best.pth.tar", map_location=torch.device('cpu'))['state_dict']
  # Load the PyTorch state_dict

  # Since there is a problem in the names of layers, we will change the keys to meet the MobileViT model architecture
  for key in list(state_dict.keys()):
    # state_dict[key.replace('module.', '')] = state_dict.pop(key)
    new_key = key.replace('module.', '')
    if 'fc.weight' in new_key or 'fc.bias' in new_key:
        continue  # Skip loading fc layer weights

  # Once the keys are fixed, we can modify the parameters of MobileViT
  # net.load_state_dict(state_dict)
    state_dict[new_key] = state_dict.pop(key)

  return net
if __name__ == "__main__":
    img = torch.randn(1, 3, 256, 256)
    net= load_mobilevit_weights('S')
    x, fea= net(img)


    # XXS: 1.3M 、 XS: 2.3M 、 S: 5.6M
    print("MobileViT-S params: ", sum(p.numel() for p in net.parameters()))
    print(f"Output shape: {net(img)[1].shape}")