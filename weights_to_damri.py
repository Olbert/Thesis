
import torch
from DomainVis.unet.model.unet_damri import UNet2D




model_path = 'E:\\Thesis\\gdrive\\model\\ge_15_CP_epoch200.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet2D(n_chans_in=1, n_chans_out=1)
net.load_state_dict(torch.load(model_path, map_location=device))

net.to(device=device)

torch.save(net, 'ge_15_CP_epoch200.pt')