import os
from os import listdir
from os.path import isfile, join
from os import walk

import numpy as np
from torch.utils.data import DataLoader

import platform
import torch
import torch.nn.functional as F


from DomainVis.unet.dice_loss import dice_coeff
from DomainVis.unet.model import UNet2D
from DomainVis.database_process.dataset import BasicDataset, H5Dataset
import surface_distance


if platform.system()=='Windows': n_cpu= 0
def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_chans_out == 1 else torch.long
    # TODO: redefine length
    n_val = len(loader)  # the number of batch
    tot = 0

    # with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        if net.n_chans_out > 1:
            tot += F.cross_entropy(mask_pred, true_masks).item()
        else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()
        # pbar.update()

    net.train()
    return tot / n_val



if __name__ == '__main__':
    model_list = [
        'philips_15',
        'philips_3',
        'siemens_15',
        'siemens_3',
        'ge_15',
        'ge_3'
    ]
    name_list = [
        'philips_15',
        'philips_3',
        'siemens_15',
        'siemens_3',
        'ge_15',
        'ge_3'
    ]

    for model in model_list:
        dir_img = 'E:\\Thesis\\gdrive\\test'

        """ Net setup """
        net = UNet2D(n_chans_in=1, n_chans_out=1)
        net.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device='cuda')
        model_path = os.path.join('E:\Thesis\gdrive\model', model+'_CP_epoch200.pth')
        net.load_state_dict(torch.load(model_path, map_location=device))
        slices = 40
        img_size = (128, 128)


        onlyfiles = [f for f in listdir(dir_img) if isfile(join(dir_img, f))]

        dirnames = np.array(walk(dir_img).__next__()[1])

        test_loaders = []


        # print("Dice score: ")
        # for name in dirnames:
        #     dataset = BasicDataset(os.path.join(dir_img, name), os.path.join(dir_mask, name), slices, img_size, '_ss')
        #     test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        #     val_score = eval_net(net, test_loader, device)
        #
        #     print(str(name) + ': ' + str(val_score))


        print("Surface distance: ")
        print('======================================================')
        print(str(model))
        for name in name_list:
            # dataset = BasicDataset(os.path.join(dir_img, name), os.path.join(dir_mask, name), slices, img_size, '_ss')
            # dataset = BasicDataset(os.path.join(dir_img, name, 'images/'), os.path.join(dir_img, name, 'masks/'), slices,
            #                        img_size, '_ss')
            dataset = H5Dataset(name,
                                os.path.join(dir_img, name, 'data.h5'),
                                os.path.join(dir_img, name, 'masks.h5'),
                                slices,
                                img_size, '_ss')

            test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
            """Evaluation without the densecrf with the dice coefficient"""
            net.eval()
            mask_type = torch.float32 if net.n_chans_out == 1 else torch.long

            i = 0
            total = 0
            true_masks_all = []
            mask_pred_all = []
            for batch in test_loader:
                imgs, true_masks = batch['image'], batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.cpu().numpy()[0,0]
                true_masks_all.append(true_masks)

                with torch.no_grad():
                    mask_pred = net(imgs)
                    mask_pred = torch.sigmoid(mask_pred)
                    mask_pred = mask_pred.cpu().numpy()[0,0]
                    mask_pred = (mask_pred > 0.5)
                    mask_pred_all.append(mask_pred)

            val_score = surface_distance.compute_surface_distances(np.array(mask_pred_all), np.array(true_masks_all, dtype=bool), [1,1,1])

            dice = surface_distance.compute_surface_dice_at_tolerance(val_score, 1)



            print(str(name) + ': ' + "{:.4f}".format(dice))



"""
Surface distance: 2mm philips_15 2D
ge_15: 0.7882200408190674
ge_3: 0.6297838360700679
philips_15: 0.87457095303606
philips_3: 0.2743725966714217
siemens_15: 0.7327510562328482
siemens_3: 0.8081202860991571
"""


"""
Surface distance: 1mm philips_15 2D
ge_15: 0.4241895847958042
ge_3: 0.3949352637951144
philips_15: 0.35987503172707885
philips_3: 0.062380443667114455
siemens_15: 0.37092389382350954
siemens_3: 0.45467486734707013
"""

""" 1mm philips_15 3D
Surface distance: 
ge_15: 0.6877610776228444
ge_3: 0.7427685547441867
philips_15: 0.7324438841707778
philips_3: 0.24514733242980716
siemens_15: 0.6734573380780277
siemens_3: 0.717354212164526
"""


""" 1mm philips_15 3D multiple
ge_15: 0.6291184916710084
ge_3: 0.9012111533221929
philips_15_test: 0.7968217265060121
philips_15_train: 0.8302636824647416
philips_3: 0.22820770862443862
siemens_15: 0.8306324651525312
siemens_3: 0.8582756925217326
"""
"""
philips_15: 0.9756
philips_3: 0.8155
siemens_15: 0.9579
siemens_3: 0.9699
ge_15: 0.8119
ge_3: 0.9496



Surface distance: 
======================================================
philips_3
philips_15: 0.9633001871279359
philips_3: 0.9553448670181977
siemens_15: 0.9529311648983847
siemens_3: 0.9667813244840272
ge_15: 0.9520132064098863
ge_3: 0.94513814130654
Surface distance: 
======================================================
siemens_15
philips_15: 0.9471269728870477
philips_3: 0.7626928510453174
siemens_15: 0.9835990961912362
siemens_3: 0.9389154154161927
ge_15: 0.7958938027511182
ge_3: 0.9531130859177054
Surface distance: 
======================================================
siemens_3
philips_15: 0.9059521975281931
philips_3: 0.8092337568480817
siemens_15: 0.8668865031537654
siemens_3: 0.9858939674654495
ge_15: 0.7857840548667986
ge_3: 0.8479102991347846
Surface distance: 
======================================================
ge_15
philips_15: 0.899452413967345
philips_3: 0.673812138765158
siemens_15: 0.9307572737858021
siemens_3: 0.8852182133030649
ge_15: 0.9795070738824554
ge_3: 0.8245024170269162
Surface distance: 
======================================================
ge_3
philips_15: 0.6157947823526987
philips_3: 0.5776572333454136
siemens_15: 0.7199837571527964
siemens_3: 0.8773342651167956
ge_15: 0.5003680749582881
ge_3: 0.9849115608785557


"""











