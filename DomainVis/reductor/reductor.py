import os
import numpy as np

import torch

from DomainVis.unet.model import UNet2D

from DomainVis.reductor.reductor_algos import TSNE, PCA, LLE, Isomap, PCA_cuda

import DomainVis.reductor.reductor_utils as reductor_utils

from os import makedirs, path
import plotly.graph_objects as go

np.random.seed(seed=0)

torch.random.manual_seed(seed=0)
activation = {}
weight = {}


def get_map(mask, pred, coeff=0.5):  # TODO: check for errors
    """
        0: True Positive
        1: True Negative
        2: False Positive
        3: False Negative
    """

    eval_map = np.zeros(mask.shape)
    for i in range(0, mask.shape[0]):
        for k in range(0, mask.shape[1]):
            if mask[i, k] > 0.5:
                if pred[i, k] > coeff:
                    eval_map[i, k] = 0
                else:
                    eval_map[i, k] = 2
            if mask[i, k] == 0.5:
                if pred[i, k] < coeff:
                    eval_map[i, k] = 1
                else:
                    eval_map[i, k] = 3

    # plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    # plt.show()
    # plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
    # plt.show()
    # plt.imshow(eval_map, vmin=2, vmax=3)
    # plt.show()

    return eval_map


class BasePresenter():

    def get_database(self, net, test_loaders, layer_name, threshold, img_size, mask_cut, upsampling, plot_mode):

        try:
            self.image_mids = np.load('data/net' + str(layer_name) + 'eval_map.npy')
            self.eval_maps = np.load('data/net/eval_maps.npy')
            self.input = np.load('data/net/input.npy')
            self.domains = np.load('data/net/domains.npy')
        except:
            for test_loader in test_loaders:
                image_mids_temp = []
                eval_maps_temp = []
                true_masks = []
                input_temp = []
                # self.domains.append(test_loader[1])
                # test_loader = test_loader[0]
                batch_id = 0  # TODO: Find better solution for iteration
                for batch in test_loader:
                    image, true_mask = batch['image'], batch['mask']
                    # image = image.reshape(image.shape[0], 1, image.shape[1], image.shape[2])
                    # true_mask = true_mask.reshape(true_mask.shape[0], 1, true_mask.shape[1], true_mask.shape[2])
                    true_mask = torch.sigmoid(true_mask)
                    true_mask = true_mask.cpu().numpy()[0, 0]
                    true_masks.append(true_mask)

                    if image == 'input_image':
                        image_mid = image
                        eval_map = []
                    else:
                        image_mid, net_output = reductor_utils.get_mid_output(net, image, layer_name)

                        if threshold is not None:
                            image_mid = reductor_utils.threshold_transform(image_mid, threshold)
                        """Evaluation map preparation"""

                        if image_mid.shape[1] != true_mask.shape[0]:
                            true_mask = reductor_utils.resize_image(true_mask, (image_mid.shape[1], image_mid.shape[2]))

                        if image_mid.shape[1] != net_output.shape[0]:
                            net_output = reductor_utils.resize_image(net_output, (image_mid.shape[1], image_mid.shape[2]))

                        eval_map = get_map(true_mask, net_output)

                        if img_size is not None:
                            if image_mid.shape[1] > img_size[0] | upsampling:
                                # if upsampling mode  or initial image is bigger than requested
                                image_mid = reductor_utils.resize_image(image_mid, img_size)
                                eval_map = reductor_utils.resize_image(eval_map, img_size)
                                true_mask = reductor_utils.resize_image(true_mask, img_size)

                        if mask_cut == 'true':
                            image_mid = reductor_utils.crop_mask(image_mid, true_mask)
                        elif mask_cut == 'predict':
                            image_mid = reductor_utils.crop_mask(image_mid, net_output)

                    """Plot mode"""
                    if batch_id % plot_mode == 0:
                        image_mids_temp.append(image_mid / plot_mode)
                    else:
                        image_mids_temp[-1] += image_mid / plot_mode

                    eval_maps_temp.append(eval_map)  # no plot_mode considered
                    input_temp.append(image.numpy())
                    batch_id += 1

                image_mids_temp = np.array(image_mids_temp)
                eval_maps_temp = np.array(eval_maps_temp)
                input_temp = np.array(input_temp)

                self.image_mids.append(image_mids_temp)
                self.eval_maps.append(eval_maps_temp)
                self.input.append(input_temp)

            self.image_mids = np.array(self.image_mids)
            self.eval_maps = np.array(self.eval_maps)
            self.input = np.array(self.input)
            self.domains = np.array(self.domains)

        np.save('data/net/' + str(layer_name) + 'eval_map', self.image_mids)
        np.save('data/net/eval_maps', self.eval_maps)
        np.save('data/net/input', self.input)
        np.save('data/net/domains', self.domains)



class Reductor(BasePresenter):
    def __init__(self,
                 model_path="E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Train2\checkpoints\CP_epoch200.pth",
                 domains=[],
                 full_model=True):

        """Parameters Init"""

        self.image_mids = []
        self.eval_maps = []
        self.input = []
        self.output = []
        self.input_img_size = (128, 128)

        """ Net setup """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if full_model:
            self.net = torch.load(model_path)
        else:
            self.net = UNet2D(n_chans_in=1, n_chans_out=1)
            self.net.load_state_dict(torch.load(model_path, map_location=device))

        self.net.to(device=device)

        self.domains = domains

    def reduction(self, algos, modes, layer_name, threshold, img_size, volume_num, sample_num, mask_cut, perp=None,
                  neighbours=None,
                  n_iter=500, save=False, data_path="", mask_path=""):

        upsampling = False

        self.activ_img_size = img_size
        img_types = ['TP', 'TN', 'FP', 'FN']

        """Plot mode structure """  # number of images to stack on each other (Doesn't work)
        plot_mode = 1
        desired_points = 200000  # Works not everywhere
        test_loaders = reductor_utils.get_data(self.domains, volume_num, sample_num, data_path=data_path,
                                               mask_path=mask_path)

        self.get_database(self.net, test_loaders, layer_name, threshold, img_size, mask_cut, upsampling, plot_mode)

        """ Dimensionality reduction """
        try:
            # if True:
            for algo in algos:
                for mode in modes:
                    if algo == 'tsne':
                        orig_shape = self.image_mids.shape
                        if perp is None:
                            if mode == 'pixel':
                                perp_range = range(1,
                                                   self.image_mids.swapaxes(2, -1).reshape(-1, orig_shape[2]).shape[0],
                                                   int(
                                                       np.around(
                                                           self.image_mids.swapaxes(2, -1).reshape(-1,
                                                                                                   orig_shape[2]).shape[
                                                               0] / 10)))
                            else:
                                perp_range = range(1,
                                                   self.image_mids.swapaxes(2, 0).reshape(orig_shape[0] * orig_shape[2],
                                                                                          -1).shape[1], int(
                                        np.around(
                                            self.image_mids.swapaxes(2, 0).reshape(orig_shape[0] * orig_shape[2],
                                                                                   -1).shape[
                                                1] / 10)))
                        else:
                            perp_range = range(perp, perp + 1, 1)
                        for perp in perp_range:
                            # range(perp_range[mode][0], perp_range[mode][1], perp_range[mode][2]):
                            fig = go.Figure()

                            all_outputs = []  # np.zeros([image_mids.shape[0], img_size[0] * img_size[1]], dtype=np.float)
                            self.output = []

                            orig_shape = self.image_mids.shape
                            tsne = TSNE(mode, perp, option='std', n_iter=n_iter)
                            try:
                                if mode == 'pixel':
                                    # 	tsne.transform(image_mids[l, 0].reshape(shape[1], shape[2], shape[3]), save=False)
                                    tsne.transform(self.image_mids.swapaxes(2, -1).reshape(-1, orig_shape[2]))
                                else:
                                    tsne.transform(
                                        self.image_mids.swapaxes(2, 0).reshape(orig_shape[0] * orig_shape[2], -1))

                                if mode == 'pixel':
                                    outputs = tsne.outputs[0].reshape(orig_shape[0], orig_shape[1], orig_shape[3],
                                                                      orig_shape[4], 2)
                                    for dom in range(0, orig_shape[0]):
                                        for i in range(0, 4):
                                            output = outputs[dom, self.eval_maps[dom] == i]
                                            outputs_subset = np.random.choice(range(len(output)),
                                                                              min(desired_points, len(output)),
                                                                              replace=False)
                                            if len(outputs_subset > 0):
                                                fig.add_trace(go.Scatter(x=output[outputs_subset, 0],
                                                                         y=output[outputs_subset, 1],
                                                                         name=self.domains[dom] + ' ' + img_types[i],
                                                                         mode='markers',
                                                                         opacity=0.7))

                                        all_outputs.append(outputs[dom])
                                else:

                                    outputs = tsne.outputs[0].reshape(orig_shape[0], orig_shape[2], -1)
                                    for dom in range(0, orig_shape[0]):
                                        outputs_subset = np.random.choice(range(len(outputs[dom])),
                                                                          min(desired_points, len(outputs[dom])),
                                                                          replace=False)
                                        fig.add_trace(go.Scatter(x=outputs[dom, outputs_subset][:, 0],
                                                                 y=outputs[dom, outputs_subset][:, 1],
                                                                 name=self.domains[dom], mode='markers',
                                                                 opacity=0.7))

                                        all_outputs.append(outputs[dom, outputs_subset])
                                # plt.scatter(outputs[0][:, 0], outputs[0][:, 1])
                                # plt.show()

                                if save:
                                    np.save("data/graphs/tsne/" + str(layer_name) + "_mode_" + str(
                                        tsne.mode) + "_thresh_" + str(
                                        threshold) + "_cut_" + str(mask_cut) + "_perp_" + str(
                                        tsne.perp) + str(img_size), np.array(all_outputs))
                                    fig.write_image(
                                        "data/graphs/tsne/" + str(layer_name) + "_mode_" + str(
                                            tsne.mode) + "_thresh_" + str(
                                            threshold) + "_cut_" + str(mask_cut) + "_perp_" + str(
                                            tsne.perp) + str(img_size) + ".jpg")

                                self.output = np.array(all_outputs)
                            except:
                                pass
                    if algo == 'pca':
                        fig = go.Figure()
                        all_outputs = []
                        self.output = []
                        fit = True
                        pca = PCA(mode, fit=fit, n_components=min(2, self.image_mids[0].shape[1]))

                        for dom in range(0, self.image_mids.shape[0]):
                            shape = self.image_mids[dom].shape
                            # TODO: Check how it is better to feed samples, together or separately
                            if mode == 'pixel':
                                pca.transform(self.image_mids[dom].swapaxes(1, -1).reshape(-1, shape[1]))
                            else:
                                pca.transform(self.image_mids[dom].reshape(shape[0] * shape[1], -1))
                            pca.fit = False
                            outputs = np.array(pca.outputs[dom]).reshape(-1, 2)
                            outputs_subset = np.random.choice(range(outputs.shape[0]),
                                                              min(desired_points, outputs.shape[0]),
                                                              replace=False)
                            fig.add_trace(go.Scatter(x=outputs[outputs_subset][:, 0],
                                                     y=outputs[outputs_subset][:, 1], name=self.domains[dom],
                                                     mode='markers', opacity=0.7))
                            if mode == 'pixel':
                                all_outputs.append(outputs[outputs_subset].reshape(shape[0], shape[2], shape[3], 2))
                            else:
                                all_outputs.append(outputs[outputs_subset].reshape(-1, 2))

                        if save:
                            fig.write_image(
                                "data/graphs/pca/" + str(layer_name) + "_mode_" + str(pca.mode) + "_thresh_" + str(
                                    threshold) + "_cut_" + str(mask_cut) + str(img_size) + ".jpg")
                            np.save(
                                "data/graphs/pca/" + str(layer_name) + "_mode_" + str(pca.mode) + "_thresh_" + str(
                                    threshold) + "_cut_" + str(mask_cut) + str(img_size), np.array(all_outputs))
                        self.output = np.array(all_outputs)

                    if algo == 'isomap':
                        if neighbours is None:
                            neighbours = 5
                        fig = go.Figure()

                        all_outputs = []
                        self.output = []
                        fit = True
                        isomap = Isomap(mode, fit=fit, n_components=min(2, self.image_mids[0].shape[1]),
                                        n_neighbors=neighbours)
                        for dom in range(0, self.image_mids.shape[0]):
                            shape = self.image_mids[dom].shape

                            if mode == 'pixel':
                                isomap.transform(self.image_mids[dom].swapaxes(1, -1).reshape(-1, shape[1]))
                            else:
                                isomap.transform(self.image_mids[dom].reshape(shape[0] * shape[1], -1))

                            isomap.fit = False
                            outputs = np.array(isomap.outputs[dom]).reshape(-1, 2)
                            outputs_subset = np.random.choice(range(outputs.shape[0]),
                                                              min(desired_points, outputs.shape[0]),
                                                              replace=False)
                            fig.add_trace(go.Scatter(x=outputs[outputs_subset][:, 0],
                                                     y=outputs[outputs_subset][:, 1], name=self.domains[dom],
                                                     mode='markers', opacity=0.7))
                            if mode == 'pixel':
                                all_outputs.append(outputs[outputs_subset].reshape(shape[0], shape[2], shape[3], 2))
                            else:
                                all_outputs.append(outputs[outputs_subset].reshape(-1, 2))

                        if save:
                            fig.write_image(
                                "data/graphs/isomap/" + str(layer_name) + "_mode_" + str(
                                    isomap.mode) + "_thresh_" + str(
                                    threshold) + "_cut_" + str(mask_cut) + "_n_" + str(neighbours) + str(
                                    img_size) + ".jpg")
                            np.save(
                                "data/graphs/isomap/" + str(layer_name) + "_mode_" + str(
                                    isomap.mode) + "_thresh_" + str(
                                    threshold) + "_cut_" + str(mask_cut) + "_n_" + str(neighbours) + str(img_size),
                                np.array(all_outputs))

                        self.output = np.array(all_outputs)

                    if algo == 'lle':
                        if neighbours is None:
                            neighbours = 5
                        fig = go.Figure()

                        all_outputs = []
                        self.output = []
                        fit = True
                        lle = LLE(mode, fit=fit, n_components=min(2, self.image_mids[0].shape[1]),
                                  n_neighbors=neighbours)
                        for dom in range(0, self.image_mids.shape[0]):
                            shape = self.image_mids[dom].shape

                            if mode == 'pixel':
                                lle.transform(self.image_mids[dom].swapaxes(1, -1).reshape(-1, shape[1]))
                            else:
                                lle.transform(self.image_mids[dom].reshape(shape[0] * shape[1], -1))
                            lle.fit = False
                            outputs = np.array(lle.outputs[dom]).reshape(-1, 2)
                            outputs_subset = np.random.choice(range(outputs.shape[0]),
                                                              min(desired_points, outputs.shape[0]),
                                                              replace=False)
                            fig.add_trace(go.Scatter(x=outputs[outputs_subset][:, 0],
                                                     y=outputs[outputs_subset][:, 1], name=self.domains[dom],
                                                     mode='markers', opacity=0.7))
                            if mode == 'pixel':
                                all_outputs.append(outputs[outputs_subset].reshape(shape[0], shape[2], shape[3], 2))
                            else:
                                all_outputs.append(outputs[outputs_subset].reshape(-1, 2))

                        if save:
                            fig.write_image(
                                "data/graphs/lle/" + str(layer_name) + "_mode_" + str(lle.mode) + "_thresh_" + str(
                                    threshold) + "_cut_" + str(mask_cut) + "_n_" + str(neighbours) + str(
                                    img_size) + ".jpg")
                            np.save(
                                "data/graphs/lle/" + str(layer_name) + "_mode_" + str(lle.mode) + "_thresh_" + str(
                                    threshold) + "_cut_" + str(mask_cut) + "_n_" + str(neighbours) + str(img_size),
                                np.array(all_outputs))

                        self.output = np.array(all_outputs)

                    if algo == 'full':
                        orig_shape = self.image_mids.shape
                        if perp is None:
                            if mode == 'pixel':
                                perp_range = range(1,
                                                   self.image_mids.swapaxes(2, -1).reshape(-1, orig_shape[2]).shape[0],
                                                   int(
                                                       np.around(
                                                           self.image_mids.swapaxes(2, -1).reshape(-1,
                                                                                                   orig_shape[2]).shape[
                                                               0] / 10)))
                            else:
                                perp_range = range(1,
                                                   self.image_mids.swapaxes(2, 0).reshape(orig_shape[0] * orig_shape[2],
                                                                                          -1).shape[1], int(
                                        np.around(
                                            self.image_mids.swapaxes(2, 0).reshape(orig_shape[0] * orig_shape[2],
                                                                                   -1).shape[
                                                1] / 10)))
                        else:
                            perp_range = range(perp, perp + 1, 1)
                        for perp in perp_range:
                            # range(perp_range[mode][0], perp_range[mode][1], perp_range[mode][2]):

                            fig = go.Figure()
                            all_outputs = []  # np.zeros([image_mids.shape[0], img_size[0] * img_size[1]], dtype=np.float)
                            self.output = []

                            orig_shape = self.image_mids.shape
                            tsne = TSNE(mode, perp, n_iter=n_iter, init='pca')
                            if mode == 'pixel':
                                # 	tsne.transform(image_mids[l, 0].reshape(shape[1], shape[2], shape[3]), save=False)
                                tsne.transform(self.image_mids.swapaxes(2, -1).reshape(-1, orig_shape[2]))
                            else:
                                tsne.transform(
                                    self.image_mids.swapaxes(2, 0).reshape(orig_shape[0] * orig_shape[2], -1))

                            if mode == 'pixel':
                                outputs = tsne.outputs[0].reshape(orig_shape[0], orig_shape[1], orig_shape[3],
                                                                  orig_shape[4], 2)
                                for dom in range(0, orig_shape[0]):
                                    for i in range(0, 4):
                                        output = outputs[dom, self.eval_maps[dom] == i]
                                        outputs_subset = np.random.choice(range(len(output)),
                                                                          min(desired_points, len(output)),
                                                                          replace=False)
                                        if len(outputs_subset > 0):
                                            fig.add_trace(go.Scatter(x=output[outputs_subset, 0],
                                                                     y=output[outputs_subset, 1],
                                                                     name=self.domains[dom] + ' ' + img_types[i],
                                                                     mode='markers',
                                                                     opacity=0.7))
                                    all_outputs.append(np.concatenate([outputs[dom, self.eval_maps[dom] == 0],
                                                                       outputs[dom, self.eval_maps[dom] == 2],
                                                                       outputs[dom, self.eval_maps[dom] == 1],
                                                                       outputs[dom, self.eval_maps[dom] == 3]]))
                            else:
                                # TODO: subset length is wrong
                                outputs = tsne.outputs[0].reshape(orig_shape[0], orig_shape[2], -1)
                                for dom in range(0, orig_shape[0]):
                                    outputs_subset = np.random.choice(range(len(outputs[dom])),
                                                                      min(desired_points, len(outputs[dom])),
                                                                      replace=False)
                                    fig.add_trace(go.Scatter(x=outputs[dom, outputs_subset][:, 0],
                                                             y=outputs[dom, outputs_subset][:, 1],
                                                             name=self.domains[dom],
                                                             mode='markers',
                                                             opacity=0.7))

                                    all_outputs.append(outputs[dom, outputs_subset])

                            if save:
                                fig.write_image(
                                    "data/graphs/full/" + str(layer_name) + "_mode_" + str(
                                        tsne.mode) + "_thresh_" + str(
                                        threshold) + "_cut_" + str(mask_cut) + "_perp_" + str(
                                        tsne.perp) + str(img_size) + ".jpg")
                                np.save(
                                    "data/graphs/full/" + str(layer_name) + "_mode_" + str(
                                        tsne.mode) + "_thresh_" + str(
                                        threshold) + "_cut_" + str(mask_cut) + "_perp_" + str(
                                        tsne.perp) + str(img_size), np.array(all_outputs))

                            self.output = np.array(all_outputs)

                    if algo == 'pca_cuda':
                        fig = go.Figure()

                        all_outputs = []
                        self.output = []
                        fit = True
                        pca = PCA_cuda(mode, fit=fit, n_components=min(2, self.image_mids[0].shape[1]))
                        for dom in range(0, self.image_mids.shape[0]):
                            shape = self.image_mids[dom].shape
                            # TODO: Check how it is better to feed samples, together or separately
                            pca.transform(self.image_mids[dom].swapaxes(1, -1).reshape(-1, shape[1]))
                            pca.fit = False
                            outputs = np.array(pca.outputs[dom]).reshape(-1, 2)
                            outputs_subset = np.random.choice(range(outputs.shape[0]),
                                                              min(desired_points, outputs.shape[0]),
                                                              replace=False)
                            fig.add_trace(go.Scatter(x=outputs[outputs_subset][:, 0],
                                                     y=outputs[outputs_subset][:, 1], name=self.domains[dom],
                                                     mode='markers', opacity=0.7))
                            # plt.scatter(x=outputs[outputs_subset][:, 0],
                            #              y=outputs[outputs_subset][:, 1], name=self.domains[dom], mode='markers', opacity=0.7))
                            # plt.show()
                            all_outputs.append(outputs[outputs_subset].reshape(shape[0], shape[2], shape[3], 2))

                        if save:
                            fig.write_image(
                                "data/graphs/pca/" + str(layer_name) + "_mode_" + str(pca.mode) + "_thresh_" + str(
                                    threshold) + "_cut_" + str(mask_cut) + str(img_size) + ".jpg")

                        self.output = np.array(all_outputs)


        except Exception as e:
            print('For ' + str(algo) + '_' + str(layer_name) + "_mode_" + str(mode) + "_thresh_" + str(
                threshold) + "_cut_" + str(mask_cut) + str(img_size) + '_n_' + str(neighbours))
            print(e)
            pass

    def get_data(self):
        keys = ['input', 'output', 'names', 'eval_maps', 'image_mids']
        self.__dict__.pop('net', None)

        return self.__dict__

    @staticmethod
    def get_net(path):
        model = torch.load(path)
        return np.array(list(model.named_children()))[:, 0]

    @staticmethod
    def auto(algos, modes, layer_name, threshold, img_size, volume_num, sample_num, mask_cut,
             perp, neighbours, n_iter, save, path, names, full_model=True, data_path="", mask_path=""):
        names = [
            'philips_15',
            'siemens_15',
            'siemens_3',

        ]
        reductor = Reductor(path, names, full_model)
        reductor.reduction(algos, modes, layer_name, threshold, img_size, volume_num, sample_num, mask_cut,
                           perp, neighbours, n_iter, save, data_path, mask_path)

        return reductor

    @staticmethod
    def precomputed(algos, modes, layer_name, threshold, img_size, volume_num, sample_num, mask_cut,
                    perp, neighbours, n_iter, save, path, names, full_model=True, data_path="", mask_path=""):

        name_list = [
            'philips_15',
            'philips_3',
            'siemens_15',
            'siemens_3',
            'ge_15',
            'ge_3'
        ]
        names_id = []
        for name in names:
            names_id.append(name_list.index(name))

        reductor = Reductor(path, names, full_model)
        folder = "E:\\Thesis\\DomainVis\\reductor\\data"
        reductor.input = np.load(os.path.join(folder, 'input.npy'))[names_id]
        reductor.eval_maps = np.load(os.path.join(folder, layer_name + 'eval_map.npy'))[names_id]
        folder = "E:\\Thesis\\DomainVis\\reductor\\data\\graphs"
        algo = algos[0]
        mode = modes[0]

        if algo == 'tsne' or algo == 'full':
            name = str(layer_name) + "_mode_" + str(mode) + "_thresh_" + \
                   str(threshold) + "_cut_" + str(mask_cut) + "_perp_" + \
                   str(perp) + str(img_size) + '.npy'
        elif algo == 'isomap' or algo == 'lle':
            name = str(layer_name) + "_mode_" + str(mode) + "_thresh_" + \
                   str(threshold) + "_cut_" + str(mask_cut) + "_n_" + \
                   str(neighbours) + str(img_size) + '.npy'
        else:
            name = str(layer_name) + "_mode_" + str(mode) + "_thresh_" + str(
                threshold) + "_cut_" + str(mask_cut) + str(img_size) + '.npy'

        reductor.output = np.load(os.path.join(folder, algo, name))[names_id]

        return reductor


if __name__ == '__main__':
    """ Variables setup """
    layer = 'init_path'

    path = "E:\\Thesis\\gdrive\\model\\siemens_3_CP_epoch200.pt"

    base_path = "E:\\Thesis\\gdrive\\test\\"

    names = [
        'philips_15',
        'philips_3',
        'siemens_15',
        'siemens_3',
        'ge_15',
        'ge_3'
    ]
    data_path = os.path.join('E:\Thesis\DomainVis\server_files\processed', 'full_data.h5')
    mask_path = os.path.join('E:\Thesis\DomainVis\server_files\processed', 'full_masks.h5')

    # data_path_files = []
    # mask_path_files = []
    # for name in names:
    # 	data_path_files.append(os.path.join(base_path,name,'data.h5'))
    # 	mask_path_files.append(os.path.join(base_path, name, 'masks.h5'))
    # 	try:
    # 		makedirs("data/graphs/" + name + "/")
    # 	except:
    # 		pass
    #
    # data_path = H5Dataset.combine(data_path_files, data_path)
    # mask_path = H5Dataset.combine(mask_path_files, mask_path)

    # map = MapPresenter.auto(0, layer, img_size, 0)

    # algos = ['lle','pca','isomap', 'tsne']
    # algos = ['lle','isomap']
    algos = ['pca']
    modes = ['pixel']  # ['feature', 'pixel','pic_to_coord']  # smth else?


    layers = ['up2']
    # layers = ['init_path', 'down1', 'down2', 'down3', 'up1', 'up2','up3']

    thresholds = [None]
    img_sizes = [(128, 128)]

    show_im = False
    crop_im = False
    volume_num = 3
    sample_num = 1
    perp = 1
    # perps = [50,100, 200,5,10,30,3, 25,80]
    save = True
    neighbours = [3, 5, 10, 50, 70, 100]#, 100,150,200]
    neighbours = [35, 40, 45, 80,120,150,200]
    mask_cuts = ['None']  # 'true', 'predict', None
    n_init = 250

    n_iters = [500]

    params = neighbours

    layers = ['init_path', 'down1', 'down2', 'down3', 'up1', 'up2', 'up3']
    algos = ['lle', 'pca', 'isomap']
    params = [1,2,3, 5, 7, 10, 25, 50, 70, 100,150,200]
    for name in algos:
        try:
            makedirs("data/graphs/" + name + "/")
        except:
            pass
    """ Image processing and layer output """
    for param in params:
        for n_iter in n_iters:
            for threshold in thresholds:
                for mask_cut in mask_cuts:
                    for img_size in img_sizes:
                        for layer_name in layers:
                            Reductor.auto(algos, modes, layer_name, threshold, img_size, volume_num, sample_num, mask_cut,
                                          param, param, n_iter, save, path, names, full_model=True, data_path=data_path,
                                          mask_path=mask_path)

    algos = ['tsne']
    for name in algos:
        try:
            makedirs("data/graphs/" + name + "/")
        except:
            pass
    """ Image processing and layer output """
    for param in params:
        for n_iter in n_iters:
            for threshold in thresholds:
                for mask_cut in mask_cuts:
                    for img_size in img_sizes:
                        for layer_name in layers:
                            Reductor.auto(algos, modes, layer_name, threshold, img_size, volume_num, sample_num, mask_cut,
                                          param, param, n_iter, save, path, names, full_model=True, data_path=data_path,
                                          mask_path=mask_path)
