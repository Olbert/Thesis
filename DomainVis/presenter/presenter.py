import os
import numpy as np

import torch
import torchvision

import DomainVis.reductor.reductor_utils as reductor_utils

np.random.seed(seed=0)

torch.random.manual_seed(seed=0)
activation = {}
weight = {}


class MapPresenter():
	def __init__(self,
	             model_path="E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Train2\checkpoints\CP_epoch200.pth",
	             domains=[]):

		"""Parameters Init"""
		# self.input = []
		# self.output = []
		# self.input_img_size = (128, 128)
		# self.activ_img_size = (8, 8)
		# self.coord = np.zeros(self.activ_img_size)
		self.domains = domains

		""" Net setup """
		self.net = torch.load(model_path)
		# self.net = UNet2D(n_chans_in=1, n_chans_out=1)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.net.to(device=device)

		self.data = reductor_utils.get_data(domains)


	def filter(self, layer_name):

		# TODO: for chosen testloader
		test_loader = self.data[0]

		for batch in test_loader:
			image, true_mask = batch['image'], batch['mask']
			# image = image.reshape(image.shape[0], 1, image.shape[1], image.shape[2])
			# true_mask = true_mask.reshape(true_mask.shape[0], 1, true_mask.shape[1], true_mask.shape[2])
			true_mask = torch.sigmoid(true_mask)
			self.true_mask = true_mask.cpu().numpy()[0, 0]
			if layer_name == 'input_image':
				filters = image.cpu().numpy()[0, 0]
			else:
				getattr(self.net, layer_name).register_forward_hook(
					reductor_utils.get_activation(layer_name))  # get the layer by it's name
				self.net.eval()
				img = image.to(device='cuda', dtype=torch.float32)

				with torch.no_grad():
					raw_out = self.net(img)

					modules = getattr(self.net, layer_name)

					if isinstance(modules, torch.nn.Sequential):
						for module in modules:
							try:
								filters = module.weight.detach().clone()
								nrow = int(np.ceil(np.sqrt(filters.shape[0])))
								n, c, w, h = filters.shape

								filters = filters.view(n * c, -1, w, h)

								grid = torchvision.utils.make_grid(filters, nrow=nrow, normalize=True, padding=1)

								self.image_mid = grid[0].cpu().detach().numpy()
								break
							except:
								pass
					else:
						filters = modules.weight.detach().clone()
						nrow = int(np.ceil(np.sqrt(filters.shape[0])))
						n, c, w, h = filters.shape

						filters = filters.view(n * c, -1, w, h)

						grid = torchvision.utils.make_grid(filters, nrow=nrow, normalize=True, padding=1)

						self.image_mid = grid[0].cpu().detach().numpy()

				break




	def activation(self, layer_name):
		# TODO: for chosen testloader
		test_loader = self.data[0]

		for batch in test_loader:
			image, true_mask = batch['image'], batch['mask']
			# image = image.reshape(image.shape[0], 1, image.shape[1], image.shape[2])
			# true_mask = true_mask.reshape(true_mask.shape[0], 1, true_mask.shape[1], true_mask.shape[2])
			true_mask = torch.sigmoid(true_mask)
			self.true_mask = true_mask.cpu().numpy()[0, 0]
			if layer_name == 'input_image':
				self.image_mid=image.cpu().numpy()[0]
			else:
				self.image_mid, self.net_output = reductor_utils.get_mid_output(self.net, image, layer_name)
			break

		nrow = int(np.ceil(np.sqrt(self.image_mid.shape[0])))
		# add epty cells to make it square
		self.image_mid = self.image_mid.reshape(self.image_mid.shape[0], -1, self.image_mid.shape[1],
		                                        self.image_mid.shape[2])
		self.image_mid = torch.tensor(self.image_mid)
		grid = torchvision.utils.make_grid(self.image_mid, nrow=nrow, normalize=True, padding=1)

		self.image_mid = grid[0].cpu().detach().numpy()


	def get_data(self):
		keys = ['net', 'data']
		for key in keys:
			self.__dict__.pop(key, None)

		return self.__dict__

	@staticmethod
	def auto(mode, layer_name,model_path,name):
		mapPresenter = MapPresenter(model_path,[name])
		if mode == "filter":
			mapPresenter.filter(layer_name)
		elif mode == "activation":
			mapPresenter.activation(layer_name)
		return mapPresenter
