import base64
import pathlib
import numpy as np
import dash
from dash import dcc
from dash import html
from PIL import Image
from io import BytesIO
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import dash_interface.layout
import dash_interface.layout_utils
import plotly.graph_objs as go

import plotly.express as px
import time


def pil_to_b64(im, enc_format='png', verbose=False, **kwargs):
	"""
		Converts a PIL Image into base64 string for HTML displaying
		:param verbose:
		:param im: PIL Image object
		:param enc_format: The image format for displaying. If saved the image will have that extension.
		:return: base64 encoding
		"""
	t_start = time.time()

	buff = BytesIO()
	im.save(buff, format=enc_format, **kwargs)
	encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

	t_end = time.time()
	if verbose:
		print(f"PIL converted to b64 in {t_end - t_start:.3f} sec")

	return encoded


def numpy_to_b64(np_array, enc_format='png', scalar=True, **kwargs):
	"""
		Converts a numpy image into base 64 string for HTML displaying
		:param np_array:
		:param enc_format: The image format for displaying. If saved the image will have that extension.
		:param scalar:
		:return:
		"""
	# Convert from 0-1 to 0-255
	if scalar:
		np_array = np.uint8(255 * np_array)
	else:
		np_array = np.uint8(np_array)

	im_pil = Image.fromarray(np_array)

	return pil_to_b64(im_pil, enc_format, **kwargs)


def plot2d(names, output, eval_map=None):
	axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)

	layout = go.Layout(
		margin=dict(l=0, r=0, b=0, t=0),
		scene=dict(xaxis=axes, yaxis=axes),
	)
	data = []
	if eval_map is not None:
		img_types = ['TP', 'TN', 'FP', 'FN']
		for i in range(0, output.shape[0]):
			for type in range(0, len(img_types)):
				scatter = go.Scatter(
					name=names[i] + "_" + img_types[type],
					x=output[i, eval_map[i] == type][:, 0],
					y=output[i, eval_map[i] == type][:, 1],
					text="some text",  # [idx for _ in range(val["x"].shape[0])],
					textposition="top center",
					mode="markers",
					marker=dict(size=10, symbol="circle"),
				)
				data.append(scatter)
	else:
		for i in range(0, names.shape[0]):
			scatter = go.Scatter(
				name=names[i],
				x=output[i][:, 0],
				y=output[i][:, 1],
				text="sometext",  # [idx for _ in range(val["x"].shape[0])],
				textposition="top center",
				mode="markers",
				marker=dict(size=10, symbol="circle"),
			)
			data.append(scatter)

	figure = go.Figure(data=data, layout=layout)

	return figure


def find_point(reductor, point):
	point = np.array(point)
	index = np.unique(np.argwhere(np.array(reductor['output']) == point)[:, 0:4], axis=0)
	return index

