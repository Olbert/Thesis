import base64
import io
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from PIL import Image
from io import BytesIO
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import pandas as pd
import plotly.graph_objs as go
import scipy.spatial.distance as spatial_distance
import plotly.express as px

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
import unet.dim_reduction

import datetime
import json
import time

# Variables
HTML_IMG_SRC_PARAMETERS = 'data:image/png;base64, '

with open(PATH.joinpath("demo_intro.md"), "r") as file:
	demo_intro_md = file.read()

with open(PATH.joinpath("demo_description.md"), "r") as file:
	demo_description_md = file.read()

styles = {
	'pre': {
		'border': 'thin lightgrey solid',
		'overflowX': 'scroll'
	}
}


def numpy_to_b64(array, scalar=True):
	# Convert from 0-1 to 0-255
	if scalar:
		array = np.uint8(255 * array)

	im_pil = Image.fromarray(array)
	buff = BytesIO()
	im_pil.save(buff, format="png")
	im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

	return im_b64


# Methods for creating components in the layout code
def Card(children, **kwargs):
	return html.Section(children, className="card-style")


def NamedDrpdown(name, id, options, placeholder, value):
	return html.Div(
		style={"margin": "25px 5px 30px 0px"},
		children=[
			f"{name}:",
			dcc.Dropdown(
				id=id,
				searchable=False,
				clearable=False,
				options=options,
				placeholder=placeholder,
				value=value,
			),
		],
	)


def NamedSlider(name, short, min, max, step, val, marks=None):
	if marks:
		step = None
	else:
		marks = {i: i for i in range(min, max + 1, step)}

	return html.Div(
		style={"margin": "25px 5px 30px 0px"},

		children=[
			f"{name}:",
			html.Div(
				style={"margin-left": "5px"},
				children=[
					dcc.Slider(
						id=f"slider-{short}",
						min=min,
						max=max,
						marks=marks,
						step=step,
						value=val,
					)
				],
			),
		],
	)


def NamedInlineRadioItems(name, short, options, val, **kwargs):
	return html.Div(
		id=f"div-{short}",
		style={"display": "inline-block"},
		children=[
			f"{name}:",
			dcc.RadioItems(
				id=f"radio-{short}",
				options=options,
				value=val,
				labelStyle={"display": "inline-block", "margin-right": "7px"},
				style={"display": "inline-block", "margin-left": "7px"},
			),
		],
	)


def create_layout(app):
	# Actual layout of the app
	return html.Div(
		className="row",
		style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
		children=[
			# Header
			html.Div(
				className="row header",
				id="app-header",
				style={"background-color": "#f9f9f9"},
				children=[
					html.Div(
						[
							html.Img(
								src=app.get_asset_url("bonn-logo.png"),
								className="logo",
								id="bonn-image",
							)
						],
						className="three columns header_img",
					),
					html.Div(
						[
							html.H3(
								"Visualizing the effects of domain shift on CNN based image segmentation",
								className="header_title",
								id="app-title",
							)
						],
						className="nine columns header_title_container",
					),
				],
			),
			# Demo Description
			html.Div(
				className="row background",
				id="demo-explanation",
				style={"padding": "50px 45px"},
				children=[
					html.Div(
						id="description-text", children=dcc.Markdown(demo_intro_md)
					),
					html.Div(
						html.Button(id="learn-more-button", children=["Learn More"])
					),
				],
			),
			# Body

			dcc.Store(id='memory'),

			html.Div(
				className="row background",
				style={"padding": "10px"},
				children=[
					html.Div(
						className="three columns",
						children=[
							Card(
								[
									NamedDrpdown(
										name="Algorithm",
										id="dropdown-algo",
										options=[
											{
												"label": "T-SNE",
												"value": "tsne",
											},
											{
												"label": "PCA",
												"value": "pca",
											},
											{
												"label": "LLA",
												"value": "lla",
											},
											{
												"label": "IsoMap",
												"value": "isomap",
											},
										],
										placeholder="Select an algorithm",
										value="pca",
									),
									NamedDrpdown(
										name="Mode",
										id="dropdown-mode",
										options=[
											{
												"label": "Pixel",
												"value": "pixel",
											},
											{
												"label": "Feature",
												"value": "feature",
											},
										],
										placeholder="Select mode",
										value="pixel",
									),
									# TODO: Generate on fly
									NamedDrpdown(
										name="Layer",
										id="dropdown-layer",
										options=[
											{"label": "Initial", "value": "init_path", },
											{"label": "Down1", "value": "down1", },
											{"label": "Down2", "value": "down2", },
											{"label": "Down3", "value": "down3", },
											{"label": "Up3", "value": "up3", },
											{"label": "Up2", "value": "up2", },
											{"label": "Up1", "value": "up1", },
											{"label": "Output path", "value": "out_path", },

										],
										placeholder="Select layer",
										value="init_path",
									),
									NamedSlider(
										# TODO: Change dynamically with layer output size?
										name="Image size",
										short="img_size",
										min=8,
										max=128,
										step=None,
										val=32,
										marks={
											i: str(i) for i in [8, 16, 32, 64, 128]
										},
									),

									NamedSlider(
										name="Samples per volume",
										short="samples",
										min=1,
										max=10,
										step=None,
										val=2,
										marks={i: str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
									),
									NamedSlider(
										name="Volumes per domain",
										short="samples2",
										min=1,
										max=10,
										step=None,
										val=1,
										marks={i: str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
									),
									NamedDrpdown(
										name="Mask Cut",
										id="dropdown-mask_cut",
										options=[
											{
												"label": "Prediction",
												"value": "predict",
											},
											{
												"label": "True mask",
												"value": "true_mask",
											},
											{
												"label": "None",
												"value": "None",
											},
										],
										placeholder="Type of mask to cut",
										value="None",
									),

									NamedSlider(
										name="Number Of Iterations",
										short="iterations",
										min=250,
										max=1000,
										step=None,
										val=500,
										marks={
											i: str(i) for i in [250, 500, 750, 1000]
										},
									),
									NamedSlider(
										name="Perplexity",
										short="perplexity",
										min=3,
										max=100,
										step=None,
										val=30,
										marks={i: str(i) for i in [3, 10, 30, 50, 100]},
									),
									# NamedSlider(
									#     name="Initial PCA Dimensions",
									#     short="pca-dimension",
									#     min=25,
									#     max=100,
									#     step=None,
									#     val=50,
									#     marks={i: str(i) for i in [25, 50, 100]},
									# ),
									# NamedSlider(
									#     name="Learning Rate",
									#     short="learning-rate",
									#     min=10,
									#     max=200,
									#     step=None,
									#     val=100,
									#     marks={i: str(i) for i in [10, 50, 100, 200]},
									# ),
									# html.Div(
									# 	id="div-wordemb-controls",
									# 	style={"display": "none"},
									# 	children=[
									# 		NamedInlineRadioItems(
									# 			name="Display Mode",
									# 			short="wordemb-display-mode",
									# 			options=[
									# 				{
									# 					"label": " Regular",
									# 					"value": "regular",
									# 				},
									# 				{
									# 					"label": " Top-100 Neighbors",
									# 					"value": "neighbors",
									# 				},
									# 			],
									# 			val="regular",
									# 		),
									# 		dcc.Dropdown(
									# 			id="dropdown-word-selected",
									# 			placeholder="Select word to display its neighbors",
									# 			style={"background-color": "#f2f3f4"},
									# 		),
									# 	],
									# ),
								]
							)
						],
					),
					html.Div(
						className="six columns",
						children=[
							dcc.Graph(id="graph-3d-plot-tsne", style={"height": "98vh"})
						],
					),
					html.Div(
						className="three columns",
						id="euclidean-distance",
						children=[
							Card(
								style={"padding": "5px"},
								children=[
									html.Div(
										id="div-plot-click-message",
										style={
											"text-align": "center",
											"margin-bottom": "7px",
											"font-weight": "bold",
										},
									),
									html.Div(id="div-plot-click-image1"),
									html.Div(id="div-plot-click-image2"),

									html.Div(id="div-plot-click-wordemb"),
									html.Pre(id='selected-data', style=styles['pre']),
									html.Div([
										html.Div(id='container')
									])

								],
							)
						],
					),

				],
			),
		],
	)


def demo_callbacks(app):
	def generate_thumbnail(image):
		return html.Div([
			html.A([
				html.Img(
					style={
						'height': '40%',
						'width': '40%',
						'float': 'left',
						'position': 'relative',
						'padding-top': 0,
						'padding-right': 0
					},
					src=HTML_IMG_SRC_PARAMETERS + pil_to_b64(image, enc_format='png'),
					width='100%'
				)
			], href='https://www.google.com'),
		])

	def image_show(images):
		images_div = []
		for i in images:
			images_div.append(generate_thumbnail(i))
		app.layout = html.Div(images_div)

	def pil_to_b64(im, enc_format='png', verbose=False, **kwargs):
		"""
		Converts a PIL Image into base64 string for HTML displaying
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
							name=names[i]+"_"+img_types[type],
							x=output[i, eval_map[i] == type][:, 0],
							y=output[i, eval_map[i] == type][:, 1],
							text="sometext",  # [idx for _ in range(val["x"].shape[0])],
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
		index = np.unique(np.argwhere(np.array(reductor['output']) == point)[:,0:4], axis=0)
		full_coord = []
		# TODO: returl list of all indexes with this value
		for ind in index:
			sample_num = ind[1] // (reductor['activ_img_size'][0] * reductor['activ_img_size'][1])
			coord_num = ind[1] - ((reductor['activ_img_size'][0] * reductor['activ_img_size'][1]) * sample_num)

			x = coord_num // reductor['activ_img_size'][0]
			y = coord_num - x * reductor['activ_img_size'][0]
			full_coord.append([ind[0], sample_num, x, y])
		return index

	# Callback function for the learn-more button
	@app.callback(
		[
			Output("description-text", "children"),
			Output("learn-more-button", "children"),
		],
		[Input("learn-more-button", "n_clicks")],
	)
	def learn_more(n_clicks):
		# If clicked odd times, the instructions will show; else (even times), only the header will show
		if n_clicks is None:
			n_clicks = 0
		if (n_clicks % 2) == 1:
			n_clicks += 1
			return (
				html.Div(
					style={"padding-right": "15%"},
					children=[dcc.Markdown(demo_description_md)],
				),
				"Close",
			)
		else:
			n_clicks += 1
			return (
				html.Div(
					style={"padding-right": "15%"},
					children=[dcc.Markdown(demo_intro_md)],
				),
				"Learn More",
			)

	@app.callback(
		Output("slider-perplexity", "style"),
		[Input("dropdown-algo", "value")]
	)
	def show_wordemb_controls(algo):
		# TODO: hide more
		if algo == 'tsne':
			return None
		else:
			return {"display": "none"}

	@app.callback(
		Output("dropdown-word-selected", "disabled"),
		[Input("radio-wordemb-display-mode", "value")],
	)
	def disable_word_selection(mode):
		return not mode == "neighbors"

	@app.callback(
		[
			Output("graph-3d-plot-tsne", "figure"),
			Output('memory', 'reductor'),
		],
		[
			Input("dropdown-algo", "value"),
			Input("slider-perplexity", "value"),
			Input("slider-iterations", "value"),
			Input("dropdown-mode", "value"),
			Input("dropdown-layer", "value"),
			Input("slider-img_size", "value"),
			Input("slider-samples", "value"),
			Input("dropdown-mask_cut", "value"),
		],
		State('memory', 'reductor')
	)
	def display_3d_scatter_plot(
			algo,
			perplexity,
			n_iter,
			mode,
			layer,
			img_size,
			samples,
			mask_cut,
			reductor,
	):
		if algo:
			# TODO: update arrays to singular values
			reductor = unet.dim_reduction.Reductor.auto([algo], [mode], layer, None, (img_size, img_size),
			                                            samples, mask_cut, perplexity, n_iter, False)

			# plt.scatter(reductor.output[0][:, 0], reductor.output[0][:, 1])
			# plt.show()
			figure = plot2d(reductor.names, reductor.output, reductor.eval_maps)
			figure.update_layout(legend=dict(
				yanchor="top",
				y=0.99,
				xanchor="right",
				x=0.01
			))
			return figure, reductor.get_data()

	@app.callback(
		Output("div-plot-click-image1", "children"),
		[
			Input("graph-3d-plot-tsne", "clickData"),
			Input("dropdown-algo", "value"),
			Input("dropdown-mode", "value"),
			# Input("slider-iterations", "value"),
			Input("slider-perplexity", "value"),
		],
		State('memory', 'reductor')
	)
	def display_click_image(clickData, algo, mode, perplexity, reductor):
		# Only for pixel mode
		if clickData and mode == 'pixel':
			point = list([clickData['points'][0]['x'], clickData['points'][0]['y']])

			k, sample_num, x, y = find_point(reductor, point)

			ratio = map(lambda x, y: x / y, reductor['input_img_size'], reductor['activ_img_size']).__next__()
			true_coord = x * ratio, y * ratio
			x0 = max(0, true_coord[0] - ratio / 2)
			y0 = max(0, true_coord[1] - ratio / 2)
			x1 = min(reductor['input_img_size'][0], true_coord[0] + ratio / 2)
			y1 = min(reductor['input_img_size'][1], true_coord[1] + ratio / 2)

			fig = px.imshow(reductor['input'][k][sample_num][0][0])
			fig.add_shape(
				type='rect',
				x0=x0, x1=x1, y0=y0, y1=y1,
				xref='x', yref='y',
				line_color='cyan'
			)

			return dcc.Graph(figure=fig)

	@app.callback(
		Output("container", "children"),
		[Input("graph-3d-plot-tsne", "selectedData"), ],
		State('memory', 'reductor')
	)
	def displaySelectedData(selectedData, reductor):
		if selectedData:
			images = np.zeros((len(reductor['input']), len(reductor['input'][0]), reductor['input_img_size'][0],
			                   reductor['input_img_size'][1], 3))
			# ratio = map(lambda x, y: x / y, reductor['input_img_size'], reductor['activ_img_size']).__next__()
			ratio = np.array(reductor['input_img_size'])[0]/np.array(reductor['output'])[0,0,:,:,0].shape[0]
			figs = []
			coord = []
			n = 0
			for i in range(len(reductor['input'])):
				for k in range(len(reductor['input'][0])):
					image = np.array(
						np.repeat(np.array(reductor['input'][i][0][0][0])[:, :, np.newaxis], 3, axis=2) * 255,
						dtype=np.uint8)
					images[i, k] = image

			all_coords = np.zeros((len(reductor['input']), len(reductor['input'][0]), reductor['input_img_size'][0],
			                       reductor['input_img_size'][1]), dtype=np.bool)

			for data in selectedData['points']:

				point = list([data['x'], data['y']])
				index = find_point(reductor, point)
				# k, sample_num, x, y

				for ind in index:
					if not all_coords[ind[0], ind[1], ind[2], ind[3]]:
						all_coords[ind[0], ind[1], ind[2], ind[3]] = True
						true_coord = ind[2] * ratio, ind[3] * ratio  # Of upper left corner
						x0 = int(max(0, true_coord[0]))
						y0 = int(max(0, true_coord[1]))
						x1 = int(min(reductor['input_img_size'][0] - 1, true_coord[0] + ratio))
						y1 = int(min(reductor['input_img_size'][1] - 1, true_coord[1] + ratio))

						for x in range(x0, x1):
							images[ind[0], ind[1]][x, y0] = [0, 255, 255]
						for x in range(x0, x1):
							images[ind[0], ind[1]][x, y1] = [0, 255, 255]
						for y in range(y0, y1):
							images[ind[0], ind[1]][x0, y] = [0, 255, 255]
						for y in range(y0, y1):
							images[ind[0], ind[1]][x1, y] = [0, 255, 255]
			#
			# # figs[0].data[0].z[x0,y0]
			# figs[ind[0]].add_shape(
			# 	type='rect',
			# 	x0=x0, x1=x1, y0=y0, y1=y1,
			# 	xref='x', yref='y',
			# 	line_color='cyan'
			# )

			# n += 1
			# coord.append(true_coord)
			images_b64 = []
			for i in range(len(reductor['input'])):
				for k in range(len(reductor['input'][0])):
					images_b64.append(HTML_IMG_SRC_PARAMETERS +
					                  pil_to_b64(Image.fromarray(np.array(images[i, k, :, :], dtype=np.uint8)),
					                             enc_format='png'))

			return html.Div([
				html.Div([
					html.Img(
						src=images_b64[i],
						id='image-{}'.format(i))
					for i in range(len(images_b64))
				]),
				html.Div(id='dynamic-output')
			])

	@app.callback(
		Output("div-plot-click-message", "children"),
		[Input("graph-3d-plot-tsne", "clickData"), Input("dropdown-algo", "value")],
	)
	def display_click_message(clickData, dataset):
		# # Displays message shown when a point in the graph is clicked, depending whether it's an image or word
		# if dataset in [IMAGE_DATASETS]:
		#     if clickData:
		#         return "Image Selected"
		#     else:
		#         return "Click a data point on the scatter plot to display its corresponding image."
		#
		# elif dataset in WORD_EMBEDDINGS:
		#     if clickData:
		#         return None
		#     else:
		#         return "Click a word on the plot to see its top 5 neighbors."
		pass

# # add a click to the appropriate store.
# @app.callback(Output('memory', 'data'),
#               Input('?','>'),
#               State('memory', 'data'))
#
# def on_click(n_clicks, data):
# 	if n_clicks is None:
# 		# prevent the None callbacks is important with the store component.
# 	    # you don't want to update the store for nothing.
#         raise PreventUpdate
#
# 	# Give a default data dict with 0 clicks if there's no data.
#     data = data or {'clicks': 0}
#
# 	data['clicks'] = data['clicks'] + 1
# 	return data
#
# # output the stored clicks in the table cell.
# @app.callback(Output('{}-clicks'.format(store), 'children'),
#               # Since we use the data prop in an output,
#               # we cannot get the initial data on load with the data prop.
#               # To counter this, you can use the modified_timestamp
#               # as Input and the data as State.
#               # This limitation is due to the initial None callbacks
#               # https://github.com/plotly/dash-renderer/pull/81
#               Input(store, 'modified_timestamp'),
#               State(store, 'data'))
# def on_data(ts, data):
#     if ts is None:
#         raise PreventUpdate
#
#     data = data or {}
#
#     return data.get('clicks', 0)
