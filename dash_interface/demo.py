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

import plotly.graph_objs as go

import plotly.express as px

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
import unet.dim_reduction

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


# Methods for creating components in the layout code
def Card(children, **kwargs):
	return html.Section(children, className="card-style")


def NamedDropdown(name, id, options, placeholder, value):
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
				style={"margin-left": "5px", 'display': 'block'},
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
									NamedDropdown(
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
									NamedDropdown(
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
									NamedDropdown(
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
									NamedDropdown(
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
						className="six columns",
						children=[
							dcc.Graph(id="div-plot-click-image", style={"height": "98vh"}),
						],
					),

				],
			),
		],
	)


def demo_callbacks(app):
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
			return {"display": "block"}
		else:
			return {"display": "none"}

	@app.callback(
		[
			Output("graph-3d-plot-tsne", "figure"),
			Output('memory', 'data'),
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
		State('memory', 'data')
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
			figure.update_layout(
				legend=dict(
					yanchor="top",
					y=0.96,
					xanchor="right",
					x=0.99
				),
				paper_bgcolor='rgba(0,0,0,0)',
				modebar_bgcolor= 'white'
				# plot_bgcolor='rgba(0,0,0,0)'
			)
			return figure, reductor.get_data()


	@app.callback(
		Output("div-plot-click-image", "figure"),

		[
			Input("graph-3d-plot-tsne", "clickData"),
			Input("graph-3d-plot-tsne", "selectedData"),
			Input("dropdown-mode", "value"),
		],
		State('memory', 'data'),
		prevent_initial_call=True
	)
	def displaySelectedData(clickData, selectedData, mode, reductor):

		ctx = dash.callback_context
		if ctx.triggered[0]['prop_id'].split('.')[0] != 'dropdown-mode' and mode == 'pixel':
			points_list = []
			if ctx.triggered[0]['prop_id'].split('.')[1] == 'clickData':
				points_list = clickData['points']
			elif ctx.triggered[0]['prop_id'].split('.')[1] == 'selectedData':
				points_list = selectedData['points']

			images = np.zeros((len(reductor['input']), len(reductor['input'][0]), reductor['input_img_size'][0],
			                   reductor['input_img_size'][1], 3))
			ratio = np.array(reductor['input_img_size'])[0] / np.array(reductor['output'])[0, 0, :, :, 0].shape[0]

			for i in range(len(reductor['input'])):
				for k in range(len(reductor['input'][0])):
					image = np.array(
						np.repeat(np.array(reductor['input'][i][0][0][0])[:, :, np.newaxis], 3, axis=2) * 255,
						dtype=np.uint8)
					images[i, k] = image

			all_coords = np.zeros((len(reductor['input']), len(reductor['input'][0]), reductor['input_img_size'][0],
			                       reductor['input_img_size'][1]), dtype=np.bool_)

			for data in points_list:

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
						for y in range(y0, y1+1):
							images[ind[0], ind[1]][x1, y] = [0, 255, 255]

			new_im = np.concatenate(np.concatenate(np.array(images), axis=2), axis=0).astype(np.uint8)
			fig = px.imshow(new_im)

			fig.update_layout(width=600,height=600, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)')
			return fig









