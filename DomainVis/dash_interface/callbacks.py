import pathlib
import numpy as np
import dash
import json
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.dash import no_update

import plotly.graph_objs as go
import DomainVis.dash_interface.layout as layout
import DomainVis.dash_interface.layout_utils as layout_utils
import DomainVis.reductor.reductor as dim_reduction
import DomainVis.presenter.presenter as presenter
import DomainVis.database_process.dataset_convert as dataset_convert
from DomainVis.database_process.dataset import H5Dataset
import plotly.express as px

import base64
import os
import config

import matplotlib.pyplot as plt

if not os.path.exists(config.UPLOAD_DIRECTORY):
	os.makedirs(config.UPLOAD_DIRECTORY)

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

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

current_id = "None"


def demo_callbacks(app):
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
		[
			Output("sidebar", "style"),
			Output("page-content", "style"),
			Output("side_click", "data"),
		],

		[Input("btn_sidebar", "n_clicks")],
		[
			State("side_click", "data"),
		]
	)
	def toggle_sidebar(n, nclick):
		if n:
			if nclick == "SHOW":
				sidebar_style = layout.SIDEBAR_HIDEN
				content_style = layout.CONTENT_STYLE1
				cur_nclick = "HIDDEN"
			else:
				sidebar_style = layout.SIDEBAR_STYLE
				content_style = layout.CONTENT_STYLE
				cur_nclick = "SHOW"
		else:
			sidebar_style = layout.SIDEBAR_STYLE
			content_style = layout.CONTENT_STYLE
			cur_nclick = 'SHOW'

		return sidebar_style, content_style, cur_nclick

	@app.callback(
		[
			Output("div-perplexity", "style"),
			Output("div-iterations", "style"),
		],
		[Input("dropdown-algo", "value"), ],
		State('page-memory', 'data'),
	)
	def show_wordemb_controls(algo, page_on):

		if page_on == 1:
			if algo == 'tsne':
				return {"display": "block"}, {"display": "block"}
			else:
				return {"display": "none"}, {"display": "none"}

	@app.callback(
		[
			Output("tsne-graph", "figure"),
			Output('reductor_memory', 'data'),
			Output('user-cache', 'children'),
		],
		[
			Input("dropdown-algo", "value"),
			Input("slider-perplexity", "value"),
			Input("slider-iterations", "value"),
			Input("dropdown-mode", "value"),
			Input("dropdown-layer", "value"),
			Input("slider-img_size", "value"),
			Input("slider-volumes", "value"),
			Input("slider-samples", "value"),
			Input("dropdown-mask_cut", "value"),
			# for update
			Input("div-plot-click-image", "clickData"),
			Input("div-plot-click-image", "selectedData"),
			# for reset
			Input("reset-button", "n_clicks"),

			Input("tabs-content-inline", 'children'),
			Input("tabs-styled-with-inline", 'value'),
	        Input('invisible_text', 'value'),

		],
		State('reductor_memory', 'data'),
		State('user-cache', 'children'),
		State('model_path-memory', 'data'),
		State('database_domains-memory', 'data'),
		State('page-memory', 'data'),
		prevent_initial_call=True
	)
	def display_3d_scatter_plot(
			algo,
			perplexity,
			n_iter,
			mode,
			layer,
			img_size,
			volumes,
			samples,
			mask_cut,

			image_clickData,
			image_selectedData,

			reset,
			tabs1,
			tabs2,
			text,

			reductor,
			user_cache,
			model_path,
			database_domains,
			page_on
	):
		if page_on == 1:
			ctx = dash.callback_context
			if ctx.triggered[0]['prop_id'].split('.')[0] == 'div-plot-click-image':
				reductor_plot = update_plot(image_clickData, image_selectedData, mode, reductor, page_on)
				return reductor_plot, reductor, user_cache
			elif ctx.triggered[0]['prop_id'].split('.')[0] == 'reset-button':
				reductor_plot = reset_graph(reset)
				return reductor_plot, reductor, user_cache
			else:
				if algo and page_on == 1:

					cache = json.loads(user_cache)
					current_id = "_".join([str(element) for element in
					                       [algo, mode, layer, (img_size, img_size), volumes, samples, mask_cut,
					                        perplexity,
					                        n_iter]])
					if current_id in cache:
						reductor = cache[current_id]
						reductor_dict = cache[current_id]
					else:

						reductor = dim_reduction.Reductor.precomputed([algo], [mode], layer, None, (img_size, img_size),
						                                       volumes, samples, mask_cut, perplexity, n_iter,
						                                       False, model_path, database_domains)
						reductor_dict = reductor.get_data()

					# plt.scatter(reductor.output[0][:, 0], reductor.output[0][:, 1])
					# plt.show()
					reductor_plot = layout_utils.plot2d(np.array(reductor_dict['domains']),
					                                    np.array(reductor_dict['output']),
					                                    np.array(reductor_dict['eval_maps']))

					reductor_plot.update_layout(
						legend=dict(
							yanchor="top",
							y=0.96,
							xanchor="right",
							x=0.99
						),
						paper_bgcolor='rgba(0,0,0,0)',
						modebar_bgcolor='white',
						height=600,
						width=600,
						# plot_bgcolor='rgba(0,0,0,0)'

					)

					# Add to cache

					cache[current_id] = reductor_dict
					for item in cache[current_id]:
						try:
							cache[current_id][item] = cache[current_id][item].tolist()
						except Exception:
							cache[current_id][item] = list(cache[current_id][item])
					return reductor_plot, cache[current_id], json.dumps(cache)
				else:
					return no_update
		else:
			return no_update

	@app.callback(
		Output("div-plot-click-image", "figure"),

		[
			Input("tsne-graph", "figure"),
			Input("tsne-graph", "clickData"),
			Input("tsne-graph", "selectedData"),
			Input("dropdown-mode", "value"),
			Input("reset-button", "n_clicks"),
		],
		State('reductor_memory', 'data'),
		State('page-memory', 'data'),
		prevent_initial_call=True
	)
	def displaySelectedData(reductor_figure, clickData, selectedData, mode, reset, reductor, page_on):
		if page_on == 1:
			ctx = dash.callback_context
			if ctx.triggered[0]['prop_id'].split('.')[0] == 'reset-button':
				return reset_graph(reset)
			if ctx.triggered[0]['prop_id'].split('.')[0] != 'dropdown-mode' and mode == 'pixel':
				points_list = []
				if ctx.triggered[0]['prop_id'].split('.')[1] == 'clickData':
					points_list = clickData['points']
				elif ctx.triggered[0]['prop_id'].split('.')[1] == 'selectedData':
					points_list = selectedData['points']
				elif ctx.triggered[0]['prop_id'].split('.')[1] == 'figure':
					points_list = []
				images = np.zeros((len(reductor['input']), len(reductor['input'][0]), reductor['input_img_size'][0],
				                   reductor['input_img_size'][1], 3))
				ratio = np.array(reductor['input_img_size'])[0] / np.array(reductor['output'])[0, 0, :, :, 0].shape[0]

				for i in range(len(reductor['input'])):
					for k in range(len(reductor['input'][0])):
						image = np.array(
							np.repeat(np.array(reductor['input'][i][k][0][0])[:, :, np.newaxis], 3, axis=2) * 255,
							dtype=np.uint8)
						images[i, k] = image

				all_coords = np.zeros((len(reductor['input']), len(reductor['input'][0]), reductor['input_img_size'][0],
				                       reductor['input_img_size'][1]), dtype=np.bool_)

				for data in points_list:

					point = list([data['x'], data['y']])
					index = layout_utils.find_point(reductor, point)
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
							for y in range(y0, y1 + 1):
								images[ind[0], ind[1]][x1, y] = [0, 255, 255]

				new_im = np.concatenate(np.concatenate(np.array(images), axis=2), axis=0).astype(np.uint8)
				fig = px.imshow(new_im)

				xaxes = dict(title="Domains", showgrid=False, zeroline=False, showticklabels=False, visible=True)
				yaxes = dict(title="Samples", showgrid=False, zeroline=False, showticklabels=False,visible=True)
				fig.update_layout(
					paper_bgcolor='rgba(0,0,0,0)',
					plot_bgcolor='rgba(0,0,0,0)',
					modebar_bgcolor='white',
					margin=dict(l=0, r=0, b=0, t=0),
					scene=dict(xaxis=xaxes, yaxis=yaxes),
					height=600,
					width=600,
					xaxis_title="Domains",
					yaxis_title="Samples",
					coloraxis_showscale=False)
				fig.update_layout(coloraxis_showscale=False)
				fig.update_xaxes(showticklabels=False)
				fig.update_yaxes(showticklabels=False)
				return fig
		else:
			return no_update

	def update_plot(clickData, selectedData, mode, reductor, page_on):
		if page_on == 1:
			ctx = dash.callback_context
			if ctx.triggered[0]['prop_id'].split('.')[0] != 'dropdown-mode' and mode == 'pixel':
				coord_list = []
				if ctx.triggered[0]['prop_id'].split('.')[1] == 'clickData':
					coord_list = clickData['points']
				elif ctx.triggered[0]['prop_id'].split('.')[1] == 'selectedData':

					# Take image divide in pieces. check which pieces belong to the figure
					coord_list = selectedData['points']


				# 1. Find image
				# 2. find point
				value_list = []
				point_list = []
				point_list_type = []
				for data in coord_list:
					point = list([data['x'], data['y']])

					domain_num = point[0] // len(reductor['input'][0][0][0][0])
					sample_num = point[1] // len(reductor['input'][0][0][0][0][0])

					# x = point[0] % len(reductor['input'][0][0][0][0])
					x = int(point[0] % len(reductor['input'][0][0][0][0]) / (
							reductor['input_img_size'][0] / reductor['activ_img_size'][0]))
					# y = point[1] % len(reductor['input'][0][0][0][0][0])
					y = int(point[1] % len(reductor['input'][0][0][0][0][0]) / (
							reductor['input_img_size'][1] / reductor['activ_img_size'][1]))
					# ids?

					if len(reductor['eval_maps']) > 0:
						point_list_type.append(reductor['eval_maps'][domain_num][sample_num][y][x] + domain_num * 4)
					# 4 is amount of types (TN,TP,FP,FN)

					point_list.append(y * reductor['activ_img_size'][0] + x)
					value_list.append(reductor['output'][domain_num][sample_num][y][x])
				reductor_plot = layout_utils.plot2d(np.array(reductor['domains']), np.array(reductor['output']),
				                                    np.array(reductor['eval_maps']))
				reductor_plot.update_layout(
					legend=dict(
						yanchor="top",
						y=0.96,
						xanchor="right",
						x=0.99
					),
					paper_bgcolor='rgba(0,0,0,0)',
					modebar_bgcolor='white',
					height=600,
					width=600,
				)
				point_list_type = np.array(point_list_type)
				point_list = np.array(point_list)
				value_list = np.array(value_list)
				point_ids = []
				if len(reductor['eval_maps']) > 0:

					for i in range(point_list.shape[0]):
						ids = np.intersect1d(np.where(reductor_plot.data[point_list_type[i]].x == value_list[i][0])[0],
						                     np.where(reductor_plot.data[point_list_type[i]].y == value_list[i][1])[0])
						for k in ids:
							point_ids.append(k)

					point_ids = np.array(point_ids)
					for point_type in range(len(reductor_plot.data)):
						reductor_plot.data[point_type].update(selectedpoints=point_ids[point_list_type == point_type],
						                                      selected=dict(marker=dict(color='red')),
						                                      # color of selected points
						                                      unselected=dict(marker=dict(color='rgb(200,200, 200)',
						                                                                  # color of unselected pts
						                                                                  opacity=0.9)))
				else:
					reductor_plot.data[0].update(selectedpoints=point_list,
					                             selected=dict(marker=dict(color='red')),  # color of selected points
					                             unselected=dict(
						                             marker=dict(color='rgb(200,200, 200)',  # color of unselected pts
						                                         opacity=0.9)))

				return reductor_plot



	@app.callback([Output("dropdown-layer", 'options'),
	               Output("dropdown-layer", 'value'),
	               Output("domain-layer", 'options'),
	               Output("domain-layer", 'value')],
	              [Input("tabs-content-inline", 'children'),
	               Input("tabs-styled-with-inline", 'value'),
	               Input('invisible_text', 'value')],
	              [State('model_path-memory', 'data'),
	               State('database_domains-memory', 'data'),
	               State('page-memory', 'data')])
	def layers_update(tab_cont, tab,text, model_path, database_domains, page_on):
		x=tab_cont,text
		is_initial = (dash.callback_context.triggered[0]['value'] is None)
		if tab == 'tab-1':
			if page_on != 1 or is_initial:
				layer_list = dim_reduction.Reductor.get_net(model_path)
				layer_dict_list = [{"label": "input_image", "value": "input_image", }]
				for layer in layer_list:
					layer_dict_list.append({"label": layer, "value": layer, })
				return layer_dict_list, layer_dict_list[0]['value'], None, None
		elif tab == 'tab-2':
			if page_on != 2 or is_initial:
				layer_list = dim_reduction.Reductor.get_net(model_path)
				layer_dict_list = [{"label": "input", "value": "input_image", }]
				for layer in layer_list:
					layer_dict_list.append({"label": layer, "value": layer, })

				domain_dict_list = []
				for domain in database_domains:
					domain_dict_list.append({"label": domain, "value": domain, })
				return layer_dict_list, layer_dict_list[0]['value'], domain_dict_list, domain_dict_list[0]['value']
		else:
			return no_update


	@app.callback([Output("tabs-content-inline", 'children'),
	               Output('page-memory', 'data'),
	               Output('invisible_text', 'value')],
	              Input("tabs-styled-with-inline", 'value'),
	              State('page-memory', 'data'))
	def tab_update(tab, page_on):
		is_initial = (dash.callback_context.triggered[0]['value'] is None)
		if tab == 'tab-0':
			if page_on != 0 or is_initial:
				page_on = 0
				return layout.PAGE0, page_on,'0'
		if tab == 'tab-1':
			if page_on != 1 or is_initial:
				page_on = 1
				return layout.PAGE1, page_on,'1'
		elif tab == 'tab-2':
			if page_on != 2 or is_initial:
				page_on = 2
				return layout.PAGE2, page_on,'2'
		elif tab == 'tab-3':
			if page_on != 3 or is_initial:
				page_on = 3
				return layout.PAGE3, page_on,'3'
		elif tab == 'tab-4':
			if page_on != 4 or is_initial:
				page_on = 4
				return layout.PAGE4, page_on,'4'


	@app.callback(
		[
			Output("map-graph", "figure"),
			Output('map-memory', 'data'),
		],
		[
			Input("domain-layer", 'value'),
			Input("dropdown-mode", "value"),
			Input("dropdown-layer", "value"),
			# Input("slider-img_size", "value"),
			# Input("dropdown-mask_cut", "value"),
			Input("tabs-content-inline", 'children'),
			Input("tabs-styled-with-inline", 'value')
		],
		[State('map-memory', 'data'),
		 State('model_path-memory', 'data'),
		 State('database_domains-memory', 'data'),
		 State('page-memory', 'data'), ],
		prevent_initial_call=True
	)
	def display_3d_scatter_plot(
			name,
			mode,
			layer,
			tabs_cont,
			tabs,
			map,
			model_path,
			names,
			page_on
	):
		if page_on == 2:
			map = presenter.MapPresenter.auto(mode, layer, model_path, name)
			new_im = (255 * (map.image_mid - np.min(map.image_mid)) / np.ptp(map.image_mid)).astype(int)

			figure = px.imshow(new_im, color_continuous_scale='gray')

			figure.update_layout(
				legend=dict(
					yanchor="top",
					y=0.96,
					xanchor="right",
					x=0.99
				),
				paper_bgcolor='rgba(0,0,0,0)',
				modebar_bgcolor='white',
				height=800,
				width=800,
				# plot_bgcolor='rgba(0,0,0,0)'
			)
			figure.update_xaxes(visible=False)
			figure.update_yaxes(visible=False)
			return [figure, map.get_data()]

		return no_update

	def reset_graph(n_clicks):
		fig = go.Figure(data=[go.Scatter(x=[], y=[])])
		return fig

	@app.callback(
		[Output('model_path-input', 'value'),
		 Output('model_path-memory', 'data')],
		[Input("model-upload", "filename"),
		 Input("model-upload", "contents")],
	)
	def upload_model(uploaded_filename, uploaded_file_content):
		"""Save uploaded files and regenerate the file list."""
		is_initial = (dash.callback_context.triggered[0]['value'] is None)
		if is_initial:
			# return 'default', 'E:\\Thesis\\DomainVis\\server_files\\model\\unet_damri.pt'
			return 'default', 'E:\Thesis\DomainVis\server_files\model\siemens_3_CP_epoch200.pt'
		if uploaded_filename is not None and uploaded_file_content is not None:
			data = uploaded_file_content.encode("utf8").split(b";base64,")[1]
			model_path = os.path.join(config.UPLOAD_DIRECTORY, config.MODEL_DIRECTORY, uploaded_filename)
			with open(model_path, "wb") as fp:
				fp.write(base64.decodebytes(data))
			return uploaded_filename, model_path

	@app.callback(
		[Output('database_path-input', 'value'),
		 Output('database_domains-memory', 'data')],
		[Input("database-upload", "filename"),
		 Input("database-upload", "contents")],
	)
	def upload_database(uploaded_filenames, uploaded_file_contents):
		"""Save uploaded files and regenerate the file list."""
		is_initial = (dash.callback_context.triggered[0]['value'] is None)
		if is_initial:
			# return 'default', ['philips', 'siemens']
			return 'default', ['siemens_3', 'philips_3']
		if uploaded_filenames is not None and uploaded_file_contents is not None:
			for name, content in zip(uploaded_filenames, uploaded_file_contents):
				data = content.encode("utf8").split(b";base64,")[1]
				# path_list.append(os.path.join(config.UPLOAD_DIRECTORY, name))
				with open(os.path.join(config.UPLOAD_DIRECTORY, config.DATA_DIRECTORY, name), "wb") as fp:
					fp.write(base64.decodebytes(data))

			if (uploaded_filenames[0].split('.')[1]=='h5'):
				full_names = []
				for name in uploaded_filenames:
					full_names.append(os.path.join(config.UPLOAD_DIRECTORY, config.DATA_DIRECTORY, name))

				full_dataset = H5Dataset.combine(full_names, os.path.join(config.UPLOAD_DIRECTORY, config.PROCESSED_DIRECTORY, 'data.h5'))
				names = H5Dataset.keys(full_dataset)

			else:
				names = dataset_convert.convert_to_h5(os.path.join(config.UPLOAD_DIRECTORY, config.DATA_DIRECTORY),
			                                      os.path.join(config.UPLOAD_DIRECTORY, config.PROCESSED_DIRECTORY),
			                                      'data')

			return str(uploaded_filenames), names

	@app.callback(
		[Output('label_path-input', 'value'),
		 Output('label_domains-memory', 'data')],
		[Input("label-upload", "filename"),
		 Input("label-upload", "contents")],
	)
	def upload_label(uploaded_filenames, uploaded_file_contents):
		is_initial = (dash.callback_context.triggered[0]['value'] is None)
		if is_initial:
			# return 'default', ['philips', 'siemens']
			return 'default', ['ge_15', 'ge_3']

		if uploaded_filenames is not None and uploaded_file_contents is not None:
			for name, content in zip(uploaded_filenames, uploaded_file_contents):
				data = content.encode("utf8").split(b";base64,")[1]
				# path_list.append(os.path.join(config.UPLOAD_DIRECTORY, name))
				with open(os.path.join(config.UPLOAD_DIRECTORY, config.DATA_DIRECTORY, name), "wb") as fp:
					fp.write(base64.decodebytes(data))

			if (uploaded_filenames[0].split('.')[1]=='h5'):
				full_paths = []
				for path in uploaded_filenames:
					full_paths.append(os.path.join(config.UPLOAD_DIRECTORY, config.DATA_DIRECTORY, path))

				full_dataset = H5Dataset.combine(full_paths, os.path.join(config.UPLOAD_DIRECTORY, config.PROCESSED_DIRECTORY, 'label.h5'))
				names = H5Dataset.keys(full_dataset)

			else:

				names = dataset_convert.convert_to_h5(os.path.join(config.UPLOAD_DIRECTORY, config.MASK_DIRECTORY),
				                                      os.path.join(config.UPLOAD_DIRECTORY, config.PROCESSED_DIRECTORY),
				                                      'label')
			return str(uploaded_filenames), names