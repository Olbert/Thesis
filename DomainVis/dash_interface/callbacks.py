import pathlib
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

import DomainVis.dash_interface.layout as layout
import DomainVis.dash_interface.layout_utils as layout_utils
import DomainVis.reductor.dim_reduction as dim_reduction
import plotly.express as px

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


# TODO: reset button


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
		[Input("dropdown-algo", "value"), ]
	)
	def show_wordemb_controls(algo):
		# TODO: hide more
		if layout.PAGE_ON == 1:
			if algo == 'tsne':
				return {"display": "block"}, {"display": "block"}
			else:
				return {"display": "none"}, {"display": "none"}

	@app.callback(
		[
			Output("tsne-graph", "figure"),
			Output('memory', 'data'),
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
			volumes,
			samples,
			mask_cut,
			reductor,
	):
		if algo and layout.PAGE_ON == 1:
			# TODO: update arrays to singular values
			reductor = dim_reduction.Reductor.auto([algo], [mode], layer, None, (img_size, img_size), volumes,
			                                       samples, mask_cut, perplexity, n_iter, False)

			# plt.scatter(reductor.output[0][:, 0], reductor.output[0][:, 1])
			# plt.show()
			figure = layout_utils.plot2d(reductor.names, reductor.output, reductor.eval_maps)
			figure.update_layout(
				legend=dict(
					yanchor="top",
					y=0.96,
					xanchor="right",
					x=0.99
				),
				paper_bgcolor='rgba(0,0,0,0)',
				modebar_bgcolor='white',
				height = 700,
				width= 700,
				# plot_bgcolor='rgba(0,0,0,0)'
			)
			return figure, reductor.get_data()

	@app.callback(
		Output("div-plot-click-image", "figure"),

		[
			Input("tsne-graph", "clickData"),
			Input("tsne-graph", "selectedData"),
			Input("dropdown-mode", "value"),
		],
		State('memory', 'data'),
		prevent_initial_call=True
	)
	def displaySelectedData(clickData, selectedData, mode, reductor):
		if layout.PAGE_ON == 1:
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

				axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)
				fig.update_layout(
					paper_bgcolor='rgba(0,0,0,0)',
					plot_bgcolor='rgba(0,0,0,0)',
					modebar_bgcolor='white',
					margin=dict(l=0, r=0, b=0, t=0),
					scene=dict(xaxis=axes, yaxis=axes),
				height = 600,
				width= 600,)
				return fig

	@app.callback(Output('tabs-content-inline', 'children'),
	              Input('tabs-styled-with-inline', 'value'),
	              # prevent_initial_call=True)
	              )
	def render_content(tab):
		if tab == 'tab-1':
			if layout.PAGE_ON != 1:
				layout.PAGE_ON = 1
				return layout.PAGE1
		elif tab == 'tab-2':
			if layout.PAGE_ON != 2:
				layout.PAGE_ON = 2
				return layout.PAGE2
		elif tab == 'tab-3':
			if layout.PAGE_ON != 3:
				layout.PAGE_ON = 3
				return layout.PAGE3
		elif tab == 'tab-4':
			if layout.PAGE_ON != 4:
				layout.PAGE_ON = 4
				return layout.PAGE4

	@app.callback(
		[
			Output("map-graph", "figure"),
			Output('map-memory', 'data'),
		],
		[
			Input("dropdown-mode", "value"),
			Input("dropdown-layer", "value"),
			Input("slider-img_size", "value"),
			Input("dropdown-mask_cut", "value"),
		],
		State('map-memory', 'data'),
		prevent_initial_call=True
	)
	def display_3d_scatter_plot(
			mode,
			layer,
			img_size,
			mask_cut,
			map,
	):
		if layout.PAGE_ON == 2:
			map = dim_reduction.MapPresenter.auto(mode, layer, (img_size, img_size), mask_cut)

			new_im = np.concatenate(np.concatenate(np.array(map.image_mid), axis=2), axis=0).astype(np.uint8)
			figure = px.imshow(new_im,color_continuous_scale='gray')

			figure.update_layout(
				legend=dict(
					yanchor="top",
					y=0.96,
					xanchor="right",
					x=0.99
				),
				paper_bgcolor='rgba(0,0,0,0)',
				modebar_bgcolor='white',
				height = 600,
				width= 600,
				# plot_bgcolor='rgba(0,0,0,0)'
			)
			return figure, map.get_data()
