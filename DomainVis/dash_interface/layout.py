import pathlib
import json
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
with open(PATH.joinpath("demo_intro.md"), "r") as file:
	demo_intro_md = file.read()

with open(PATH.joinpath("demo_description.md"), "r") as file:
	demo_description_md = file.read()


# Methods for creating components in the layout code
def Card(children, **kwargs):
	return html.Section(children, className="card-style")


def NamedDropdown(name, id, options=[], placeholder='', value=''):
	return html.Div(
		style={"margin": "25px 5px 30px 0px"},
		id=f"div-{id}",
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
		id=f"div-{short}",
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


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
	"position": "relative",

	"left": "0",
	"bottom": "0",
	"width": "20%",
	"height": "600px",
	"z-index": 1,
	"overflow-x": "hidden",
	"transition": "all 0.3s",
	"padding": "0.5rem 1rem",
	"background-color": "#f8f9fa",
	"display": "inline-block",
}

SIDEBAR_HIDEN = {
	"position": "relative",
	"left": "-16rem",
	"bottom": "0",
	"width": "0",
	"height": "600px",
	"z-index": 1,
	"overflow-x": "hidden",
	"transition": "all 0.3s",
	"padding": "0rem 0rem",
	"background-color": "#f8f9fa",
	"display": "inline-block",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
	"transition": "margin-left .3s",
	"height": "0",
	"background-color": "#f8f9fa",
	"width": "80%",
	"display": "inline-block",
}

CONTENT_STYLE1 = {
	"transition": "margin-left .3s",
	"background-color": "#f8f9fa",
	"height": "0",
	"width": "100%",
	"display": "inline-block",
}


tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}


PAGE_ON = 2
PAGE0 =html.Div(
	id='page0',
	className="row background",
	# style={"padding": "10px"},
	children=[
		html.Div(style={'fontSize': 18},
			children=[
				html.P('Select model path'),
				dcc.Input(
					id='model_path-input',
					style={"display": "inline-block", "margin-left": "7px", 'width':'500px'},
				),
				dcc.Upload(id='model-upload', children=[html.Button('Upload File')]),
			],
		),
		html.Div(style={'fontSize': 18},
			children=[
				html.P('Select data path'),
				dcc.Input(
					id='database_path-input',
					style={"display": "inline-block", "margin-left": "7px", 'width':'500px'},
				),
				dcc.Upload(id='database-upload', children=[html.Button('Upload Files')], multiple=True),
			],
		),
		html.Div(style={'fontSize': 18},
				children=[
					html.P('Select label path'),
					dcc.Input(
						id='label_path-input',
						style={"display": "inline-block", "margin-left": "7px", 'width':'500px'},

					),
					dcc.Upload(id='label-upload', children=[html.Button('Upload Files')],multiple=True),
				],
			)

	],
),
PAGE1 = html.Div(
	id='page1',
	className="row background",
	# style={"padding": "10px"},
	children=[
		dbc.Button("Sidebar", outline=True, color="secondary", className="mr-1", id="btn_sidebar",
		           style={"left": 0}),
		html.Div(
			style={"white-space": "nowrap"},
			children=[
				html.Div(
					id="sidebar",
					style=SIDEBAR_STYLE,
					children=[
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
									"label": "LLE",
									"value": "lle",
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
						NamedDropdown(
							name="Layer",
							id="dropdown-layer",
							placeholder="Select layer",
							options=[
								{"label": "input", "value": "input", },
								{"label": "init_path", "value": "init_path", },
								{"label": "down1", "value": "down1", },
								{"label": "down2", "value": "down2", },
								{"label": "down3", "value": "down3", },
								{"label": "up1", "value": "up1", },
								{"label": "up2", "value": "up2", },
								{"label": "up3", "value": "up3", },
								{"label": "out_path", "value": "out_path", },

							],
							value="down1",
						),
						NamedSlider(
							# TODO: Change dynamically with layer output size?
							name="Image size",
							short="img_size",
							min=8,
							max=128,
							step=None,
							val=8,
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
							short="volumes",
							min=1,
							max=10,
							step=None,
							val=1,
							marks={i: str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
						),
						NamedDropdown(
							name="Background remove",
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
						html.Button(id="reset-button", children=["Reset graph"])
					]
				),
				html.Div(
					id="page-content",
					style=CONTENT_STYLE,
					children=[
						dcc.Graph(id="tsne-graph",
						          style={
							          "display": "inline-block",
							          "width": "45%",


						          }),
						dcc.Graph(id="div-plot-click-image",
						          style={
							          "display": "inline-block",
							          "width": "45%",
							          "xaxis": {"visible": False,},
							          "yaxis": {"visible": False, },
						          },
						          config = {'modeBarButtons': [['zoom2d', 'pan2d','select2d','lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']]}
						          ),
						]
				)
			]
		)
	],
),

PAGE2 = html.Div(
	id='page2',
	className="row background",
	# style={"padding": "10px"},
	children=[
		dbc.Button("Sidebar", outline=True, color="secondary", className="mr-1", id="btn_sidebar",
		           style={"left": 0}),
		html.Div(
			style={"white-space": "nowrap"},
			children=[
				html.Div(
					id="sidebar",
					style=SIDEBAR_STYLE,
					children=[
						NamedDropdown(
							name="Domain",
							id="domain-layer",
							options=[
								{"label": "Domain1", "value": "domain1", },
								{"label": "Domain2", "value": "domain2", },

							],
							placeholder="Select layer",
							value="init_path",
						),
						NamedDropdown(
							name="Mode",
							id="dropdown-mode",
							options=[
								{
									"label": "Filter",
									"value": "filter",
								},
								{
									"label": "Activation",
									"value": "activation",
								},
							],
							placeholder="Select mode",
							value="activation",
						),
						NamedDropdown(
							name="Layer",
							id="dropdown-layer",
							placeholder="Select layer",
							options=[
								{"label": "init_path", "value": "init_path", },
								{"label": "down1", "value": "down1", },
								{"label": "down2", "value": "down2", },
								{"label": "down3", "value": "down3", },
								{"label": "up1", "value": "up1", },
								{"label": "up2", "value": "up2", },
								{"label": "up3", "value": "up3", },
								{"label": "out_path", "value": "out_path", },

							],
						),
						# NamedSlider(
						# 	# TODO: Change dynamically with layer output size?
						# 	name="Image size",
						# 	short="img_size",
						# 	min=8,
						# 	max=128,
						# 	step=None,
						# 	val=32,
						# 	marks={
						# 		i: str(i) for i in [8, 16, 32, 64, 128]
						# 	},
						# ),
						# NamedDropdown(
						# 	name="Mask Cut",
						# 	id="dropdown-mask_cut",
						# 	options=[
						# 		{
						# 			"label": "Prediction",
						# 			"value": "predict",
						# 		},
						# 		{
						# 			"label": "True mask",
						# 			"value": "true_mask",
						# 		},
						# 		{
						# 			"label": "None",
						# 			"value": "None",
						# 		},
						# 	],
						# 	placeholder="Type of mask to cut",
						# 	value="None",
						# ),
						html.Button(id="reset-button", children=["Reset graph"])

					]
				),
				html.Div(
					id="page-content",

					style=CONTENT_STYLE,
					children=[
						dcc.Graph(id="map-graph",
						          style={
							          "display": "inline-block",
							          "width": "100%",
							          "xaxis": {"visible": False, },
							          "yaxis": {"visible": False, },
						          }),
						]
				)
			]
		)
	],
),
PAGE3 =html.Div(html.P("Something"))
PAGE4 = html.Div(
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
initial_layout = {}
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


			# Body
			dcc.Store(id='long_term_memory'),
			dcc.Store(id='reductor_memory'),
			dcc.Store(id='map-memory',data=(0,0)),
			dcc.Store(id='side_click'),
			dcc.Store(id='model_path-memory'),
			dcc.Store(id='database_domains-memory'),
			dcc.Store(id='label_domains-memory'),
			dcc.Store(id='page-memory'),
			html.Div([
				dcc.Tabs(id="tabs-styled-with-inline", value='tab-0', children=[
					dcc.Tab(label='Tool setup',                 value='tab-0', style=tab_style, selected_style=tab_selected_style),
					dcc.Tab(label='Dimentionality reduction',   value='tab-1', style=tab_style, selected_style=tab_selected_style),
					dcc.Tab(label='Feature map explorer',       value='tab-2', style=tab_style, selected_style=tab_selected_style),
					# dcc.Tab(label='Autoencoder?',             value='tab-3', style=tab_style, selected_style=tab_selected_style),
					dcc.Tab(label='About',                      value='tab-4', style=tab_style, selected_style=tab_selected_style),
				], style=tabs_styles),
				html.Div(id='tabs-content-inline'

				         )
			]),
			html.Div(id='user-cache', style={'display': 'none'},
			         children=json.dumps(initial_layout)),
			html.Div(id='invisible_text', style={'visible': 'False'}),

		],
	)



