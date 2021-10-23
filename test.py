# #!/usr/bin/env python
#
# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
# import numpy as np
# import skcuda.linalg as linalg
# from skcuda.linalg import PCA as cuPCA
# from matplotlib import pyplot as plt
# from sklearn import datasets
#
# iris = datasets.load_iris()
# X_orig = iris.data
# y = iris.target
#
# pca = cuPCA(4)  # take all 4 principal components
#
# demo_types = [np.float32, np.float64]  # we can use single or double precision
# precisions = ['single', 'double']
#
# print("Principal Component Analysis Demo!")
# print("Compute 2 principal components of a 1000x4 IRIS data matrix")
# print("Lets test if the first two resulting eigenvectors (principal components) are orthogonal,"
#       " by dotting them and seeing if it is about zero, then we can see the amount of the origial"
#       " variance explained by just two of the original 4 dimensions. Then we will plot the reults"
#       " for the double precision experiment.\n\n\n")
#
# for i in range(len(demo_types)):
#
# 	demo_type = demo_types[i]
#
# 	# 1000 samples of 4-dimensional data vectors
# 	X = X_orig.astype(demo_type)
# 	X = np.asfortranarray(X)
# 	X_gpu = gpuarray.to_gpu(X)  # copy data to gpu
#
# 	T_gpu = pca.fit_transform(X_gpu)  # calculate the principal components
#
# 	# show that the resulting eigenvectors are orthogonal
# 	# Note that GPUArray.copy() is necessary to create a contiguous array
# 	# from the array slice, otherwise there will be undefined behavior
# 	dot_product = linalg.dot(T_gpu[:, 0].copy(), T_gpu[:, 1].copy())
# 	T = T_gpu.get()
#
# 	print("The dot product of the first two " + str(precisions[i]) +
# 	      " precision eigenvectors is: " + str(dot_product))
#
# 	# now get the variance of the eigenvectors and create the ratio explained from the total
# 	std_vec = np.std(T, axis=0)
# 	print("We explained " + str(100 * np.sum(std_vec[:2]) / np.sum(std_vec)) +
# 	      "% of the variance with 2 principal components in " +
# 	      str(precisions[i]) + " precision\n\n")
#
# 	# Plot results for double precision
# 	if i == len(demo_types) - 1:
# 		# Different color means different IRIS class
# 		plt.scatter(T[:, 0], T[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=20)
# 		plt.show()
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash import Dash


app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

navbar = dbc.NavbarSimple(
    children=[
        dbc.Button("Sidebar", outline=True, color="secondary", className="mr-1", id="btn_sidebar"),
        dbc.NavItem(dbc.NavLink("Page 1", href="#")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Page 2", href="#"),
                dbc.DropdownMenuItem("Page 3", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Brand",
    brand_href="#",
    color="dark",
    dark=True,
    fluid=True,
)


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 62.5,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#f8f9fa",
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 62.5,
    "left": "-16rem",
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
	id="sidebar",
	style=SIDEBAR_STYLE,
    children=[
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with navigation links", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Page 1", href="/page-1", id="page-1-link"),
                dbc.NavLink("Page 2", href="/page-2", id="page-2-link"),
                dbc.NavLink("Page 3", href="/page-3", id="page-3-link"),
            ],
            vertical=True,
            pills=True,
        ),
    ],

)

content = html.Div(

    id="page-content",
    style=CONTENT_STYLE)

app.layout = html.Div(
    [
        dcc.Store(id='side_click'),
        dcc.Location(id="url"),
        navbar,
        sidebar,
        content,
    ],
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
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'

    return sidebar_style, content_style, cur_nclick

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1"]:
        return html.P("This is the content of page 1!")
    elif pathname == "/page-2":
        return html.P("This is the content of page 2. Yay!")
    elif pathname == "/page-3":
        return html.P("Oh cool, this is page 3!")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=8086)
