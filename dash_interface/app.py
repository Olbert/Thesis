# -*- coding: utf-8 -*-
import os

import dash
import dash_interface.callbacks as callbacks
import dash_interface.layout as layout
import dash_interface.layout_utils
# for the Local version, import local_layout and local_callbacks
# from local import local_layout, local_callbacks

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
app.title = "t-SNE Explorer"

server = app.server
app.layout = layout.create_layout(app)
callbacks.demo_callbacks(app)

# Running server
if __name__ == "__main__":
    app.run_server(debug=True)
