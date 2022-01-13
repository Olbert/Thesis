# -*- coding: utf-8 -*-
import os
import dash
import DomainVis.dash_interface.callbacks as callbacks
import DomainVis.dash_interface.layout as layout

# for the Local version, import local_layout and local_callbacks
# from local import local_layout, local_callbacks

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    # suppress_callback_exceptions=True
)
app.title = "t-SNE Explorer"

server = app.server
app.layout = layout.create_layout(app)
callbacks.demo_callbacks(app)

UPLOAD_DIRECTORY = "/project/"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)



# Running server
if __name__ == "__main__":
    app.run_server(debug=True,port=8050) # Debug
    # app.run_server(debug=False, port=8050) # Test
