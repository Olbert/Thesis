
import os
import numpy as np
import plotly.graph_objects as go

directory = 'E:\\Thesis\\DomainVis\\reductor\\data\\graphs\\new'
# E:\Thesis\DomainVis\reductor\data\graphs\new
domains = [
    'philips_15',
    'philips_3',
    'siemens_15',
    'siemens_3',
    'ge_15',
    'ge_3'
]



for root, subdirectories, files in os.walk(directory):
    # for subdirectory in subdirectories:
            files2 = files
        # for root2, subdirectories2, files2 in os.walk(os.path.join(directory,subdirectory)):
            for file in files2:
                if file.split('.')[1] == 'npy':
                    for dom in range(0, len(domains)):
                        # filename = os.path.join(root, subdirectory, file)
                        filename = os.path.join(root, file)
                        data = np.load(filename)
                        data = data.reshape(6,-1,2)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data[3, :, 0],
                                                 y=data[3, :, 1], name=domains[3],
                                                 mode='markers', opacity=0.7))

                        fig.add_trace(go.Scatter(x=data[dom, :, 0],
                                                 y=data[dom, :, 1], name=domains[dom],
                                                 mode='markers', opacity=0.7))

                        fig.write_image(os.path.join(directory,"pca",file.split('.')[0] + str(dom)+".jpg"))
