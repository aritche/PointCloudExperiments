# A file that includes various analysis functions
from plotly import graph_objs as go
import plotly.express as px
import cv2
import numpy as np

def plotly_cloud(output, convert):
    points = convert(output)

    x, y, z = points[:,0], points[:,1], points[:,2]
    fig = go.Figure(data=go.Scatter3d(x=x, y=y, z=z,mode='markers', marker=dict(size=4)))
    fig.update_layout(scene_aspectmode='cube',
                      scene=dict(
                        xaxis=dict(range=[0,1]),
                        yaxis=dict(range=[0,1]),
                        zaxis=dict(range=[0,1])
                      ),
                      )
    fig.show()

def plot_output_maps(output, convert, num_sl, points_per_sl):
    points = convert(output)
    points = np.reshape(points, (num_sl, points_per_sl,3))

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.imshow('output', np.uint8(points*255))
    cv2.waitKey(0)
