# A file that includes various analysis functions
from plotly import graph_objs as go
import plotly.express as px
import cv2
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

camera = dict(eye=dict(x=0,y=1,z=0))

def matplotlib_tom_cloud(input_data):
    tom_points = input_data.permute(1,0).cpu().detach().numpy()
    #tom_points = tom_points[np.random.choice(tom_points.shape[0], size=1000, replace=False), :]
    x, y, z, u, v, w = tom_points[:,0], tom_points[:,1], tom_points[:,2], tom_points[:,3], tom_points[:,4], tom_points[:,5]
    cu, cv, cw = (u+1)/2, (v+1)/2, (w+1)/2

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    print(min(cu), max(cu), min(cv), max(cv), min(cw), max(cw))
    ax.quiver(x, y, z, u, v, w, color=[(cu[i], cv[i], cw[i]) for i in range(len(x))], arrow_length_ratio=0.1, length=1/144, normalize=True)

    plt.show()


def plotly_everything(input_data, output, convert, num_sl, points_per_sl):
    points = convert(output)
    x, y, z = points[:,0], points[:,1], points[:,2]

    # Plot the lines
    colors, i = [], 0
    for sl in np.reshape(points, (num_sl, points_per_sl, 3)):
        colors.extend([i for x in range(len(sl))])
        i += 1
    colors = np.array(colors)

    fig = px.line_3d(x=x, y=y, z=z, color=colors, range_x=[0,1], range_y=[0,1], range_z=[0,1])

    # Plot the points
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z,mode='markers', marker=dict(size=4, opacity=0.5)))

    # Plot the TOM
    tom_points = input_data.permute(1,0).cpu().detach().numpy()
    x, y, z, u, v, w = tom_points[:,0], tom_points[:,1], tom_points[:,2], tom_points[:,3], tom_points[:,4], tom_points[:,5]
    fig.add_trace(go.Cone(x=x, y=y, z=z, u=u, v=v, w=w, sizemode='absolute', sizeref=1))

    fig.update_layout(scene_aspectmode='cube',
                      scene=dict(
                        xaxis=dict(range=[0,1]),
                        yaxis=dict(range=[0,1]),
                        zaxis=dict(range=[0,1])
                      ),
                      scene_camera=camera
                      )
    fig.show()

def plotly_tom_and_trk(input_data, output, convert):
    points = convert(output)

    # Plot the TRK cloud
    x, y, z = points[:,0], points[:,1], points[:,2]
    fig = go.Figure(data=go.Scatter3d(x=x, y=y, z=z,mode='markers', marker=dict(size=4, opacity=0.5)))

    # Plot the TOM cloud
    tom_points = input_data.permute(1,0).cpu().detach().numpy()
    x, y, z, u, v, w = tom_points[:,0], tom_points[:,1], tom_points[:,2], tom_points[:,3], tom_points[:,4], tom_points[:,5]
    #fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=4, opacity=0.1)))
    fig.add_trace(go.Cone(x=x, y=y, z=z, u=u, v=v, w=w, sizemode='absolute', sizeref=1))

    fig.update_layout(scene_aspectmode='cube',
                      scene=dict(
                        xaxis=dict(range=[0,1]),
                        yaxis=dict(range=[0,1]),
                        zaxis=dict(range=[0,1])
                      ),
                      scene_camera=camera
                      )
    fig.show()

def plotly_tom(input_data):
    tom_points = input_data.permute(1,0).cpu().detach().numpy()
    x, y, z, u, v, w = tom_points[:,0], tom_points[:,1], tom_points[:,2], tom_points[:,3], tom_points[:,4], tom_points[:,5]
    fig = go.Figure(go.Cone(x=x, y=y, z=z, u=u, v=v, w=w, sizemode='absolute', sizeref=1))
    fig.update_layout(scene_aspectmode='cube',
                      scene=dict(
                        xaxis=dict(range=[0,1]),
                        yaxis=dict(range=[0,1]),
                        zaxis=dict(range=[0,1])
                      ),
                      scene_camera=camera
                      )
    fig.show()

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
                      scene_camera=camera
                      )
    fig.show()

def plotly_lines(output, convert):
    streamlines = convert(output)
    points = np.reshape(streamlines, (-1,3))

    colors, i = [], 0
    for sl in streamlines:
        colors.extend([i for x in range(len(sl))])
        i += 1
    colors = np.array(colors)

    x, y, z = points[:,0], points[:,1], points[:,2]
    fig = px.line_3d(x=x, y=y, z=z, color=colors, range_x=[0,1], range_y=[0,1], range_z=[0,1])
    fig.update_layout(scene_aspectmode='cube',
                      scene=dict(
                        xaxis=dict(range=[0,1]),
                        yaxis=dict(range=[0,1]),
                        zaxis=dict(range=[0,1])
                      ),
                      scene_camera=camera
                      )
    fig.show()

def matplotlib_lines(output, convert):
    streamlines = convert(output)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1,2,1,projection='3d', aspect='auto')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_zlim([0, 1])

    for i in range(len(streamlines)):
        ax1.plot(streamlines[i,:,0], streamlines[i,:,1], streamlines[i,:,2])

    fig.tight_layout()
    plt.show()

def matplotlib_combined_lines(outputA, outputB, convert):
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # Plot the first streamlines
    ax1 = fig.add_subplot(1,2,1,projection='3d', aspect='auto')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_zlim([0, 1])
    streamlinesA = convert(outputA)
    for i in range(len(streamlinesA)):
        ax1.plot(streamlinesA[i,:,0], streamlinesA[i,:,1], streamlinesA[i,:,2])

    # Plot the second streamlines
    ax2 = fig.add_subplot(1,2,2,projection='3d', aspect='auto')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_zlim([0, 1])
    streamlinesB = convert(outputB)
    for i in range(len(streamlinesB)):
        ax2.plot(streamlinesB[i,:,0], streamlinesB[i,:,1], streamlinesB[i,:,2])

    fig.tight_layout()
    plt.show()


def matplotlib_overlap_lines(outputA, outputB, convert):
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # Plot the first streamlines
    ax1 = fig.add_subplot(1,1,1,projection='3d')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_zlim([0, 1])

    streamlinesA = convert(outputA)
    streamlinesB = convert(outputB)

    for i in range(len(streamlinesB)):
        ax1.plot(streamlinesB[i,:,0], streamlinesB[i,:,1], streamlinesB[i,:,2], c='blue', alpha=0.2)

    for i in range(len(streamlinesA)):
        ax1.plot(streamlinesA[i,:,0], streamlinesA[i,:,1], streamlinesA[i,:,2], c='orange', alpha=0.3)


    fig.tight_layout()
    plt.show()

def plot_output_maps(output, window_name, convert, num_sl, points_per_sl):
    points = convert(output)
    points = np.reshape(points, (num_sl, points_per_sl,3))
    points = np.swapaxes(points, 0,1)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, np.uint8(points*255))
    cv2.waitKey(0)
