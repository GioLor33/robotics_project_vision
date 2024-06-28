#!/usr/bin/python3

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm

def plot_3D_graph(data, name_file_to_save, path_to_save="./", open_preview=False):

    x = data[:,0]
    y = data[:,1]
    z = data[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(y, x, z, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()

    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.invert_zaxis()

    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_zlabel("z")

    ax.view_init(-152, -94) 

    fig.savefig(f'{path_to_save}/{name_file_to_save}.png')
    if open_preview:
        plt.show()
    plt.clf()

def plot_3D_graph_v2(data, name_file_to_save, path_to_save="./", open_preview=False):
    
    fig = plt.figure(figsize = (8, 8)) 
    ax = plt.axes(projection = '3d') 
    
    # Data for a three-dimensional line 
    z = data[:,2]  
    x = data[:,0] 
    y = data[:,1] 
    ax.plot3D(y, x, z, 'green') 
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_zlabel("z")
    
    ax.view_init(-152, -94) 

    fig.savefig(f'{path_to_save}/{name_file_to_save}.png')
    if open_preview:
        plt.show()
    plt.clf()

def plot_2D_graph(data, name_file_to_save, path_to_save="./", open_preview=False):
    x = data[:,0]
    y = data[:,1]

    plt.figure()
    plt.plot(y,x)
    plt.xlabel("y")
    plt.ylabel("x")

    plt.savefig(f'{path_to_save}/{name_file_to_save}.png')
    if open_preview:
        plt.show()
    plt.clf()