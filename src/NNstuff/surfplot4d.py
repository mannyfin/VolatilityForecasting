import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
# domains

def surfplot(x,y,z,w):
    """
    
    :param x: vector of values
    :param y: vector of values
    :param z: vector of values
    :param w: vector of values. This vector becomes the color
    :return: a plot.
    """
    x = np.logspace(-1.,np.log10(5),50) # [0.1, 5]
    y = np.linspace(6,9,50)             # [6, 9]
    z = np.linspace(0,1,50)            # [0, 1]

    # convert to 2d matrices
    Z = np.outer(z.T, z)       # 50x50
    X, Y = np.meshgrid(x, y)    # 50x50
    w = np.sin(np.sqrt(x*y+ y*z))*np.cos(z*z*z)
    W = np.outer(w.T,w)

    # fourth dimention - colormap
    # create colormap according to x-value (can use any 50x50 array)
    color_dimension = W # change to desired fourth dimension
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)

    # plot
    fig= plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot_surface(X,Y,Z, rstride=1, cstride=1,linewidth=0,antialiased=True, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # m = cm.ScalarMappable(cmap=cm.jet)
    # m.set_array(w)
    plt.colorbar(m)
    cset = ax.contour(X, Y, Z, zdir='x', offset=min(x), cmap=cm.jet)
    cset = ax.contour(X, Y, Z, zdir='y', offset=min(y), cmap=cm.jet)
    cset = ax.contour(X, Y, Z, zdir='z', offset=min(z), cmap=cm.jet)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right',size='5%',pad=0.05)
    # cbar = fig.colorbar(cax, orientation='vertical')
    # cax = ax.imshow(w, interpolation='nearest')

    plt.show()