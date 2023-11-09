import numpy as np
from scipy.spatial import ConvexHull
from InHullJudge import in_hull
from scipy.spatial import Delaunay
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Generate random points & convex hull



# Generate random points & convex hull

points = np.random.rand(20,3)

hull = ConvexHull(points)

fig = plt.figure()

ax = fig.add_subplot(projection = '3d')

# Plot hull's vertices

for vert in hull.vertices:

    ax.scatter(points[vert,0], points[vert,1], zs=points[vert,2])#, 'ro')

# Calculate Delaunay triangulation & plot

tri = Delaunay(points[hull.vertices])

for simplex in tri.simplices:

    vert1 = [points[simplex[0],0], points[simplex[0],1], points[simplex[0],2]]

    vert2 = [points[simplex[1],0], points[simplex[1],1], points[simplex[1],2]]

    vert3 = [points[simplex[2],0], points[simplex[2],1], points[simplex[2],2]]

    vert4 = [points[simplex[3],0], points[simplex[3],1], points[simplex[3],2]]

    ax.plot([vert1[0], vert2[0]], [vert1[1], vert2[1]], zs = [vert1[2], vert2[2]])

    ax.plot([vert2[0], vert3[0]], [vert2[1], vert3[1]], zs = [vert2[2], vert3[2]])

    ax.plot([vert3[0], vert4[0]], [vert3[1], vert4[1]], zs = [vert3[2], vert4[2]])

    ax.plot([vert4[0], vert1[0]], [vert4[1], vert1[1]], zs = [vert4[2], vert1[2]])

plt.show()





