import igl
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse.linalg import spsolve
import meshplot as mp
from meshplot import subplot, plot
import pygame, sys
import os

root_folder = os.getcwd()

# [1. Draw Border] --------------------------------------
pygame.init()
canvas = pygame.display.set_mode((540, 360))    #hd res but lower

# colors
black = (0, 0, 0)
white = (255, 255, 255)
yellow = (250, 219, 5)

lines = []

run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.MOUSEBUTTONUP:
            lines.append(event.pos)

    canvas.fill(white)
    if len(lines) > 1:  # yellow edge color
        pygame.draw.lines(canvas, yellow, False, lines, 3)
        for p in lines:  # black vertex color
            pygame.draw.circle(canvas, black, p, 3)
    if len(lines):
        pygame.draw.line(canvas, yellow, lines[-1], pygame.mouse.get_pos(), 3)
    pygame.display.flip()

print(lines)
linesIntoArray = np.array(lines)
flip1 = np.flip(linesIntoArray)
input = np.flip(-flip1)

# [2. Mesh Generation] --------------------------------------
# generate vertices within border
border = len(input) - 1
grid = np.copy(input)
centroid = grid.mean(axis=0)  # get center point value of the border
grid = np.vstack((grid, centroid))

for i in range(border):
    x = (input[i][0] + centroid[0]) / 2  # from border to the center, generate half length points
    y = (input[i][1] + centroid[1]) / 2
    grid = np.vstack((grid, (x, y)))

# triangulation
tri = Delaunay(grid)  # generate triangles from the vertex points (x and y only)
z = np.zeros((int(len(grid)), 1))  # get a list of 0's at the length of V
v = np.append(grid, z, axis=1)  # add the list of Z's axis of 0's on V's X and Y
f = tri.simplices  # generate a triangle face out of the new vertex list

# clean faces generated outside border
for i in range(len(v) - 1):
    if f[i][0] <= border and f[i][1] <= border and f[i][2] <= border:  # if faces xyz is <= border value
        f = np.delete(f, i, axis=0)  # delete bc face is outside border

meshGeneration = igl.write_triangle_mesh(os.path.join(root_folder, "data", "MeshGeneration1.obj"), v, f)


# [3. Mesh Inflation] --------------------------------------
# the border value is used to inflate remainings
for i in range(len(v)):
    if i > border:  # if increment of i is more than 11
        v[i][2] = 50  # row 12, column 2

meshInflation1 = igl.write_triangle_mesh(os.path.join(root_folder, "data", "MeshInflation1.obj"), v, f)

zvertices = np.copy(v)
zface = np.copy(f)

# reverse vertices
for i in range(len(zvertices)):
    zvertices[i][2] = zvertices[i][2] - (zvertices[i][2] + zvertices[i][2])

# reverse face normals
for i in range(len(zface)):
    dummy = np.zeros(shape=(1, 3))
    dummy[0][0] = zface[i][0]
    zface[i][0] = zface[i][2]
    zface[i][2] = dummy[0][0]

normaie = igl.write_triangle_mesh(os.path.join(root_folder, "data", "1.obj"), v, f)

# update vertex indices pointed to the newly stacked values
for i in range(len(zface)):
    zface[i] = zface[i] + len(v)

# join the vertices
v = np.vstack([v, zvertices])
f = np.vstack([f, zface])

duplicatedborder = round(int(len(v) / 2))  # 27, 54 vertexes in total.
facecounter = int(0)  # points to the original first 11 values

# go through shared vertex points and update faces pointing to them (original first 11 values)
for i in range(duplicatedborder,
               duplicatedborder + border + 1):  # vertex range from 0 to 11 (12?), num of loops     -> correct
    for j in range(len(f)):  # compare vertex values i with f's entire column of
        if i == f[j][0]:  # x axis
            f[j][0] = facecounter
        elif i == f[j][1]:  # y axis
            f[j][1] = facecounter
        elif i == f[j][2]:  # z axis
            f[j][2] = facecounter

    facecounter = facecounter + 1  # add from 0 and then increment vertex indice to 28

# delete unecessary vertices
duplicatedborder2 = round(int(len(v) / 2)) + 1
if v[duplicatedborder2][0] == v[1][0] and v[duplicatedborder2][1] == v[1][1]:
    v = np.delete(v, np.s_[int(duplicatedborder2) - 1:int(duplicatedborder) + border + 1], axis=0)
    # same thing but with the indices and the [] position

oldborder = int(((len(v)) + border) / 2) + 1  # do not round this

# final index updating                       -> this thing deletes the final 11 faces
for i in range(oldborder, len(v) + 1):  # 25 to 36
    for j in range(len(f)):  # compare values i with f's entire list
        if i + border + 1 == f[j][0]:  # x axis
            f[j][0] = i
        elif i + border + 1 == f[j][1]:  # y axis
            f[j][1] = i
        elif i + border + 1 == f[j][2]:  # z axis
            f[j][2] = i

meshInflation2 = igl.write_triangle_mesh(os.path.join(root_folder, "data", "MeshInflation2.obj"), v, f)

# [4. Geometry Smoothing] --------------------------------------
l = igl.cotmatrix(v, f)
n = igl.per_vertex_normals(v, f) * 0.5 + 0.5
c = np.linalg.norm(n, axis=1)
vs = [v]  # vertices source
cs = [c]  # cotangent source?

for i in range(1):
    m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    s = (m - 1000 * l)
    b = m.dot(v)
    v = spsolve(s, m.dot(v))
    n = igl.per_vertex_normals(v, f) * 0.5 + 0.5
    c = np.linalg.norm(n, axis=1)  # cotangent
    vs.append(v)
    cs.append(c)

p = subplot(vs[1], f, c, shading={"wireframe": False}, s=[1, 4, 0])

geometrySmoothing = igl.write_triangle_mesh(os.path.join(root_folder, "data", "geometrySmoothing1.obj"), vs[1], f)

#mp.offline()
#plot(vs[1], f, shading={"wireframe": False})

pygame.quit()
sys.exit()