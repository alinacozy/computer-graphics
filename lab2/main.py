import math
from math import sqrt

import numpy as np
from PIL import Image, ImageOps


def make_pink(matrix):
    matrix[0:400, 0:300] = [255, 160, 150]  # розовый цвет


def input_vertices(filename):
    array_of_vertices = []
    array_of_polygons = []
    f = open(filename)
    for s in f:
        splited = s.split()
        if len(splited) == 0:
            continue
        if splited[0] == 'v':
            array_of_vertices.append([float(splited[1]), float(splited[2]), float(splited[3])])
        elif splited[0] == 'f':
            polygon = []
            for lex in splited[1:]:
                polygon.append(int(lex.split('/')[0]))
            array_of_polygons.append(polygon)
    nparray_of_vertices = np.array(array_of_vertices)
    f.close()
    return nparray_of_vertices, array_of_polygons


def baricentic(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    n = np.cross([x1 - x2, y1 - y2, z1 - z2], [x1 - x0, y1 - y0, z1 - z0])
    return n


def pict(x0, y0, z0, x1, y1, z1, x2, y2, z2, image_matrix, color):
    xmin = round(max(min(x0, x1, x2), 0))
    ymin = round(max(min(y0, y1, y2), 0))
    xmax = round(min(max(x0, x1, x2), 1000 - 1))
    ymax = round(min(max(y0, y1, y2), 1000 - 1))
    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            l0, l1, l2 = baricentic(x, y, x0, y0, x1, y1, x2, y2)
            if (l0 >= 0 and l1 >= 0 and l2 >= 0):
                z = l0 * z0 + l1 * z1 + l2 * z2
                if (z < z_buffer[y, x]):
                    image_matrix[y, x] = color
                    z_buffer[y, x] = z


def rotation_matrix(x_alpha, y_beta, z_gamma):
    alpha = np.radians(x_alpha)
    c, s = np.cos(alpha), np.sin(alpha)
    rx = np.zeros((3, 3), dtype=np.float32)
    rx[0, 0] = 1
    rx[1, 1] = rx[2, 2] = c
    rx[2, 1] = s
    rx[1, 2] = -s

    ry = np.zeros((3, 3), dtype=np.float32)
    beta = np.radians(y_beta)
    c, s = np.cos(beta), np.sin(beta)
    ry[0, 0] = ry[2, 2] = c
    ry[1, 1] = 1
    ry[2, 0] = s
    ry[0, 2] = -s

    rz = np.zeros((3, 3), dtype=np.float32)
    gamma = np.radians(z_gamma)
    c, s = np.cos(gamma), np.sin(gamma)
    rz[0, 0] = rz[1, 1] = c
    rz[2, 2] = 1
    rz[1, 0] = s
    rz[0, 1] = -s

    r = np.dot(rx, ry, rz)
    return r


image_matrix = np.zeros((1000, 1000, 3), dtype=np.uint8)

# здесь все делаем
z_buffer = np.zeros((1000, 1000), dtype=np.float32)
z_buffer[0:1000, 0:1000] = np.inf

vertices, polygons = input_vertices("model_1.obj")  # кролик
for i, v in enumerate(vertices):
    vertices[i] = np.dot(rotation_matrix(30,30,30), vertices[i])
    vertices[i] = [(v[0] * 5000) + 500, ((v[1] - 0.05) * 5000) + 500, (v[2] * 5000) + 500]

# vertices, polygons = input_vertices("model_2.obj")  # олень
# for i, v in enumerate(vertices):
#     vertices[i] = [(v[0] * 0.4) + 500, ((v[1]-625) * 0.4) + 500, (v[2] * 0.4) + 500]

# vertices, polygons = input_vertices("cat.obj")  # кошка
# for i, v in enumerate(vertices):
#     vertices[i] = [(v[0] * 10) + 500, (v[1] * 10) + 500, (v[2] * 10) + 500]

# vertices, polygons = input_vertices("cow.obj")  # корова
# for i, v in enumerate(vertices):
#     vertices[i] = np.dot(rotation_matrix(80, 180, 0), vertices[i])
#     vertices[i] = [((v[0]) * 10) + 500, ((v[1]-10) * 10) + 500, (v[2] * 10) + 500]

for p in polygons:
    n = normal(vertices[p[0] - 1][0], vertices[p[0] - 1][1], vertices[p[0] - 1][2],
               vertices[p[1] - 1][0], vertices[p[1] - 1][1], vertices[p[1] - 1][2],
               vertices[p[2] - 1][0], vertices[p[2] - 1][1], vertices[p[2] - 1][2])
    scalar = np.dot(n, [0, 0, 1]) / np.linalg.norm(n)  # нормированное скалярное произведение
    if (scalar < 0):
        # color = list(np.random.choice(range(256), size=3))
        color = [-248 * scalar, -24 * scalar, -148 * scalar]
        pict(vertices[p[0] - 1][0], vertices[p[0] - 1][1], vertices[p[0] - 1][2],
             vertices[p[1] - 1][0], vertices[p[1] - 1][1], vertices[p[1] - 1][2],
             vertices[p[2] - 1][0], vertices[p[2] - 1][1], vertices[p[2] - 1][2], image_matrix, color)
        if (len(p) == 4):
            pict(vertices[p[0] - 1][0], vertices[p[0] - 1][1], vertices[p[0] - 1][2],
                 vertices[p[2] - 1][0], vertices[p[2] - 1][1], vertices[p[2] - 1][2],
                 vertices[p[3] - 1][0], vertices[p[3] - 1][1], vertices[p[3] - 1][2], image_matrix, color)
img = Image.fromarray(image_matrix, mode='RGB')
img = ImageOps.flip(img)
img.save('rabbit.png')
