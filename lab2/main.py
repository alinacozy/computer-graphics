import math

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


def pict(x0, y0, x1, y1, x2, y2, image_matrix, color):
    xmin = round(max(min(x0, x1, x2), 0))
    ymin = round(max(min(y0, y1, y2), 0))
    xmax = round(min(max(x0, x1, x2), 1000 - 1))
    ymax = round(min(max(y0, y1, y2), 1000 - 1))
    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            l1, l2, l3 = baricentic(x, y, x0, y0, x1, y1, x2, y2)
            if (l1 >= 0 and l2 >= 0 and l3 >= 0):
                image_matrix[y, x] = color


image_matrix = np.zeros((1000, 1000, 3), dtype=np.uint8)
# здесь все делаем
vertices, polygons = input_vertices("model_2.obj")  # олень
for i, v in enumerate(vertices):
    vertices[i] = [(v[0] * 0.4) + 500, (v[1] * 0.4) + 250, (v[2] * 0.4) + 500]
for p in polygons:
    color = list(np.random.choice(range(256), size=3))
    pict(vertices[p[0] - 1][0], vertices[p[0] - 1][1], vertices[p[1] - 1][0], vertices[p[1] - 1][1],
         vertices[p[2] - 1][0], vertices[p[2] - 1][1], image_matrix, color)
img = Image.fromarray(image_matrix, mode='RGB')
img = ImageOps.flip(img)
img.save('3.png')
