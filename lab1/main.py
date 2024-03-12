import math

import numpy as np
from PIL import Image, ImageOps


def make_pink(matrix):
    matrix[0:400, 0:300] = [255, 160, 150]  # розовый цвет


def make_gradient(matrix):
    for i in range(400):
        for j in range(300):
            matrix[i, j] = [(i - j) % 256, (i + j) % 256, (i + j) % 256]


def dotted_lines(matrix, x0, y0, x1, y1, count, color):
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        matrix[y, x] = color


def dotted_lines_v2(matrix, x0, y0, x1, y1, color):
    count = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        matrix[y, x] = color


def x_loop_line(image, x0, y0, x1, y1, color):
    for x in range(x0, round(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color


def x_loop_line_v2(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(round(x0), round(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color


def x_loop_line_v3(image, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(round(x0), round(x1)):
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if derror > 0.5:
            derror -= 1.0
            y += y_update


def bresenham_line(image, x0, y0, x1, y1, color):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0
    y_update = 1 if y1 > y0 else -1
    for x in range(round(x0), round(x1)):
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if derror > (x1 - x0):
            derror -= 2 * (x1 - x0)
            y += y_update


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


image_matrix = np.zeros((200, 200, 3), dtype=np.uint8)

# make_gradient(image_matrix)

make_pink(image_matrix)
for i in range(13):
    dotted_lines(image_matrix, 100, 100, round(100 + 95 * math.cos(2 * math.pi * i / 13)),
                 round(100 + 95 * math.sin(2 * math.pi * i / 13)), 50, [113, 0, 192])

img = Image.fromarray(image_matrix, mode='RGB')
img.save('dotted_lines.png')

make_pink(image_matrix)
for i in range(13):
    dotted_lines_v2(image_matrix, 100, 100, round(100 + 95 * math.cos(2 * math.pi * i / 13)),
                    round(100 + 95 * math.sin(2 * math.pi * i / 13)), [113, 0, 192])

img = Image.fromarray(image_matrix, mode='RGB')
img.save('dotted_lines_v2.png')

make_pink(image_matrix)
for i in range(13):
    x_loop_line(image_matrix, 100, 100, round(100 + 95 * math.cos(2 * math.pi * i / 13)),
                round(100 + 95 * math.sin(2 * math.pi * i / 13)), [113, 0, 192])

img = Image.fromarray(image_matrix, mode='RGB')
img.save('x_loop_line.png')

make_pink(image_matrix)
for i in range(13):
    x_loop_line_v2(image_matrix, 100, 100, round(100 + 95 * math.cos(2 * math.pi * i / 13)),
                   round(100 + 95 * math.sin(2 * math.pi * i / 13)), [113, 0, 192])

img = Image.fromarray(image_matrix, mode='RGB')
img.save('x_loop_line_v2.png')

make_pink(image_matrix)
for i in range(13):
    x_loop_line_v3(image_matrix, 100, 100, round(100 + 95 * math.cos(2 * math.pi * i / 13)),
                   round(100 + 95 * math.sin(2 * math.pi * i / 13)), [113, 0, 192])

img = Image.fromarray(image_matrix, mode='RGB')
img.save('x_loop_line_v3.png')

make_pink(image_matrix)
for i in range(13):
    bresenham_line(image_matrix, 100, 100, round(100 + 95 * math.cos(2 * math.pi * i / 13)),
                   round(100 + 95 * math.sin(2 * math.pi * i / 13)), [113, 0, 192])

img = Image.fromarray(image_matrix, mode='RGB')
img.save('lab1/bresenham_line.png')

# работа с модельками obj

image_matrix2 = np.zeros((1000, 1000, 3), dtype=np.uint8)
image_matrix2[0:1000, 0:1000] = [255, 0, 102]
vertices, polygons = input_vertices("lab1/model_1.obj")  # кролик
for i, v in enumerate(vertices):
    vertices[i] = [(v[0] * 5000) + 500, (v[1] * 5000) + 250, (v[2] * 5000) + 500]
    image_matrix2[round(vertices[i][1]), round(vertices[i][0])] = 255
for p in polygons:
    bresenham_line(image_matrix2, round(vertices[p[0] - 1][0]), round(vertices[p[0] - 1][1]),
                   round(vertices[p[1] - 1][0]),
                   round(vertices[p[1] - 1][1]), 255)
    bresenham_line(image_matrix2, round(vertices[p[0] - 1][0]), round(vertices[p[0] - 1][1]),
                   round(vertices[p[2] - 1][0]),
                   round(vertices[p[2] - 1][1]), 255)
    bresenham_line(image_matrix2, round(vertices[p[2] - 1][0]), round(vertices[p[2] - 1][1]),
                   round(vertices[p[1] - 1][0]),
                   round(vertices[p[1] - 1][1]), 255)

img = Image.fromarray(image_matrix2, mode='RGB')
img = ImageOps.flip(img)
img.save('lab1/model_1.png')

image_matrix2[0:1000, 0:1000] = [255, 0, 102]
vertices, polygons = input_vertices("lab1/model_2.obj")  # олень
for i, v in enumerate(vertices):
    vertices[i] = [(v[0] * 0.4) + 500, (v[1] * 0.4) + 250, (v[2] * 0.4) + 500]
    image_matrix2[round(vertices[i][1]), round(vertices[i][0])] = 255
for p in polygons:
    bresenham_line(image_matrix2, round(vertices[p[0] - 1][0]), round(vertices[p[0] - 1][1]),
                   round(vertices[p[1] - 1][0]),
                   round(vertices[p[1] - 1][1]), 255)
    bresenham_line(image_matrix2, round(vertices[p[0] - 1][0]), round(vertices[p[0] - 1][1]),
                   round(vertices[p[2] - 1][0]),
                   round(vertices[p[2] - 1][1]), 255)
    bresenham_line(image_matrix2, round(vertices[p[2] - 1][0]), round(vertices[p[2] - 1][1]),
                   round(vertices[p[1] - 1][0]),
                   round(vertices[p[1] - 1][1]), 255)

img = Image.fromarray(image_matrix2, mode='RGB')
img = ImageOps.flip(img)
img.save('lab1/model_2.png')

image_matrix2[0:1000, 0:1000] = [255, 0, 102]
vertices, polygons = input_vertices("lab1/natsuki.obj")  # нацуки
for i, v in enumerate(vertices):
    vertices[i] = [(v[0] * 800) + 500, (v[1] * 800) + 150, (v[2] * 700) + 500]
    image_matrix2[round(vertices[i][1]), round(vertices[i][0])] = 255
for p in polygons:
    bresenham_line(image_matrix2, round(vertices[p[0] - 1][0]), round(vertices[p[0] - 1][1]),
                   round(vertices[p[1] - 1][0]),
                   round(vertices[p[1] - 1][1]), 255)
    bresenham_line(image_matrix2, round(vertices[p[0] - 1][0]), round(vertices[p[0] - 1][1]),
                   round(vertices[p[2] - 1][0]),
                   round(vertices[p[2] - 1][1]), 255)
    bresenham_line(image_matrix2, round(vertices[p[2] - 1][0]), round(vertices[p[2] - 1][1]),
                   round(vertices[p[1] - 1][0]),
                   round(vertices[p[1] - 1][1]), 255)

img = Image.fromarray(image_matrix2, mode='RGB')
img = ImageOps.flip(img)
img.save('lab1/natsuki.png')

image_matrix2[0:1000, 0:1000] = [255, 0, 102]
vertices, polygons = input_vertices("lab1/cat.obj")  # кошка
for i, v in enumerate(vertices):
    vertices[i] = [(v[0] * 10) + 500, (v[1] * 10) + 500, (v[2] * 10) + 500]
    # vertices[i] = [(v[0] * 10) + 500, (v[2] * 10) + 300, (v[1] * 10) + 500] #поменяла местами оси, так получилась кошка спереди
    image_matrix2[round(vertices[i][1]), round(vertices[i][0])] = 255
for p in polygons:
    bresenham_line(image_matrix2, round(vertices[p[0] - 1][0]), round(vertices[p[0] - 1][1]),
                   round(vertices[p[1] - 1][0]),
                   round(vertices[p[1] - 1][1]), 255)
    bresenham_line(image_matrix2, round(vertices[p[0] - 1][0]), round(vertices[p[0] - 1][1]),
                   round(vertices[p[2] - 1][0]),
                   round(vertices[p[2] - 1][1]), 255)
    bresenham_line(image_matrix2, round(vertices[p[2] - 1][0]), round(vertices[p[2] - 1][1]),
                   round(vertices[p[1] - 1][0]),
                   round(vertices[p[1] - 1][1]), 255)

img = Image.fromarray(image_matrix2, mode='RGB')
img = ImageOps.flip(img)
img.save('lab1/cat.png')


image_matrix2[0:1000, 0:1000] = [255, 0, 102]
vertices, polygons = input_vertices("lab1/Common_Nase.obj")  # рыба
for i, v in enumerate(vertices):
    vertices[i] = [(v[0] * 30) + 500, (v[1] * 30) + 500, (v[2] * 30) + 500]
    image_matrix2[round(vertices[i][1]), round(vertices[i][0])] = 255
for p in polygons:
    bresenham_line(image_matrix2, round(vertices[p[0] - 1][0]), round(vertices[p[0] - 1][1]),
                   round(vertices[p[1] - 1][0]),
                   round(vertices[p[1] - 1][1]), 255)
    bresenham_line(image_matrix2, round(vertices[p[0] - 1][0]), round(vertices[p[0] - 1][1]),
                   round(vertices[p[2] - 1][0]),
                   round(vertices[p[2] - 1][1]), 255)
    bresenham_line(image_matrix2, round(vertices[p[2] - 1][0]), round(vertices[p[2] - 1][1]),
                   round(vertices[p[1] - 1][0]),
                   round(vertices[p[1] - 1][1]), 255)

img = Image.fromarray(image_matrix2, mode='RGB')
img = ImageOps.flip(img)
img.save('lab1/ryba.png')

image_matrix2[0:1000, 0:1000] = [255, 0, 102]
vertices, polygons = input_vertices("lab1/paimon.obj")  # паймон
for i, v in enumerate(vertices):
    vertices[i] = [(v[0] * 70) + 500, (v[1] * 70) + 150, (v[2] * 70) + 500]
    image_matrix2[round(vertices[i][1]), round(vertices[i][0])] = 255
for p in polygons:
    bresenham_line(image_matrix2, round(vertices[p[0] - 1][0]), round(vertices[p[0] - 1][1]),
                   round(vertices[p[1] - 1][0]),
                   round(vertices[p[1] - 1][1]), 255)
    bresenham_line(image_matrix2, round(vertices[p[0] - 1][0]), round(vertices[p[0] - 1][1]),
                   round(vertices[p[2] - 1][0]),
                   round(vertices[p[2] - 1][1]), 255)
    bresenham_line(image_matrix2, round(vertices[p[2] - 1][0]), round(vertices[p[2] - 1][1]),
                   round(vertices[p[1] - 1][0]),
                   round(vertices[p[1] - 1][1]), 255)

img = Image.fromarray(image_matrix2, mode='RGB')
img = ImageOps.flip(img)
img.save('lab1/paimon.png')

image_matrix2[0:1000, 0:1000] = [255, 0, 102]
vertices, polygons = input_vertices("lab1/dodoco.obj")  # додоко
for i, v in enumerate(vertices):
    vertices[i] = [(v[0] ) + 500, (v[1] ) + 450, (v[2] ) + 500]
    image_matrix2[round(vertices[i][1]), round(vertices[i][0])] = 255
for p in polygons:
    bresenham_line(image_matrix2, round(vertices[p[0] - 1][0]), round(vertices[p[0] - 1][1]),
                   round(vertices[p[1] - 1][0]),
                   round(vertices[p[1] - 1][1]), 255)
    bresenham_line(image_matrix2, round(vertices[p[0] - 1][0]), round(vertices[p[0] - 1][1]),
                   round(vertices[p[2] - 1][0]),
                   round(vertices[p[2] - 1][1]), 255)
    bresenham_line(image_matrix2, round(vertices[p[2] - 1][0]), round(vertices[p[2] - 1][1]),
                   round(vertices[p[1] - 1][0]),
                   round(vertices[p[1] - 1][1]), 255)

img = Image.fromarray(image_matrix2, mode='RGB')
img = ImageOps.flip(img)
img.save('lab1/dodoco.png')