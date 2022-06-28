import os

import PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

width, height = 400, 400
number_triangles = 100

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def mse(target, draw):
    return np.mean((target - draw) ** 2, dtype=np.float32)


def draw_triangles_pil(pos: np.ndarray):
    noise = np.random.uniform(0, 255, (width, height))
    image = Image.fromarray(noise).convert('RGB')

    draw = ImageDraw.Draw(image, mode='RGBA')
    for i in range(0, pos.shape[0], 10):
        draw.polygon(
            [
                (int(pos[i] * width), int(pos[i + 1] * width)),
                (int(pos[i + 2] * width), int(pos[i + 3] * width)),
                (int(pos[i + 4] * width), int(pos[i + 5] * width))
            ],
            fill=(int(pos[i + 6] * 255), int(pos[i + 7] * 255), int(pos[i + 8] * 255), int(max(0.1, pos[i + 9]) * 255)))
    del draw
    return image


def draw_triangles(image: np.ndarray, triangles: list) -> None:
    count = np.zeros_like(image)
    for triangle in triangles:
        for i in range(width):
            for j in range(height):
                if triangle.is_inside2(i, j):
                    image[i][j][0] += triangle.r
                    image[i][j][1] += triangle.g
                    image[i][j][2] += triangle.b

                    count[i][j][0] += 1

    for i in range(width):
        for j in range(height):
            if count[i][j][0] > 0:
                image[i][j][0] = int(image[i][j][0] / count[i][j][0])
                image[i][j][1] = int(image[i][j][1] / count[i][j][0])
                image[i][j][2] = int(image[i][j][2] / count[i][j][0])


def load_target_image(file_name: str = 'targets/mona.jpg', plot=False) -> np.ndarray:
    image: PIL.Image.Image = Image.open(file_name).convert('RGB')
    if plot:
        plt.imshow(image)
        plt.show()
    return image
