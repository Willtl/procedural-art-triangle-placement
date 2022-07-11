import os

import PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

width, height = 400, 400
half_width, half_height = width // 2, height // 2
shape = ['triangle', 'ellipse'][0]
number_of_shapes = 100

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def mse(target, draw):
    return np.mean((target - draw) ** 2, dtype=np.float32)


def draw_triangles_pil(pos: np.ndarray):
    noise = np.random.uniform(0, 255, (width, height))
    image = Image.fromarray(noise).convert('RGB')

    draw = ImageDraw.Draw(image, mode='RGBA')
    for i in range(0, pos.shape[0], 10):
        pos[i + 9] = max(0.01, pos[i + 9])
        draw.polygon(
            [
                (int(pos[i] * width + 2) - 1, int(pos[i + 1] * height + 2) - 1),
                (int(pos[i + 2] * width + 2) - 1, int(pos[i + 3] * height + 2) - 1),
                (int(pos[i + 4] * width + 2) - 1, int(pos[i + 5] * height + 2) - 1)
            ],
            fill=(int(pos[i + 6] * 255), int(pos[i + 7] * 255), int(pos[i + 8] * 255), int(pos[i + 9] * 255)))
    del draw
    # plt.imshow(image)
    # plt.tight_layout()
    # plt.show()
    return image


def draw_circles_pil(pos: np.ndarray):
    noise = np.random.uniform(0, 255, (width, height))
    image = Image.fromarray(noise).convert('RGB')

    draw = ImageDraw.Draw(image, mode='RGBA')
    for i in range(0, pos.shape[0], 8):
        pos[i + 7] = max(0.1, pos[i + 7])
        w, h = int(pos[i] * width), int(pos[i + 1] * height)
        half_w, half_h = w / 2, h / 2
        x1, y1 = int(pos[i + 2] * (width + w) - half_w), int(pos[i + 3] * (width + h) - half_h)
        r, g, b, a = int(pos[i + 4] * 255), int(pos[i + 5] * 255), int(pos[i + 6] * 255), int(pos[i + 7] * 255),
        draw.ellipse(
            [
                (x1, y1),
                (x1 + w + 1, y1 + h + 1)
            ],
            fill=(r, g, b, a))
    del draw
    # plt.imshow(image)
    # plt.tight_layout()
    # plt.show()
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


def render_high_res(pos, name):
    if shape == 'triangle':
        res_w, res_h = 4096, 4096
        noise = np.ones((res_w, res_h)) * 255
        image = Image.fromarray(noise).convert('RGB')

        draw = ImageDraw.Draw(image, mode='RGBA')
        for i in range(0, pos.shape[0], 10):
            pos[i + 9] = max(0.01, pos[i + 9])
            draw.polygon(
                [
                    (int(pos[i] * res_w + 2) - 1, int(pos[i + 1] * res_h + 2) - 1),
                    (int(pos[i + 2] * res_w + 2) - 1, int(pos[i + 3] * res_h + 2) - 1),
                    (int(pos[i + 4] * res_w + 2) - 1, int(pos[i + 5] * res_h + 2) - 1)
                ],
                fill=(int(pos[i + 6] * 255), int(pos[i + 7] * 255), int(pos[i + 8] * 255), int(pos[i + 9] * 255)))
        del draw

    image.save(f'results/{name}.png')
