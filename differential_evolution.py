import multiprocessing
import os
import random
import time

import PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from skimage.metrics import structural_similarity as ssim

import utils
from utils import number_of_shapes, draw_triangles_pil, draw_circles_pil

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DifferentialEvolution:
    def __init__(self, target_name: str, target: PIL.Image.Image):
        # Target image
        self.target_name = target_name
        self.target = np.array(target).astype(np.float16)

        # Hyper-parameters
        self.iterations: int = 10000
        self.crossover_rate: float = 0.01
        self.number_individuals: int = multiprocessing.cpu_count()
        self.loss = 'mse'

        # Problem related params
        self.evaluation = draw_triangles_pil if utils.shape == 'triangle' else draw_circles_pil
        self.dimensions: int = 10 * number_of_shapes if utils.shape == 'triangle' else 8 * number_of_shapes
        self.pop: np.ndarray = np.random.uniform(0.0, 1.0, (self.number_individuals, self.dimensions)).astype(
            np.float32)
        self.pop_t1: np.ndarray = np.copy(self.pop)

        # Current fitness and t + 1 fitness
        self.fitness: np.ndarray = np.zeros(self.number_individuals).astype(np.float32)
        self.fitness = self.evaluate()
        self.fitness_t1 = np.copy(self.fitness)

    def optimize(self):
        start = time.time()
        for iteration in range(self.iterations):
            for i in range(self.number_individuals):
                a, b, c = self.select_three_rand(i)
                R = random.randint(0, self.dimensions - 1)
                if random.random() <= 0.5:
                    a = np.argmin(self.fitness)
                self.move(i, a, b, c, R)

            self.fitness_t1 = self.evaluate()
            for i in range(self.number_individuals):
                if self.fitness[i] >= self.fitness_t1[i]:
                    self.fitness[i] = self.fitness_t1[i]
                    self.pop[i] = np.copy(self.pop_t1[i])
                else:
                    self.pop_t1[i] = np.copy(self.pop[i])

            if (iteration + 1) % 100 == 0:
                print(f"Iteration: {iteration}, fitness: {self.fitness}, time: {time.time() - start}")
                start = time.time()

        best = np.argmin(self.fitness)
        img = self.evaluation(self.pop_t1[best])
        plt.imshow(img)
        plt.tight_layout()
        plt.savefig(f'results/{self.target_name}_diff_evo.png', bbox_inches='tight')

    def evaluate(self):
        fitness = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(self.render)(i) for i in range(self.number_individuals))
        return fitness

    def select_three_rand(self, index):
        indices = [i for i in range(self.number_individuals) if i != index]
        random.shuffle(indices)
        return indices[0], indices[1], indices[2]

    def move(self, index, a, b, c, R):
        for i in range(self.dimensions):
            if random.random() <= self.crossover_rate or R == i:
                self.pop_t1[index][i] = self.pop[a][i] + (
                        np.random.uniform(0.9, 1.1) * (self.pop[b][i] - self.pop[c][i])
                )
                self.pop_t1[index][i] = min(1.0, max(0.0, self.pop_t1[index][i]))

    def render(self, index: int):
        # Draw triangles
        img = self.evaluation(self.pop_t1[index])

        if self.loss == 'mse':
            # fitness = np.mean((self.target - img) ** 2, dtype=np.float32)
            fitness = utils.mse(self.target, img)
        elif self.loss == 'ssim':
            fitness = 1.0 - ssim(self.target, np.array(img), channel_axis=2)

        return fitness

    def get_best(self):
        index = np.argmin(self.fitness)
        return self.pop[index]
