import multiprocessing
import sys

from joblib import Parallel, delayed

import utils
from differential_evolution import DifferentialEvolution
from pgpelib import PGPE
from utils import *

target_name = 'mona'
target = load_target_image(file_name=f'targets/{target_name}.jpg', plot=False)
target = np.array(target).astype(np.float32)

de = DifferentialEvolution(target_name, target)
de.optimize()
x0 = de.get_best()

eval_func = draw_triangles_pil if utils.shape == 'triangle' else draw_circles_pil

n = x0.shape[0]
pgpe = PGPE(
    solution_length=n,
    popsize=250,
    center_init=x0,
    center_learning_rate=0.075,
    optimizer='clipup',
    optimizer_config={'max_speed': 0.15},
    stdev_init=0.08,
    stdev_learning_rate=0.1,
    stdev_max_change=0.2,
    solution_ranking=True,
    dtype='float32'
)


def evaluate(solution):
    minv, maxv = solution.min(), solution.max()
    solution = (np.copy(solution) - minv) / (maxv - minv)
    img = eval_func(solution)
    img = np.array(img).astype(np.float32)
    return -utils.mse(target, img)


best_center, best_obj = None, -sys.maxsize
num_iterations = 40000
for i in range(1, 1 + num_iterations):
    solutions = pgpe.ask()
    fitnesses = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(evaluate)(x) for x in solutions)
    pgpe.tell(fitnesses)
    obj = evaluate(pgpe.center)
    if best_obj < obj:
        best_center = np.copy(pgpe.center)
        best_obj = obj
        print('New best found', i, "  cost: ", obj)
    else:
        print("Iteration:", i, "  cost: ", obj)

    if i % 100 == 0:
        print(f'Rendering', best_obj)
        sol = np.copy(best_center)
        min_v, max_v = sol.min(), sol.max()
        sol = (sol - min_v) / (max_v - min_v)
        min_v, max_v = sol.min(), sol.max()
        utils.render_high_res(sol, f'pgpe_{target_name}_{i}')
