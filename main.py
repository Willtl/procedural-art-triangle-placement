import multiprocessing

from joblib import Parallel, delayed

import utils
from differential_evolution import DifferentialEvolution
from pgpelib import PGPE
from utils import *

target_name = 'darwin'
target = load_target_image(file_name=f'targets/{target_name}.jpg', plot=False)
target = np.array(target).astype(np.float32)

de = DifferentialEvolution(target_name, target)
de.optimize()
x0 = de.get_best()

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
    min_v, max_v = solution.min(), solution.max()
    solution = (np.copy(solution) - min_v) / (max_v - min_v)
    img = draw_triangles_pil(solution)
    img = np.array(img).astype(np.float32)
    return -utils.mse(target, img)


num_iterations = 10000
for i in range(1, 1 + num_iterations):
    solutions = pgpe.ask()
    fitnesses = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(evaluate)(x) for x in solutions)
    pgpe.tell(fitnesses)
    print("Iteration:", i, "  cost: ", evaluate(pgpe.center))

sol = np.copy(pgpe.center)
min_v, max_v = sol.min(), sol.max()
sol = (np.copy(sol) - min_v) / (max_v - min_v)
min_v, max_v = sol.min(), sol.max()
img = draw_triangles_pil(sol)
plt.imshow(img)
plt.tight_layout()
plt.savefig(f'results/{target_name}_pgpe.png', bbox_inches='tight')
