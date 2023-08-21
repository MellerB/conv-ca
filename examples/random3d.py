import sys

sys.path.append("..")
from automaton import Automaton
from display import Display3D
import numpy as np
import math

kernel = np.random.uniform(low=-1, high=1, size=(3, 3, 3))

kernel = np.array(
    [
        [
            [-0.11842196, 0.93389825, 0.53412807],
            [0.31985993, 0.38075689, -0.92693758],
            [-0.23629656, -0.43827381, -0.85000269],
        ],
        [
            [-0.74181325, -0.28769207, 0.34517163],
            [0.16842913, -0.1443949, 0.77728211],
            [0.34473825, -0.52290874, -0.76161923],
        ],
        [
            [-0.11743971, 0.80619295, 0.34069011],
            [0.25129404, 0.44161321, 0.10762217],
            [-0.28882791, -0.69417313, -0.37355514],
        ],
    ]
)

kersum = np.sum(kernel)
print(kersum)


def rule(x):
    return max(min(x / kersum, 1), 0)


print(kernel)

n = 16
size = 16
prop = 0.01

matrix = np.random.choice([0, 1], size=(n, n, n), p=[1 - prop, prop])
matrix = np.pad(matrix, (size - n) // 2, mode="constant", constant_values=(0))

n_iters = 180
automaton = Automaton(init_tensor=matrix, kernel=kernel, rule=rule, wrap=True)
record = automaton.run(n_iters)


disp = Display3D(record, offscreen=True, color="g4", fps=5)
disp.draw_all()
