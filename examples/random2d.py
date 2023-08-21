import sys

sys.path.append("..")
from automaton import Automaton
from display import Display2D
import numpy as np

kernel = np.random.uniform(low=-1, high=1, size=(3, 3))

kersum = np.sum(kernel)


def rule(x):
    return max(min(x / kersum, 1), 0)


print(kernel)

n = 64
size = 64
prop = 0.01

matrix = np.random.choice([0, 1], size=(n, n), p=[1 - prop, prop])
matrix = np.pad(matrix, (size - n) // 2, mode="constant", constant_values=(0))

n_iters = 360
automaton = Automaton(init_tensor=matrix, kernel=kernel, rule=rule, wrap=True)
record = automaton.run(n_iters)


disp = Display2D(record, offscreen=True, color="g4", fps=30)
disp.draw_all()
