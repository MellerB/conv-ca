import numpy as np
from scipy import ndimage


class Automaton:
    def __init__(self, init_tensor, kernel, rule, wrap=False):
        self.record = []
        self.rule = np.vectorize(rule)
        self.kernel = np.array(kernel)
        self.matrix = np.array(init_tensor)
        self.record.append(self.matrix)

        self.pad = None
        if not wrap:
            self.pad = "constant"
        else:
            self.pad = "wrap"

    def run(self, iters):
        for i in range(iters):
            self.matrix = ndimage.convolve(self.matrix, self.kernel, mode=self.pad)
            self.matrix = self.rule(self.matrix)
            self.record.append(self.matrix)
        return self.record
