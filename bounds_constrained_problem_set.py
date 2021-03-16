import numpy as np


class Problem:
    def __init__(self, problem_num):
        if problem_num == 0:
            self.problem = TestProblem()
        self.dim = self.problem.dim
        self.low_bounds = self.problem.low_bounds
        self.up_bounds = self.problem.up_bounds

    def objective_function(self, x):
        return self.problem.objctive_function(x)


class TestProblem:  # RC04
    def __init__(self):
        self.dim = 4
        self.up_bounds = [100, 100, 100, 100]
        self.low_bounds = [-100, -100, -100, -100]

    def objctive_function(self, x):
        y = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2
        return y