import matplotlib.pyplot as plt
import numpy as np
from Constrained_Optimization.epsilon_method import EpsilonMethod
from Problem_Set.problem_set import Problem

from Constrained_Optimization.PFA import Penalty
from Constrained_Optimization.superiority_method import SuperiorityMethod
from Constrained_Optimization.epsilon_method import EpsilonMethod
from Constrained_Optimization.pfa_epsilon import PenaltyEpsilon
import time
import pandas as pd

TEST = 'SHADE_TEST_' + time.strftime("%m_%d_%H_%M", time.localtime())

NP = 100
INIT_CR_MEAN = 0.3
CR_STD = 0.1
INIT_F_MEAN = 0.1
F_SCALING_FACTOR = 0.1
H = 5
DIFFERENCE_NUM = 1
MAX_FES = 1e5

# Epsilon method require parameters below
INIT_EPSILON_POSITION = 0.2
EPSILON_MAX_GENERATION = 0.2
CP = 7

CONSTRAINED_HANDLING_METHOD = 'EpsilonMethod'  # 'Penalty', 'SuperiorityMethod', 'EpsilonMethod', 'PenaltyEpsilon'
assert CONSTRAINED_HANDLING_METHOD in ['Penalty', 'SuperiorityMethod', 'EpsilonMethod', 'PenaltyEpsilon'], \
    'Method should be one of Penalty, SuperiorityMethod, EpsilonMethod, PenaltyEpsilon'


class SHADE:
    def __init__(self, problem_num, population_num, init_cr_mean, cr_std, init_f_mean, f_scaling_factor, h, 
                 difference_num, max_fes, init_epsilon_position, epsilon_max_generation, cp):
        
        # Regular differential parameters
        self.problem = Problem(problem_num)
        self.population_num = population_num
        max_eval_num = np.ceil(self.problem.dim / 10) * max_fes
        self.generation = int(max_eval_num / self.population_num)
        self.dim = self.problem.dim
        self.low_bounds = np.array(self.problem.low_bounds)[np.newaxis, :]
        self.low_bounds = np.repeat(self.low_bounds, self.population_num, axis=0)
        self.up_bounds = np.array(self.problem.up_bounds)[np.newaxis, :]
        self.up_bounds = np.repeat(self.up_bounds, self.population_num, axis=0)
        self.x = self.low_bounds + np.random.rand(self.population_num, self.dim) * (self.up_bounds - self.low_bounds)
        self.u = self.x.copy()
        self.v = self.x.copy()
        self.x_eval_result = self.constraint_handling.initialization(self.x)
        self.u_eval_result = self.x_eval_result.copy()
        self.best_feasible_solution, self.best_feasible_fitness, self.eval_result, self.violation = \
            np.nan, np.nan, np.zeros([self.population_num, self.dim]), np.zeros([self.population_num, self.dim])
        self.difference_num = difference_num
        
        # Adaptive parameters
        self.m_f = np.array([init_f_mean for i in range(self.h)])
        self.m_cr = np.array([init_cr_mean for i in range(self.h)])
        self.h = h
        self.cr_std = cr_std
        self.f_scaling_factor = f_scaling_factor
        self.archive = self.x[np.random.randint(0, self.population_num), :][np.newaxis, :]
        
        self.current_generation = 0

        # tmp_sort_mat is not in original SHADE just using for generating x_r
        self.tmp_sort_mat = np.zeros([self.population_num, self.population_num - 1])
        for i in range(self.population_num):
            sample_pool_tmp = np.arange(0, self.population_num)
            sample_pool_tmp = np.delete(sample_pool_tmp, i)
            self.tmp_sort_mat[i] = sample_pool_tmp.copy()
        
        if CONSTRAINED_HANDLING_METHOD is 'Penalty':
            self.constraint_handling = Penalty(self.problem)
        elif CONSTRAINED_HANDLING_METHOD is 'SuperiorityMethod':
            self.constraint_handling = SuperiorityMethod(self.problem, self.population_num)
        elif CONSTRAINED_HANDLING_METHOD is 'EpsilonMethod':
            self.constraint_handling = EpsilonMethod(problem=self.problem, num_popn=self.population_num, 
                                                     init_epsilon_position=init_epsilon_position, 
                                                     epsilon_max_generation=epsilon_max_generation, cp=cp)
        elif CONSTRAINED_HANDLING_METHOD is 'PenaltyEpsilon':
            self.constraint_handling = PenaltyEpsilon(self.problem)



    def de_current_to_pbest(self, factor):
        best_p_select_range = np.random.uniform(2 / self.population_num, 0.2)
        if CONSTRAINED_HANDLING_METHOD in ['SuperiorityMethod', 'EpsilonMethod']:
            sort_group_index = self.constraint_handling.sort_by_superiority(self.x_eval_result)[
                               0: int(best_p_select_range * self.population_num)]
        else:
            sort_group_index = np.argsort(self.x_eval_result)[0: int(best_p_select_range * self.population_num)]
        member_num = 2 * self.difference_num
        x_r = np.zeros([member_num, self.population_num, self.dim], np.float32)
        x_best_group = self.x[sort_group_index].copy()
        best_rand_index = np.random.choice(np.arange(0, len(sort_group_index)), self.population_num)
        x_best = x_best_group[best_rand_index]
        for i in range(member_num-1):
            x_r_index_tmp = self.tmp_sort_mat[(np.arange(self.population_num), np.random.randint(0, self.dim, self.population_num))].astype(
                np.int64)
            x_r[i] = self.x[x_r_index_tmp]
        archive_random_pool = np.array([np.arange(self.dim) for _ in range(len(self.archive))])
        x_r_index_tmp = self.tmp_sort_mat[(np.arange(self.population_num), np.random.randint(0, self.dim, self.population_num))].astype(np.int64)
        x_a_r_index_tmp = archive_random_pool[(np.arange(len(self.archive)),
                                               np.random.randint(0, self.dim, len(self.archive)))].astype(np.int64)
        x_a_r_index = np.concatenate([x_a_r_index_tmp, x_r_index_tmp], 0)
        np.random.shuffle(x_a_r_index)
        x_r[member_num-1] = self.x[x_a_r_index[:self.population_num]]
        self.v = self.x + factor * (x_best - self.x)
        for i in range(self.difference_num):
            self.v += factor * (x_r[i * 2] - x_r[i * 2 + 1])

    def uniform_crossover(self, cr):
        self.u = np.copy(self.x)
        r_xover = np.random.rand(self.population_num, self.dim)
        xover_index = np.where(r_xover <= cr)
        self.u[xover_index] = self.v[xover_index].copy()
        j_rand = np.random.randint(0, self.dim, size=self.population_num)
        self.u[(np.arange(self.population_num), j_rand)] = self.v[(np.arange(self.population_num), j_rand)].copy()
        self.u = np.clip(self.u, self.low_bounds, self.up_bounds)

    def selection(self):
        self.u_eval_result, replace_index, success_index = self.constraint_handling.update(self.u, self.x,
                                                                                           self.x_eval_result,
                                                                                           self.current_generation,
                                                                                           self.generation)
        self.x[replace_index] = self.u[replace_index].copy()
        self.x_eval_result[replace_index] = self.u_eval_result[replace_index].copy()
        self.archive = np.concatenate([self.archive, self.x[success_index]], 0)
        if len(self.archive) > self.population_num:
            rand_index = np.arange(0, len(self.archive))
            np.random.shuffle(rand_index)
            self.archive = self.archive[rand_index[:self.population_num]]
        return replace_index, success_index

    def update_adaptive_para(self, f, cr, success_index):
        if len(success_index[0]) > 0:
            f_set = np.mean(f[success_index], axis=1)
            f_m = np.sum(f_set ** 2) / np.sum(f_set)
            self.m_f = np.delete(self.m_f, 0)
            self.m_f = np.append(self.m_f, f_m)
            cr_m = np.mean(cr[success_index])
            self.m_cr = np.delete(self.m_cr, 0)
            self.m_cr = np.append(self.m_cr, cr_m)
        else:
            self.m_cr = np.delete(self.m_cr, 0)
            self.m_cr = np.append(self.m_cr, self.m_cr[-1])
            self.m_f = np.delete(self.m_f, 0)
            self.m_f = np.append(self.m_f, self.m_f[-1])

    def process(self):
        memory_rand_index = np.random.randint(0, self.h)
        f = np.random.standard_cauchy([self.population_num, 1]) * self.f_scaling_factor + self.m_f[memory_rand_index]
        regenerate_index = f <= 0
        while len(f[regenerate_index]) > 0:
            f[regenerate_index] = np.random.standard_cauchy(len(f[regenerate_index])) * self.f_scaling_factor + self.m_f[
                memory_rand_index]
            regenerate_index = f <= 0
        f = np.clip(f, 0, 1)
        f = np.repeat(f, self.dim, 1)
        self.de_current_to_pbest(f)
        cr = np.random.normal(self.m_cr[memory_rand_index], self.cr_std, [self.population_num, self.dim])
        cr = np.clip(cr, 0, 1)
        self.uniform_crossover(cr)
        _, success_index = self.selection()
        self.update_adaptive_para(f, cr, success_index)
        self.best_feasible_solution, self.best_feasible_fitness, self.eval_result, self.violation = \
            self.constraint_handling.get_result(self.x, self.x_eval_result)
        if (self.current_generation % 100 == 0) | (self.current_generation == self.generation - 1):
            self.print_result()
        return self.best_feasible_solution, self.best_feasible_fitness, np.min(self.eval_result), np.min(self.violation)

    def print_result(self):
        print(f'******{self.current_generation}********')
        print("best fitness:", self.best_feasible_fitness)
        print("best solution:", self.best_feasible_solution)
        print("best eval:", self.best_eval_result)
        print("smallest violation:", np.min(self.violation))
        print("mean violation:", np.mean(self.violation))


def run_shade(problem_num, population_num=40, init_cr_mean=0.5, cr_std=0.1, init_f_mean=0.1, f_scaling_factor=0.1, h=5,
              difference_num=1, max_fes=1e5, init_epsilon_position=0.1, epsilon_max_generation=0.2, cp=5):
    shade = SHADE(problem_num, population_num, init_cr_mean, cr_std, init_f_mean, f_scaling_factor, h,
                  difference_num, max_fes, init_epsilon_position, epsilon_max_generation, cp)
    for i in range(shade.generation):
        shade.current_generation = i
        best_solution, best_fitness, best_evaluation_result, best_violation = shade.process()
    return best_solution, best_fitness


if __name__ == '__main__':
    problem_number = 1
    best_feasible_solution, best_feasible_fitness = run_shade(problem_number, NP, INIT_CR_MEAN, CR_STD, INIT_F_MEAN,
                                                              F_SCALING_FACTOR, H, DIFFERENCE_NUM, MAX_FES,
                                                              INIT_EPSILON_POSITION, EPSILON_MAX_GENERATION, CP)