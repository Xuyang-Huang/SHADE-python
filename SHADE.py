import numpy as np
from Problem_Set.problem_set_bounds_constrained import Problem


NP = 100
INIT_CR_MEAN = 0.5
CR_STD = 0.1
INIT_F_MEAN = 0.1
F_SCALING = 0.1
H = 20
DIFFERENCE_NUM = 1
MAX_FES = 5e4


class SHADE:
    def __init__(self, problem_num, population_num=40, init_cr_mean=0.5, cr_std=0.1, init_f_mean=0.1, 
                 f_scaling_factor=0.1, h=5, difference_num=1, max_fes=1e5):
        self.problem = Problem(problem_num)
        self.population_num = population_num
        self.h = h
        self.f_scaling = f_scaling_factor
        self.cr_std = cr_std
        max_eval_num = max_fes
        self.generation = int(max_eval_num / population_num)
        self.dim = self.problem.dim
        self.low_bounds = np.array(self.problem.low_bounds)[np.newaxis, :]
        self.low_bounds = np.repeat(self.low_bounds, self.population_num, axis=0)
        self.up_bounds = np.array(self.problem.up_bounds)[np.newaxis, :]
        self.up_bounds = np.repeat(self.up_bounds, self.population_num, axis=0)
        self.x = self.low_bounds + np.random.rand(population_num, self.dim) * (self.up_bounds - self.low_bounds)
        self.u = self.x.copy()
        self.v = self.x.copy()
        self.archive = self.x[np.random.randint(0, self.population_num), :][np.newaxis, :]
        self.difference_num = difference_num
        self.x_eval_result = self.problem.objective_function(self.x)
        self.u_eval_result = self.x_eval_result.copy()
        self.current_generation = 0
        self.m_f = np.array([init_f_mean for i in range(self.h)])
        self.m_cr = np.array([init_cr_mean for i in range(self.h)])
        self.sample_pool = np.zeros([self.population_num, self.population_num - 1])
        for i in range(self.population_num):
            sample_pool_tmp = np.arange(0, self.population_num)
            sample_pool_tmp = np.delete(sample_pool_tmp, i)
            self.sample_pool[i] = sample_pool_tmp.copy()

    def de_current_to_pbest(self, factor):
        best_p_select_range = np.random.uniform(2 / self.population_num, 0.2)
        sort_group_index = np.argsort(self.x_eval_result)[0: int(best_p_select_range * self.population_num)]
        member_num = 2 * self.difference_num
        x_r = np.zeros([member_num, self.population_num, self.dim], np.float32)
        x_best_group = self.x[sort_group_index].copy()
        best_rand_index = np.random.choice(np.arange(0, len(sort_group_index)), self.population_num)
        x_best = x_best_group[best_rand_index]
        for i in range(member_num-1):
            x_r_index_tmp = self.sample_pool[(np.arange(self.population_num), np.random.randint(0, self.dim, self.population_num))].astype(
                np.int64)
            x_r[i] = self.x[x_r_index_tmp]
        archive_random_pool = np.array([np.arange(self.dim) for _ in range(len(self.archive))])
        x_r_index_tmp = self.sample_pool[(np.arange(self.population_num), np.random.randint(0, self.dim, self.population_num))].astype(np.int64)
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
        self.u_eval_result = self.problem.objective_function(self.u)
        replace_index = np.where(self.u_eval_result <= self.x_eval_result)
        success_index = np.where(self.u_eval_result < self.x_eval_result)
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
        f = np.random.standard_cauchy([self.population_num, 1]) * self.f_scaling + self.m_f[memory_rand_index]
        regenerate_index = f <= 0
        while len(f[regenerate_index]) > 0:
            f[regenerate_index] = np.random.standard_cauchy(len(f[regenerate_index])) * self.f_scaling + self.m_f[
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
        if (self.current_generation % 100 == 0) | (self.current_generation == self.generation - 1):
            self.print_result()
        return np.min(self.x_eval_result), self.x[np.argmin(self.x_eval_result)]

    def print_result(self):
        print(f'******Current generation: {self.current_generation}*******')
        print("best eval:", np.min(self.x_eval_result))


def run_shade(problem_num, population_num=40, init_cr_mean=0.5, cr_std=0.1, init_f_mean=0.1, f_scaling_factor=0.1, h=5,
              difference_num=1, max_fes=1e5):
    shade = SHADE(problem_num, population_num, init_cr_mean, cr_std, init_f_mean, f_scaling_factor, h,
                  difference_num, max_fes)
    for i in range(shade.generation):
        shade.current_generation = i
        best_fitness, best_solution = shade.process()
    return best_fitness, best_solution


if __name__ == '__main__':
    problem_number = 0
    fitness, solutions = run_shade(problem_number, NP, INIT_CR_MEAN, CR_STD, INIT_F_MEAN, F_SCALING, H, DIFFERENCE_NUM,
                                   MAX_FES)

