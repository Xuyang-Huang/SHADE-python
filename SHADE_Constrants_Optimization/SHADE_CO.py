#-- coding: utf-8 --
#@Time : 2021/3/16 23:46
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@File : SHADE_CO.py
#@Software: PyCharm



import numpy as np
from SHADE_Constraints_Optimization.constrained_problem_set import Problem
from SHADE_Constraints_Optimization.epsilon_method import EpsilonMethod


NP = 100  # Number of population
# CR is the crossover probability
INIT_CR_MEAN = 0.3  # Initial mean of uniform distribution to generate CR.
CR_STD = 0.1  # Standard deviation while generate CR obeying uniform distribution.
# F is the scaling factor when mutation
INIT_F_MEAN = 0.1  # Initial mean of uniform distribution to generate F.
F_SCALING_FACTOR = 0.1  # The scaling factor of Cauchy distribution for generating f.
H = 5  # Memory size
DIFFERENCE_NUM = 1  # This variable means how many pairs of individual difference when do the current-to-pbest mutation
MAX_FES = 1e5  # The max times number of function evaluation running

# Epsilon method require parameters below
INIT_EPSILON_POSITION = 0.2  # When INIT_EPSILON_POSITION=0.2 It means take the (0.2 * NP)th violation as the
# first generation epsilon--epsilon(0)
EPSILON_MAX_GENERATION = 0.2 # When EPSILON_MAX_GENERATION=0.2, epsilon(0.2 * max generation) = 0
CP = 7  # The exponent factor to decrease epsilon from initial epsilon--epsilon(0)

# Choose one constrained handling method
CONSTRAINED_HANDLING_METHOD = 'EpsilonMethod'  # 'Penalty', 'SuperiorityMethod', 'EpsilonMethod', 'PenaltyEpsilon'
assert CONSTRAINED_HANDLING_METHOD in ['Penalty', 'SuperiorityMethod', 'EpsilonMethod', 'PenaltyEpsilon'], \
    'Method should be one of Penalty, SuperiorityMethod, EpsilonMethod, PenaltyEpsilon'


class SHADE:
    """SHADE algorithm

    Just set some parameters and type problem in problem set py file,
    then run function 'run_shade' to get a optimization result.

    Attributes:

        'Regular DE parameters'
        problem: A class object generated from Problem(problem num in problem set py file).
        population_num: An integer of number of population.
        generation: A integer of the max generation to run SHADE.
        dim: A integer of the dimension of problem, it will be set in the problem set py file.
        low_bounds: A Numpy array shape is (population_num, dim),
            the lower bounds for each element in x, same dimension has same bounds.
        up_bounds: A Numpy array like low_bounds, but upper bounds.
        x: A Numpy array of the parent in DE, shape is (population_num, dim).
        u: A Numpy array shape like x, the offspring in DE.
        v: A Numpy array shape like x, the gene library generated for crossover.
        x_eval_result: A Numpy array shape is (population_num), the evaluation value of parent x solutions.
        u_eval_result: A Numpy array shape like x_eval_result, the evaluation value of offspring u solutions.
        best_feasible_solution: A Numpy array shape is (dim), the best feasible solution when violation < threshold
            (like 1e-5) for each generation, if no single violation < threshold, it will be nan.
        best_feasible_fitness: A Numpy array of the fitness of the best feasible solution when violation < threshold
            (like 1e-5) for each generation, if no single violation < threshold, it will be nan.
        violation: A Numpy array shape like x_eval_result, the current violation value of parent x solutions.
        difference_num: An integer of the differential term number in mutation strategy.

         'Adaptive parameters'
        h: An integer of memory size.
        m_f: A Numpy array shape is (memory size), the memory storage historical success scaling factor.
        m_cr: A Numpy array shape is like m_f,  the memory storage historical success crossover probability.
        cr_std: A floating number of standard deviation while generate CR obeying uniform distribution.
        f_scaling_factor: A floating number, the scaling factor of Cauchy distribution for generating f.
        archive: A Numpy array, storage recent bad offspring may reused.
        current_generation: An integer.

        _sort_mat: A Numpy array, not in original SHADE just using for generating x_r faster.
        constraint_handling: A class of specific constrained handling method you want to use.

    """

    def __init__(self, problem_num, population_num, init_cr_mean, cr_std, init_f_mean, f_scaling_factor, h,
                 difference_num, max_fes, init_epsilon_position, epsilon_max_generation, cp):
        # Regular DE parameters
        self.problem = Problem(problem_num)
        self.population_num = population_num
        max_eval_num = np.ceil(self.problem.dim / 10) * max_fes  # For different dim, different max_eval_num.
        self.generation = int(max_eval_num / self.population_num)
        self.dim = self.problem.dim
        self.low_bounds = np.array(self.problem.low_bounds)[np.newaxis, :]
        self.low_bounds = np.repeat(self.low_bounds, self.population_num, axis=0)
        self.up_bounds = np.array(self.problem.up_bounds)[np.newaxis, :]
        self.up_bounds = np.repeat(self.up_bounds, self.population_num, axis=0)
        self.x = self.low_bounds + np.random.rand(self.population_num, self.dim) * (self.up_bounds - self.low_bounds)
        self.u = self.x.copy()
        self.v = self.x.copy()
        self.best_feasible_solution, self.best_feasible_fitness = np.nan, np.nan
        self.violation = np.zeros([self.population_num, self.dim])
        self.difference_num = difference_num


        # Adaptive parameters
        self.h = h
        self.m_f = np.array([init_f_mean for i in range(self.h)])
        self.m_cr = np.array([init_cr_mean for i in range(self.h)])
        self.cr_std = cr_std
        self.f_scaling_factor = f_scaling_factor
        self.archive = self.x[np.random.randint(0, self.population_num), :][np.newaxis, :]

        self.current_generation = 0

        # _sort_mat is not in original SHADE just using for generating x_r faster
        self._sort_mat = np.zeros([self.population_num, self.population_num - 1])
        for i in range(self.population_num):
            sample_pool_tmp = np.arange(0, self.population_num)
            sample_pool_tmp = np.delete(sample_pool_tmp, i)
            self._sort_mat[i] = sample_pool_tmp.copy()

        if CONSTRAINED_HANDLING_METHOD is 'Penalty':
            self.constraint_handling = Penalty(self.problem)
        elif CONSTRAINED_HANDLING_METHOD is 'SuperiorityMethod':
            self.constraint_handling = SuperiorityMethod(self.problem, self.population_num)
        elif CONSTRAINED_HANDLING_METHOD is 'EpsilonMethod':
            self.constraint_handling = EpsilonMethod(problem=self.problem, num_popn=self.population_num,
                                                     init_epsilon_pos=init_epsilon_position,
                                                     epsilon_max_generation=int(epsilon_max_generation*self.generation),
                                                     cp=cp)
        elif CONSTRAINED_HANDLING_METHOD is 'PenaltyEpsilon':
            self.constraint_handling = PenaltyEpsilon(self.problem)
        self.constraint_handling.initialization(self.x)
        self.x_eval_result = self.problem.objective_function(self.x)
        self.u_eval_result = self.x_eval_result.copy()

    def de_current_to_pbest(self, factor):
        """generate and update mutation library v obeying DE/current-to-best/1 mutation strategy with an archive method.

        Find the pbest, generate differential terms in list x_r, if difference number is 1, get 2 Numpy array in
        x_r, finally v = x + f * (pbest - x) + f * (x_r0 - x_r1), where x_r1 is picked from archive.

        :param factor: A Numpy array shape is (population_num, 1), the scaling factor when generating v in mutation.
        :return: No return.
        """

        # Get pbest.
        # Superiority method has a different ranking for generating pbest.
        best_p_select_range = np.random.uniform(2 / self.population_num, 0.2)
        if CONSTRAINED_HANDLING_METHOD in ['SuperiorityMethod', 'EpsilonMethod']:
            sort_group_index = self.constraint_handling.sort_by_superiority(self.x_eval_result)[
                               0: int(best_p_select_range * self.population_num)]
        else:
            sort_group_index = np.argsort(self.x_eval_result)[0: int(best_p_select_range * self.population_num)]
        x_best_group = self.x[sort_group_index].copy()
        best_rand_index = np.random.choice(np.arange(0, len(sort_group_index)), self.population_num)
        x_best = x_best_group[best_rand_index]

        # Get x_r, the differential terms after current2pbest terms.
        member_num = 2 * self.difference_num
        x_r = np.zeros([member_num, self.population_num, self.dim], np.float32)
        for i in range(member_num-1):
            x_r_index_tmp = self._sort_mat[(np.arange(self.population_num),
                                               np.random.randint(0, self.dim, self.population_num))]
            x_r_index_tmp = x_r_index_tmp.astype(np.int64)
            x_r[i] = self.x[x_r_index_tmp]
        x_r_index_tmp = self._sort_mat[(np.arange(self.population_num),
                                           np.random.randint(0, self.dim, self.population_num))]
        x_r_index_tmp = x_r_index_tmp.astype(np.int64)
        # Get last x_r from union of archive and pbest
        archive_random_pool = np.array([np.arange(self.dim) for _ in range(len(self.archive))])
        x_a_r_index_tmp = archive_random_pool[(np.arange(len(self.archive)),
                                               np.random.randint(0, self.dim, len(self.archive)))]
        x_a_r_index_tmp = x_a_r_index_tmp.astype(np.int64)
        x_a_r_index = np.concatenate([x_a_r_index_tmp, x_r_index_tmp], 0)
        np.random.shuffle(x_a_r_index)
        x_r[member_num-1] = self.x[x_a_r_index[:self.population_num]]

        # Update v for crossover.
        self.v = self.x + factor * (x_best - self.x)
        for i in range(self.difference_num):
            self.v += factor * (x_r[i * 2] - x_r[i * 2 + 1])

    def uniform_crossover(self, cr):
        """Do uniform crossover

        After update v, using cr to generate offspring u, and do bound constrained handling.

        :param cr: A Numpy array, shape is like x, each dimension of each individual has a crossover probability value.
        :return: No return
        """
        self.u = np.copy(self.x)
        r_xover = np.random.rand(self.population_num, self.dim)
        xover_index = np.where(r_xover <= cr)
        self.u[xover_index] = self.v[xover_index].copy()
        j_rand = np.random.randint(0, self.dim, size=self.population_num)
        self.u[(np.arange(self.population_num), j_rand)] = self.v[(np.arange(self.population_num), j_rand)].copy()
        self.u = np.clip(self.u, self.low_bounds, self.up_bounds)

    def selection(self):
        """Do selection

        Compare offspring and parents, update population and archive.

        """
        # Need constraint handling method to give a replace and success index.
        # Success index is more strict than replace index usually, the < compare with the <=.
        self.u_eval_result = self.problem.objective_function(self.u)
        replace_index, success_index = self.constraint_handling.update(self.u, self.u_eval_result, self.x_eval_result,
                                                                       self.current_generation)
        self.x[replace_index] = self.u[replace_index].copy()
        self.x_eval_result[replace_index] = self.u_eval_result[replace_index].copy()
        self.archive = np.concatenate([self.archive, self.x[success_index]], 0)
        # If archive size is larger than population, randomly delete to size of population.
        if len(self.archive) > self.population_num:
            rand_index = np.arange(0, len(self.archive))
            np.random.shuffle(rand_index)
            self.archive = self.archive[rand_index[:self.population_num]]
        return replace_index, success_index

    def update_adaptive_para(self, f, cr, success_index):
        """Update memory

        :param f: A Numpy array shape is (population_num, 1), the scaling factor when generating v in mutation.
        :param cr: A Numpy array, shape is like x, each dimension of each individual has a crossover probability value.
        :param success_index: An tuple of index returned by np.where(), useful f and cr index we think.
        :return: No return.
        """
        if len(success_index[0]) > 0:
            # Using Lehmer mean for f.
            f_set = np.mean(f[success_index], axis=1)
            f_m = np.sum(f_set ** 2) / np.sum(f_set)
            self.m_f = np.delete(self.m_f, 0)
            self.m_f = np.append(self.m_f, f_m)
            # Using arithmetic mean for cr.
            cr_m = np.mean(cr[success_index])
            self.m_cr = np.delete(self.m_cr, 0)
            self.m_cr = np.append(self.m_cr, cr_m)
        # If no success f and cr, use last value as new value, delete earliest one.
        else:
            self.m_cr = np.delete(self.m_cr, 0)
            self.m_cr = np.append(self.m_cr, self.m_cr[-1])
            self.m_f = np.delete(self.m_f, 0)
            self.m_f = np.append(self.m_f, self.m_f[-1])

    def process(self):
        """Organize all steps

        The whole process of one iteration.

        :return: Optimization result in this iteration.
        """
        # Generate f.
        memory_rand_index = np.random.randint(0, self.h)
        f = np.random.standard_cauchy([self.population_num, 1]) * self.f_scaling_factor + self.m_f[memory_rand_index]
        regenerate_index = f <= 0
        while len(f[regenerate_index]) > 0:
            f[regenerate_index] = np.random.standard_cauchy(len(f[regenerate_index])) * self.f_scaling_factor\
                                  + self.m_f[memory_rand_index]
            regenerate_index = f <= 0
        f = np.clip(f, 0, 1)
        f = np.repeat(f, self.dim, 1)

        self.de_current_to_pbest(f)

        # Generate cr.
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
        print("best eval:", np.min(self.eval_result))
        print("best violation:", np.min(self.violation))
        print("mean violation:", np.mean(self.violation))


def run_shade(problem_num, population_num=40, init_cr_mean=0.5, cr_std=0.1, init_f_mean=0.1, f_scaling_factor=0.1, h=5,
              difference_num=1, max_fes=1e5, init_epsilon_position=0.1, epsilon_max_generation=0.2, cp=5):
    """Main function.

    :param: Just like SHADE.__init__()
    :return: Final solution, final fitness.
    """
    shade = SHADE(problem_num, population_num, init_cr_mean, cr_std, init_f_mean, f_scaling_factor, h,
                  difference_num, max_fes, init_epsilon_position, epsilon_max_generation, cp)
    for i in range(shade.generation):
        shade.current_generation = i
        best_solution, best_fitness, best_evaluation_result, best_violation = shade.process()
    return best_solution, best_fitness


if __name__ == '__main__':
    problem_number = 28
    best_feasible_solution, best_feasible_fitness = run_shade(problem_number, NP, INIT_CR_MEAN, CR_STD, INIT_F_MEAN,
                                                              F_SCALING_FACTOR, H, DIFFERENCE_NUM, MAX_FES,
                                                              INIT_EPSILON_POSITION, EPSILON_MAX_GENERATION, CP)
