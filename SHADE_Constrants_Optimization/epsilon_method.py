import numpy as np


class EpsilonMethod:
    """Epsilon constrained handling method.

    Adding to constraints optimization GA, based on constraints, this class stores the violation, compares tow
    population and gives a replace index, or sort population.

    Attributes:
        h_num: An integer, number of equality constraints.
        g_num: An integer, number of inequality constraints.
        problem: A class of problem what we want to optimization, transfered from outside.
        num_popn: A integer, number of population.
        dim: A integer, number of problem dimension.
        init_epsilon_pos: A floating number, When init_epsilon_position=0.2 It means take the (0.2 * NP)th
            violation as the first generation epsilon--epsilon(0).
        epsilon_generation_max: An integer, lifetime of epsilon.
        cp: A floating number, the exponent factor to decrease epsilon from initial epsilon--epsilon(0)
        epsilon_init: A Numpy array shape is (h_num), initial epsilon for each equality constraints.
        new_popn_violation: A Numpy array shape is (num_popn), the violation of new generating solutions. For SHADE,
            they are offspring.
        best_popn_violation: A Numpy array shape is (num_popn), the violation of current best solutions. For SHADE,
            they are parents.
        best_popn_no_epsilon_violation: A Numpy array shape is (num_popn), the violation of current best solutions.
            without epsilon.
        new_popn_no_epsilon_violation: A Numpy array shape is (num_popn), the violation of new generating solutions.
            without epsilon.
        tolerance_thr: A floating number, the threshold we regard number as zero.
        best_h: A Numpy array shape is (num_popn, h_num), pure equality constraint values of current best solutions.
        best_g: A Numpy array shape is (num_popn, g_num), pure inequality constraint values of current best solutions.
        epsilon: A Numpy array shape is (h_num), current epsilon values for each h(x).

    """
    def __init__(self, problem, num_popn, init_epsilon_pos, epsilon_max_generation, cp, tolerance_thr=1e-5):
        self.h_num = problem.h_num
        self.g_num = problem.g_num
        self.problem = problem
        self.num_popn = num_popn
        self.dim = self.problem.dim
        self.init_epsilon_pos = init_epsilon_pos
        self.epsilon_generation_max = epsilon_max_generation
        self.cp = cp
        self.epsilon_init = np.zeros([self.h_num], np.float)
        self.new_popn_violation = np.zeros([self.num_popn])
        self.best_popn_violation = np.zeros([self.num_popn])
        self.best_popn_no_epsilon_violation = np.zeros([self.num_popn])
        self.new_popn_no_epsilon_violation = np.zeros([self.num_popn])
        self.tolerance_thr = tolerance_thr
        self.best_h = self.problem.constrain_h(np.zeros([self.num_popn, self.dim]))
        self.best_g = self.problem.constrain_g(np.zeros([self.num_popn, self.dim]))
        self.epsilon = self.epsilon_init

    def initialization(self, new_popn_init):
        """Initialize violation and epsilon.

        :param new_popn_init: A Numpy array shape is (num_popn, dim), the first generation of population.
        :return: No return.
        """
        h_tmp = np.abs(self.problem.constrain_h(new_popn_init))
        if self.h_num != 0:
            top_sigma_index = np.argsort(h_tmp, axis=1)[:, -int(self.num_popn * self.init_epsilon_pos)]
            self.epsilon_init = np.array(h_tmp)[(np.arange(0, self.h_num), top_sigma_index)]
        self.best_popn_violation, self.best_popn_no_epsilon_violation, self.best_h, self.best_g = \
            self.violation_function(new_popn_init)
        self.new_popn_violation = self.best_popn_violation.copy()
        self.new_popn_no_epsilon_violation = self.best_popn_no_epsilon_violation

    def violation_function(self, x):
        """Calculate violation.

        :param x: A Numpy array shape is (num_popn, dim), the input solutions.
        :return:
            p (A Numpy array shape is (num_popn), the violation of input solutions),
            no_epsilon_p (A Numpy array shape is (num_popn), the violation of input solutions but without epsilon),
            h (A Numpy array shape is (num_popn, h_num), equality constraints penalty),
            g (A Numpy array shape is (num_popn, g_num), inequality constraints penalty).
        """
        h = self.problem.constrain_h(x)  # Equality constraints penalty.
        g = self.problem.constrain_g(x)  # Inequality constraints penalty.
        p = 0
        no_epsilon_p = 0

        # Inequality constraints.
        if self.g_num != 0:
            for i in range(self.g_num):
                p += (np.maximum(0, g[i]) ** 2)
            g = np.array(g).transpose([1, 0])
        no_epsilon_p += p

        # Equality constraints.
        if self.h_num != 0:
            for i in range(self.h_num):
                p += ((np.maximum(0, np.abs(h[i]) - self.epsilon[i])) ** 2)
                no_epsilon_p += (h[i] ** 2)
            h = np.array(h).transpose([1, 0])
        return p, no_epsilon_p, h, g

    def update_violation(self):
        """Update last generation violation by new epsilon.

        For each generation, we need violation keeps same epsilon.

        :return: No return.
        """
        p = 0
        if self.g_num != 0:
            for i in range(self.g_num):
                p += (np.maximum(0, self.best_g[:, i]) ** 2)
        if self.h_num != 0:
            for i in range(self.h_num):
                p += ((np.maximum(0, np.abs(self.best_h[:, i]) - self.epsilon[i])) ** 2)
        return p

    def update(self, new_popn, new_popn_eval_result, best_popn_eval_result, current_generation):
        """Find better individuals between two solution groups

        Return a replace index and replace violation.

        :param new_popn: A Numpy array shape is (population_num, dim), new solutions, offspring in SHADE.
        :param new_popn_eval_result: A Numpy array shape is (population_num), the evaluation of new solutions.
        :param best_popn_eval_result: A Numpy array shape is (population_num), the evaluation of best solutions.
        :param current_generation: An integer of current generation.
        :return: replace_index(tuple,  index of Numpy array), success_index(tuple, index of Numpy array)
        """
        self.new_popn_violation, self.new_popn_no_epsilon_violation, new_h, new_g = self.violation_function(new_popn)
        self.best_popn_violation = self.update_violation()

        # 4 group using different sort method.
        both_infeasible_replace_index = (self.best_popn_violation >= self.tolerance_thr) & \
                                        (self.new_popn_violation >= self.tolerance_thr) & \
                                        (self.new_popn_violation <= self.best_popn_violation)
        strict_both_feasible_replace_index = (self.best_popn_no_epsilon_violation < self.tolerance_thr) & \
                                             (self.new_popn_no_epsilon_violation < self.tolerance_thr) & \
                                             (new_popn_eval_result <= best_popn_eval_result)
        both_fake_feasible_replace_index = ((self.best_popn_violation < self.tolerance_thr) &
                                            (self.new_popn_violation < self.tolerance_thr)) & \
                                           (~((self.best_popn_no_epsilon_violation < self.tolerance_thr) &
                                              (self.new_popn_no_epsilon_violation < self.tolerance_thr))) & \
                                           (new_popn_eval_result <= best_popn_eval_result)
        feasible_n_infeasible_replace_index = (self.best_popn_violation >= self.tolerance_thr) & \
                                              (self.new_popn_violation < self.tolerance_thr)
        replace_index = np.where(both_infeasible_replace_index |
                                 strict_both_feasible_replace_index |
                                 feasible_n_infeasible_replace_index |
                                 both_fake_feasible_replace_index)

        # Success index needs to be more strict than replace index. <= become <.
        both_infeasible_success_index = (self.best_popn_violation >= self.tolerance_thr) & \
                                        (self.new_popn_violation >= self.tolerance_thr) & \
                                        (self.new_popn_violation < self.best_popn_violation)
        both_feasible_success_index = (self.best_popn_no_epsilon_violation < self.tolerance_thr) & \
                                      (self.new_popn_no_epsilon_violation < self.tolerance_thr) & \
                                      (new_popn_eval_result < best_popn_eval_result)
        both_fake_feasible_success_index = ((self.best_popn_violation < self.tolerance_thr) &
                                            (self.new_popn_violation < self.tolerance_thr)) & \
                                           (~((self.best_popn_no_epsilon_violation < self.tolerance_thr) &
                                              (self.new_popn_no_epsilon_violation < self.tolerance_thr))) & \
                                           (new_popn_eval_result < best_popn_eval_result)
        success_index = np.where(both_infeasible_success_index |
                                 both_feasible_success_index |
                                 feasible_n_infeasible_replace_index |
                                 both_fake_feasible_success_index)

        # Some update about violation.
        if len(replace_index[0]) > 0:
            self.best_popn_violation[replace_index] = self.new_popn_violation[replace_index].copy()
            if self.h_num > 0:
                self.best_h[replace_index] = new_h[replace_index].copy()
            if self.g_num > 0:
                self.best_g[replace_index] = new_g[replace_index].copy()

        # Update epsilon.
        if self.h_num != 0:
            if current_generation < self.epsilon_generation_max:
                self.epsilon = self.epsilon_init * ((1 - current_generation/self.epsilon_generation_max) ** self.cp)
            else:
                self.epsilon = [0 for i in range(self.h_num)]

        return replace_index, success_index

    def get_result(self, best_popn, best_popn_eval_result):
        """Get best feasible solution and fitness.

        :param best_popn: A Numpy array shape is (population_num, dim), best solutions, which are parents in SHADE.
        :param best_popn_eval_result: A Numpy array shape is (population_num), the fitness of best solutions.
        :return:
            best_feasible_p (A Numpy array shape is (dim)),
            best_feasible_best_popn_fitness (A floating number of fitness of best solution),
            fitness (A Numpy array shape is (num_popn), all fitness),
            violation (A Numpy array shape is (num_popn), all violation)
        """
        violation = self.eval_violation_function(best_popn)
        fitness = self.problem.objective_function(best_popn)
        feasible_best_popn_index = np.where(violation <= self.tolerance_thr)

        # If there's no feasible solutions, assign np.nan.
        if len(feasible_best_popn_index[0]) != 0:
            feasible_p = best_popn[feasible_best_popn_index]
            feasible_best_popn_eval_result = best_popn_eval_result[feasible_best_popn_index]
            best_feasible_p = feasible_p[np.argmin(feasible_best_popn_eval_result)]
            best_feasible_best_popn_fitness = self.problem.objective_function(best_feasible_p[np.newaxis, :])[0]
        else:
            best_feasible_p, best_feasible_best_popn_fitness = np.nan, np.nan
        return best_feasible_p, best_feasible_best_popn_fitness, fitness, violation

    def compete_by_superiority(self, c1_eval_result, c2_eval_result, c1_index, c2_index):
        """Two groups compete by superiority sort method.

        Find index of better elements between two groups for tournament strategy in HCLPSO.

        :param c1_eval_result: A 1-D Numpy array, competitor 1.
        :param c2_eval_result: A 1-D Numpy array, competitor 2.
        :param c1_index: A tuple of Numpy index, index of competitor 1 in best population.
        :param c2_index: A tuple of Numpy index, index of competitor 2 in best population.
        :return: replace_index (tuple, index of Numpy array).
        """
        c1_violation = self.best_popn_violation[c1_index]
        c2_violation = self.best_popn_violation[c2_index]
        c1_no_violation = self.best_popn_no_epsilon_violation[c1_index]
        c2_no_violation = self.best_popn_no_epsilon_violation[c2_index]
        both_infeasible_replace_index = (c1_violation >= self.tolerance_thr) & \
                                        (c2_violation >= self.tolerance_thr) & \
                                        (c2_violation <= c1_violation)
        both_feasible_replace_index = (c1_no_violation < self.tolerance_thr) & \
                                      (c2_no_violation < self.tolerance_thr) & \
                                      (c2_eval_result <= c1_eval_result)
        both_fake_feasible_replace_index = ((c1_violation < self.tolerance_thr) &
                                            (c2_violation < self.tolerance_thr)) & \
                                           (~((c1_no_violation < self.tolerance_thr) &
                                              (c2_no_violation < self.tolerance_thr))) & \
                                           (c2_eval_result <= c1_eval_result)
        feasible_n_infeasible_replace_index = (c1_violation >= self.tolerance_thr) & \
                                              (c2_violation < self.tolerance_thr)
        replace_index = np.where(both_infeasible_replace_index |
                                 both_feasible_replace_index |
                                 feasible_n_infeasible_replace_index |
                                 both_fake_feasible_replace_index)
        return replace_index

    def sort_by_superiority(self, eval_result):
        """Sort solutions by superiority rules.

            The priority is: feasible, fake feasible and infeasible.
            The feasible would be sorted by fitness, the fake feasible would be sorted by fitness, the infeasible would
            be sorted by violation.

        :param eval_result: A Numpy array shape is (population_num), the evaluation values of best solutions.
        :return: sort_index (tuple, index of Numpy array)
        """
        feasible_sort_index = np.arange(self.num_popn)
        feasible_index = np.where(self.best_popn_no_epsilon_violation < self.tolerance_thr)
        feasible_eval_result = eval_result[feasible_index]
        feasible_sort_index = feasible_sort_index[feasible_index]
        feasible_sort_index = feasible_sort_index[np.argsort(feasible_eval_result)]
        if len(feasible_sort_index) < 1:
            feasible_sort_index = np.argsort(eval_result)[:1]

        fake_feasible_sort_index = np.arange(self.num_popn)
        fake_feasible_index = np.where((self.best_popn_violation < self.tolerance_thr) &
                                       (~(self.best_popn_no_epsilon_violation < self.tolerance_thr)))
        fake_feasible_eval_result = eval_result[fake_feasible_index]
        fake_feasible_sort_index = fake_feasible_sort_index[fake_feasible_index]
        fake_feasible_sort_index = fake_feasible_sort_index[np.argsort(fake_feasible_eval_result)]

        infeasible_sort_index = np.arange(self.num_popn)
        infeasible_index = np.where(self.best_popn_violation >= self.tolerance_thr)
        infeasible_violation = self.best_popn_violation[infeasible_index]
        infeasible_sort_index = infeasible_sort_index[infeasible_index]
        infeasible_sort_index = infeasible_sort_index[np.argsort(infeasible_violation)]

        sort_index = np.concatenate([feasible_sort_index, fake_feasible_sort_index, infeasible_sort_index], axis=0)
        return sort_index
