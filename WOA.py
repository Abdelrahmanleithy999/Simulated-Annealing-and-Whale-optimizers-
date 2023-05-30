#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/28 19:21
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : WOA.py
# @Statement : The whale optimization algorithm
# @Reference : Mirjalili S, Lewis A. The whale optimization algorithm[J]. Advances in Engineering Software, 2016, 95: 51-67.
import random
import math
import matplotlib.pyplot as plt
import numpy


def obj(x):
    """
    The objective function of reservoir
    :param x:
    :return:
    """
    zd = x[0]
    zb = x[1]
    za = x[2]
    zf = x[3]

    itot = (zd * zb) / (za * zf)
    itarget = 1 / 6.931

    n = (itarget - itot) * (itarget - itot)
    return n


def boundary_check(x, lb, ub, dim):
    """
    Check the boundary
    :param x: a candidate solution
    :param lb: lower bound
    :param ub: upper bound
    :param dim: dimension
    :return:
    """
    for i in range(dim):
        if x[i] < lb[i]:
            x[i] = lb[i]
        elif x[i] > ub[i]:
            x[i] = ub[i]
    return x


def main(pop, lb, ub, iter):
    """
    The main function of WOA
    :param pop: the number of whales
    :param lb: the lower bound (list)
    :param ub: the upper bound (list)
    :param iter: the iteration number
    :return:
    """
    # Step 1. Initialization
    dim = len(ub)  # dimension
    pos = []
    score = []
    iter_best = []  # the best ever value of each iteration
    for _ in range(pop):
        temp_pos = [random.uniform(lb[i], ub[i]) for i in range(dim)]
        temp_score = obj(temp_pos)
        pos.append(temp_pos)
        score.append(temp_score)
    prey_score = min(score)
    prey_pos = pos[score.index(prey_score)].copy()
    con_iter = 0

    # Step 2. The main loop
    for t in range(iter):
        a = 2 - 2 * (t + 1) / iter
        for i in range(pop):
            A = 2 * a * random.random() - a
            C = 2 * random.random()
            if random.random() < 0.5:
                for j in range(dim):
                    if abs(A) < 1:  # Encircling pray
                        D = abs(C * prey_pos[j] - pos[i][j])
                        pos[i][j] = prey_pos[j] - A * D
                    else:  # Search for prey
                        rand_pos = random.choice(pos)
                        D = abs(C * rand_pos[j] - pos[i][j])
                        pos[i][j] = rand_pos[j] - A * D
            else:  # Bubble-net attacking method
                l = random.uniform(-1, 1)
                for j in range(dim):
                    D = abs(prey_pos[j] - pos[i][j])
                    pos[i][j] = D * math.exp(l) * math.cos(2 * math.pi * l) + prey_pos[j]

            # Update the prey information
            pos[i] = boundary_check(pos[i], lb, ub, dim)
            score[i] = obj(pos[i])
            if score[i] < prey_score:
                prey_score = score[i]
                prey_pos = pos[i].copy()
                con_iter = t
        iter_best.append(prey_score)
        prey_pos = [int(x) for x in prey_pos]

    # Step 3. Sort the results
    x = [i for i in range(iter)]
    plt.figure()
    plt.plot(x, iter_best, linewidth=2, color='blue')
    plt.xlabel('Iteration number')
    plt.ylabel('Global optimal value')
    plt.title('Convergence curve')
    plt.ticklabel_format(style='sci', scilimits=(0, 0))
    
    return {'best solution': prey_pos, 'best score': prey_score, 'convergence iteration': con_iter}


if __name__ == '__main__':
    pop = 1000
    lb = [12, 12, 12, 12]
    ub = [60, 60, 60, 60]
    iter = 1000
    print(main(pop, lb, ub, iter))
    plt.show()