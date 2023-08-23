import math

import numpy as np
from scipy import optimize as op


def sigmoid(x, sl=1.0, th=0.0):
    answer = 1.0/(1 + math.exp(-sl * (x - th)))
    assert 0 <= answer <= 1, f'{answer} for {x}'
    return answer


def solve_lp(payoff):
    """
    Function for returning mixed strategies of the first step of double oracle iterations.
    :param payoff: Two dimensinal array. Payoff matrix of the players.
    The row is defender and column is attcker. This is the payoff for row player.
    :return: List, mixed strategy of the attacker and defender at NE by solving maxmini problem.
    """
    # This implementation is based on page 88 of the book multiagent systems (Shoham etc.)
    # http://www.masfoundations.org/mas.pdf
    # n_action = payoff.shape[0]
    try:
        m, n = payoff.shape
    except ValueError:
        print(payoff)
    c = np.zeros(n)
    c = np.append(c, 1)
    A_ub = np.concatenate((payoff, np.full((m, 1), -1)), axis=1)
    b_ub = np.zeros(m)
    A_eq = np.full(n, 1)
    A_eq = np.append(A_eq, 0)
    A_eq = np.expand_dims(A_eq, axis=0)
    b_eq = np.ones(1)
    bound = ()
    for i in range(n):
        bound += ((0, None),)
    bound += ((None, None),)

    res_attacker = op.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bound, method="highs")

    return res_attacker.x[0:n]