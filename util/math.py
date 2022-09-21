import math


def sigmoid(x, sl=1.0, th=0.0):
    answer = 1.0/(1 + math.exp(-sl * (x - th)))
    assert 0 <= answer <= 1, f'{answer} for {x}'
    return answer
