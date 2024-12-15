import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
from environment.WalkerEnv import WalkerEnv


def test_walker():
    np.random.seed(0)
    cfg = {"N": 2, "vis": True}
    env = WalkerEnv(cfg)
    time = 512
    s = env.vector_reset()
    for i in range(time):
        actions = np.random.rand(cfg['N'], 8) * 2 - 1  # random actions sampled from [-1, 1]
        s, r = env.vector_step(actions)
    env.close()
    assert np.allclose(s[0], np.array([0.20247714, 0.11854536, 0.19076154, 0.92268884, -0.09520236,
                                       0.05021781, -0.3702162, -0.5378907, 0.9968566, -0.26495117,
                                       -0.43535024, -0.27582476, 0.97349757, 0.5062795, 0.99271667,
                                       -0.25758946, 0.1848515, 0.2357152, -1.2221637, -1.5195539,
                                       -1.1062077, -0.48785236, -1.9811202, 2.731684, 6.242398,
                                       4.7640905, 5.5266542, -1.1236591, 1.4969984], dtype=np.float32)), "Walker env test produced wrong results"

    assert np.allclose(s[1], np.array([0.1777259, 0.03432934, 0.1798587, 0.23128757, 0.01390153,
                                       -0.028494, -0.9723687, -0.4998174, -0.955583, -0.38321453,
                                       0.9123256, -0.48040283, -0.7834734, 0.34123263, -1.0090684,
                                       0.30674702, -0.01122392, 0.20701738, 0.49942726, -0.8436993,
                                       0.62014717, -6.6677437, 4.6790705, 0.936359, -1.8181936,
                                       2.945833, 4.818913, 0.10336696, -0.69081795], dtype=np.float32)), "Walker env test produced wrong results"


if __name__ == '__main__':
    assert test_walker(), "Walker test failed"
