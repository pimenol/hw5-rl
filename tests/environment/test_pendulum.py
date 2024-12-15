import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
from environment.PendulumEnv import PendulumEnv


def test_pendulum():
    np.random.seed(0)
    cfg = {"N": 2, "vis": True}
    env = PendulumEnv(cfg)
    time = 512
    s = env.vector_reset()
    for i in range(time):
        actions = np.random.rand(cfg['N'], 1) * 2 - 1  # random actions sampled from [-1, 1]
        s, r = env.vector_step(actions)
    env.close()
    assert np.allclose(s[0], np.array([-0.6311882, -1.6342102, -0.5303053, 13.533856], dtype=np.float32)), "Env produced wrong state"
    assert np.allclose(s[1], np.array([-3.9639071e-02, -5.2952179e+01, 1.8349210e+00, 1.3122753e+00],
                                      dtype=np.float32)), "Env produced wrong state"


if __name__ == '__main__':
    assert test_pendulum(), "Pendulum test failed"
