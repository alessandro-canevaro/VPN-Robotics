import unittest
import gym

from src.environment import env_register
from src.environment.gridworld import GridWorldEnv

class TestGridWorld(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGridWorld, self).__init__(*args, **kwargs)
        self.env = gym.make('GridWorld_Empty-v0', render_mode=None, size=16)

    def test_reset(self):
        obs, info = self.env.reset(agent=(1, 1), goal=(6, 6))
        print(obs, info)


if __name__ == "__main__":
    unittest.main()
