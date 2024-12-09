
from abc import ABC

class BasePlanner(ABC):

    def plan(self, model, starting_state, objective_function):
        """ Returns an action plan to maximize the objective function which evaluate trajectories"""
        raise NotImplementedError
