from typing import Tuple

import numpy as np
from repast4py import core, random
from repast4py.space import DiscretePoint as dpt

# Neuron subclasses repast4py.core.Agent. Subclassing Agent is a requirement for all Repast4Py agent implementations.
class TH1(core.Agent):
    """The TH1 Agent

    Args:
        a_id: a integer that uniquely identifies this TH1 on its starting rank
        rank: the starting MPI rank of this TH1.
    """
    # TYPE is a class variable that defines the agent type id the TH1 agent. This is a required part of the unique agent id tuple.
    TYPE = 9
    OFFSETS_X = np.array([-2, 2, 3, -3, 1, -1])
    OFFSETS_Y = np.array([1, -4, -5])
    
    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=TH1.TYPE, rank=rank)
        self.to_move = True

    def save(self) -> Tuple:
        """Saves the state of this TH1 as a Tuple.

        Used to move this TH1 from one MPI rank to another.

        Returns:
            The saved state of this TH1.
        """
        return (self.uid, self.to_move)
    
    def walk(self, model, pt):

        if self.to_move:
            grid = model.brain_grid
            pt = grid.get_location(self)
            at = dpt(0, 0)
            nghs = model.ngh_finder.find(pt.x, pt.y)
            for ngh in nghs:
                at._reset_from_array(ngh)
                for obj in grid.get_agents(at):
                    if obj.uid[1] == 0:
                        is_alive = bool(obj.is_alive)
                        if is_alive == True:
                            self.to_move = False


        if self.to_move:
            # choose two elements from the OFFSET array
            # to select the direction to walk in the
            # x and y dimensions 
            x_dir = random.default_rng.choice(TH1.OFFSETS_X, size=1)
            y_dir = random.default_rng.choice(TH1.OFFSETS_Y, size=1)
            model.move(self, pt.x + x_dir[0], pt.y + y_dir[0], "BRAIN")