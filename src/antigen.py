from repast4py import core, random
import numpy as np
from typing import Dict, Tuple

# Antigen subclasses repast4py.core.Agent. Subclassing Agent is a requirement for all Repast4Py agent implementations.
class Antigen(core.Agent):
    """The Antigen Agent

    Args:
        a_id: a integer that uniquely identifies this Antigen on its starting rank
        rank: the starting MPI rank of this Antigen.
    """
    # TYPE is a class variable that defines the agent type id the Antigen agent. This is a required part of the unique agent id tuple.
    TYPE = 6
    OFFSETS_X = np.array([-2, 2])
    OFFSETS_Y = np.array([-1, 3])

    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=Antigen.TYPE, rank=rank)
        self.to_move = True
        self.num_encounters = 0

    def save(self) -> Tuple:
        """Saves the state of this Antigen as a Tuple.

        Used to move this Antigen from one MPI rank to another.

        Returns:
            The saved state of this Antigen.
        """
        return (self.uid, self.to_move)
    
    def walk(self, pt, model):
        
        if self.num_encounters > 2:
            self.to_move = False

        if self.to_move:
            # choose two elements from the OFFSET array
            # to select the direction to walk in the
            # x and y dimensions
            x_dir = random.default_rng.choice(Antigen.OFFSETS_X, size=1)
            y_dir = random.default_rng.choice(Antigen.OFFSETS_Y, size=1)
            model.move(self, pt.x + x_dir[0], pt.y + y_dir[0], "PERIPHERY")