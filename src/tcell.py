from typing import Tuple

from repast4py import core, random

from repast4py.space import DiscretePoint as dpt

from antigen import Antigen
from dopamine import Dopamine

# Neuron subclasses repast4py.core.Agent. Subclassing Agent is a requirement for all Repast4Py agent implementations.
class TCell(core.Agent):
    """The Cytokine Agent

    Args:
        a_id: a integer that uniquely identifies this Cytokine on its starting rank
        rank: the starting MPI rank of this Cytokine.
    """
    # TYPE is a class variable that defines the agent type id the Cytokine agent. This is a required part of the unique agent id tuple.
    TYPE = 5

    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=TCell.TYPE, rank=rank)
        self.is_activated = False

    def save(self) -> Tuple:
        """Saves the state of this Human as a Tuple.

        Used to move this Human from one MPI rank to another.

        Returns:
            The saved state of this Human.
        """
        return (self.uid, )
    
    def step(self, model, pt):

        if self.is_activated:
            if random.default_rng.integers(0, 100) >= 90:
                model.spawn_th1()
        else:
            at = dpt(0, 0)
            grid = model.periphery_grid
            nghs = model.ngh_finder.find(pt.x, pt.y)
            for ngh in nghs:
                for ngh in nghs:
                    at._reset_from_array(ngh)
                    for obj in grid.get_agents(at):
                        if obj.uid[1] == Antigen.TYPE:
                            self.is_activated = True
                            obj.num_encounters += 1
                        elif obj.uid[1] == Dopamine.TYPE:
                            self.is_activated = True
        