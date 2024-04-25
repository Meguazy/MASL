from repast4py import core
from repast4py.space import DiscretePoint as dpt
from typing import Tuple

from debris import Debris


# Neuron subclasses repast4py.core.Agent. Subclassing Agent is a requirement for all Repast4Py agent implementations.
class Microglia(core.Agent):
    """The Cytokine Agent

    Args:
        a_id: a integer that uniquely identifies this Cytokine on its starting rank
        rank: the starting MPI rank of this Cytokine.
    """
    # TYPE is a class variable that defines the agent type id the Cytokine agent. This is a required part of the unique agent id tuple.
    TYPE = 3

    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=Microglia.TYPE, rank=rank)
        self.is_activated = False

    def save(self) -> Tuple:
        """Saves the state of this Human as a Tuple.

        Used to move this Human from one MPI rank to another.

        Returns:
            The saved state of this Human.
        """
        return (self.uid, self.is_activated)
    
    def step(self, model):
        release_cytokine = True
        grid = model.brain_grid
        pt = grid.get_location(self)
        at = dpt(0, 0)
        count_antigent = 0
        if self.is_activated == False:
            nghs = model.ngh_finder.find(pt.x, pt.y)
            for ngh in nghs:
                at._reset_from_array(ngh)            
                for obj in grid.get_agents(at):
                    if obj.uid[1] == Debris.TYPE:
                        count_antigent += 1

            if count_antigent >= 2:
                self.is_activated = True
                return (release_cytokine, pt)
        
        return (not release_cytokine, pt)