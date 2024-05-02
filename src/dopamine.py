from typing import Tuple


from repast4py import core, random

# Neuron subclasses repast4py.core.Agent. Subclassing Agent is a requirement for all Repast4Py agent implementations.
class Dopamine(core.Agent):
    """The Dopamine Agent

    Args:
        a_id: a integer that uniquely identifies this Dopamine on its starting rank
        rank: the starting MPI rank of this Dopamine.
    """
    # TYPE is a class variable that defines the agent type id the Dopamine agent. This is a required part of the unique agent id tuple.
    TYPE = 10

    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=Dopamine.TYPE, rank=rank)

    def save(self) -> Tuple:
        """Saves the state of this Dopamine as a Tuple.

        Used to move this Dopamine from one MPI rank to another.

        Returns:
            The saved state of this Dopamine.
        """
        return (self.uid, )
    
    def step(self, model, pt):
        turn = True

        if random.default_rng.integers(0, 100) >=  60:
            return turn
        else:
            model.move(self, pt.x, pt.y - 2, "PERIPHERY")
            return not turn