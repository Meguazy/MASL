from typing import Tuple


from repast4py import core, random

# Neuron subclasses repast4py.core.Agent. Subclassing Agent is a requirement for all Repast4Py agent implementations.
class Levodopa(core.Agent):
    """The Levodopa Agent

    Args:
        a_id: a integer that uniquely identifies this Levodopa on its starting rank
        rank: the starting MPI rank of this Levodopa.
    """
    # TYPE is a class variable that defines the agent type id the Levodopa agent. This is a required part of the unique agent id tuple.
    TYPE = 2

    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=Levodopa.TYPE, rank=rank)

    def save(self) -> Tuple:
        """Saves the state of this Levodopa as a Tuple.

        Used to move this Levodopa from one MPI rank to another.

        Returns:
            The saved state of this Levodopa.
        """
        return (self.uid, )
    
    def step(self, model, pt):
        turn = True
        sup_number = 85
        if random.default_rng.integers(0, 100) >=  sup_number + int(sup_number * (10 * model.get_carbidopa_perc()) / 100):
            return turn
        else:
            model.move(self, pt.x, pt.y - 3, "PERIPHERY")
            return not turn