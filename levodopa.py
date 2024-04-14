from typing import Tuple


from repast4py import core
from repast4py.parameters import create_args_parser, init_params

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
        """Saves the state of this Human as a Tuple.

        Used to move this Human from one MPI rank to another.

        Returns:
            The saved state of this Human.
        """
        return (self.uid, )