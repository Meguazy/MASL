from typing import Tuple

from repast4py import core

# Neuron subclasses repast4py.core.Agent. Subclassing Agent is a requirement for all Repast4Py agent implementations.
class TH2(core.Agent):
    """The TH2 Agent

    Args:
        a_id: a integer that uniquely identifies this TH2 on its starting rank
        rank: the starting MPI rank of this TH2.
    """
    # TYPE is a class variable that defines the agent type id the TH2 agent. This is a required part of the unique agent id tuple.
    TYPE = 8

    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=TH2.TYPE, rank=rank)

    def save(self) -> Tuple:
        """Saves the state of this TH2 as a Tuple.

        Used to move this TH2 from one MPI rank to another.

        Returns:
            The saved state of this TH2.
        """
        return (self.uid, )