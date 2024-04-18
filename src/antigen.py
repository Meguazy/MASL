from typing import Tuple

from repast4py import core

# Antigen subclasses repast4py.core.Agent. Subclassing Agent is a requirement for all Repast4Py agent implementations.
class Antigen(core.Agent):
    """The Antigen Agent

    Args:
        a_id: a integer that uniquely identifies this Antigen on its starting rank
        rank: the starting MPI rank of this Antigen.
    """
    # TYPE is a class variable that defines the agent type id the Antigen agent. This is a required part of the unique agent id tuple.
    TYPE = 1

    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=Antigen.TYPE, rank=rank)

    def save(self) -> Tuple:
        """Saves the state of this Antigen as a Tuple.

        Used to move this Antigen from one MPI rank to another.

        Returns:
            The saved state of this Antigen.
        """
        return (self.uid, )