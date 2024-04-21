from typing import Tuple

from repast4py import core

# Antigen subclasses repast4py.core.Agent. Subclassing Agent is a requirement for all Repast4Py agent implementations.
class Debris(core.Agent):
    """The Debris Agent

    Args:
        a_id: a integer that uniquely identifies this Debris on its starting rank
        rank: the starting MPI rank of this Debris.
    """
    # TYPE is a class variable that defines the agent type id the Debris agent. This is a required part of the unique agent id tuple.
    TYPE = 7

    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=Debris.TYPE, rank=rank)

    def save(self) -> Tuple:
        """Saves the state of this Debris as a Tuple.

        Used to move this Debris from one MPI rank to another.

        Returns:
            The saved state of this Debris.
        """
        return (self.uid, )