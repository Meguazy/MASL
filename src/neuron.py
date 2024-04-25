from repast4py import core, random
from repast4py.space import DiscretePoint as dpt
from typing import Tuple

from cytokine import Cytokine
from levodopa import Levodopa
from th1 import TH1


# Neuron subclasses repast4py.core.Agent. Subclassing Agent is a requirement for all Repast4Py agent implementations.
class Neuron(core.Agent):
    """The Neuron Agent

    Args:
        a_id: a integer that uniquely identifies this Neuron on its starting rank
        rank: the starting MPI rank of this Neuron.
    """
    # TYPE is a class variable that defines the agent type id the Neuron agent. This is a required part of the unique agent id tuple.
    TYPE = 0

    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=Neuron.TYPE, rank=rank)
        self.is_alive = True
        self.is_alpha = False
        self.num_alpha = random.default_rng.integers(800, 1500)
        self.num_misfolded = 0
        self.alpha_ticks = 0

    def save(self) -> Tuple:
        """Saves the state of this Neuron as a Tuple.

        Used to move this Neuron from one MPI rank to another.

        Returns:
            The saved state of this Neuron.
        """
        return (self.uid, self.is_alive, self.is_alpha, self.num_alpha, self.num_misfolded, self.alpha_ticks)

    def die(self):
        self.is_alive = False

    # Given the similarities with the Zombie step() method only the relevant differences will be highlighted below.
    # See the comments on the Zombie class for more informations.
    def step(self, model):
        release_antigens = True
        grid = model.brain_grid
        pt = grid.get_location(self)
        at = dpt(0, 0)
        count_cytos, count_levos, count_deads, count_th1s = 0, 0, 0, 0
        if self.is_alive:            
            nghs = model.ngh_finder.find(pt.x, pt.y)
            for ngh in nghs:
                at._reset_from_array(ngh)            
                for obj in grid.get_agents(at):
                    if obj.uid[1] == Cytokine.TYPE:
                        count_cytos += 1
                    elif obj.uid[1] == Levodopa.TYPE:
                        count_levos += 1
                    elif obj.uid[1] == TH1.TYPE:
                        count_th1s += 1
                    elif self.is_alive and self.is_alpha == False and obj.uid[1] == Neuron.TYPE and obj.is_alive == False and obj.is_alpha:
                        obj.is_alpha = False
                        self.num_alpha += obj.num_alpha
                        self.num_misfolded += obj.num_misfolded
                        obj.num_alpha = 0
                        obj.num_misfolded = 0
                        count_deads += 1

            if self.is_alpha == False:
                self.alpha_ticks = 0
                
                number_sup = 100 * (count_levos + 1)
                if count_cytos >= 2 or count_th1s >= 2:
                    self.is_alive = False
                elif count_levos <= 5 and (count_deads >= 1 or random.default_rng.integers(0, number_sup) > number_sup - 4):
                    self.is_alpha = True                    
            else:                
                # print(self.num_misfolded/self.num_alpha)
                if count_cytos >= 2 or (self.num_misfolded / self.num_alpha) > 0.90 or count_th1s >= 2:
                    self.is_alive = False
                    return (release_antigens, pt)
                elif (self.alpha_ticks > int(14/(count_levos + 1))) and (float(self.num_misfolded) / float(self.num_alpha) < 0.45):
                    self.is_alpha = False

                self.num_alpha += int(self.num_alpha * random.default_rng.integers(1, 4) / 100)

                new_misfolded = int((self.num_alpha - self.num_misfolded) * random.default_rng.integers(35, 40) / 100)
                if (self.num_misfolded + new_misfolded > self.num_alpha):
                    self.num_misfolded = self.num_alpha
                else:
                    self.num_misfolded += new_misfolded

                self.num_misfolded -= int(self.num_misfolded * random.default_rng.integers(0, 25) / 100)

                self.alpha_ticks += 1

        return (not release_antigens, pt)