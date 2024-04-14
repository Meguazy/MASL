import sys
import math
from venv import logger
import numpy as np
from typing import Dict, Tuple
from mpi4py import MPI
from dataclasses import dataclass

import numba
from numba import int32, int64
from numba.experimental import jitclass

from repast4py import core, space, schedule, logging, random
from repast4py import context as ctx
from repast4py.parameters import create_args_parser, init_params

from repast4py.space import ContinuousPoint as cpt
from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, OccupancyType

from cytokine import Cytokine
from levodopa import Levodopa
from microglia import Microglia
from astrocyte import Astrocyte

model = None

@numba.jit((int64[:], int64[:]), nopython=True)
def is_equal(a1, a2):
    return a1[0] == a2[0] and a1[1] == a2[1]

# Create a Numba class specification.
# The specification is a list of tuples, where each tuple consists of a field name, and the native type of that field.
# The names correspond to the field names in the class for which this is the specification.
spec = [
    # Create a tuple for the mo field with a NumPy array of 32-bit integers as its type.
    ('mo', int32[:]),
    ('no', int32[:]),
    # Create a tuple for the xmin field with a 32-bit integer type.
    ('xmin', int32),
    ('ymin', int32),
    ('ymax', int32),
    ('xmax', int32)
]

# Decorate GridNghFinder with jitclass passing our spec that defines the field types
@jitclass(spec)
class GridNghFinder:
    # Pass the global grid bounds to the constructor as x and y maximum and minimum values
    def __init__(self, xmin, ymin, xmax, ymax):
        # Create the mo and no offset arrays containing the specified 32-bit integers
        self.mo = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
        self.no = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int32)
        # Set the minimum and maximum possible x and y values from the passed in global grid bounds
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    # The find method takes a 2D location specified as x and y coordinates. This location is the location we want the neighboring coordinates of.
    def find(self, x, y):
        # Add the x offset array to the x coordinate, resulting in a new array xs that contains the neighboring x-axis coordinates.
        xs = self.mo + x
        # Add the y offset array to the y coordinate, resulting in a new array ys that contains the neighboring y-axis coordinates.
        ys = self.no + y

        # Compute the array indices in the xs array whose values are within the global x-axis bounds.
        xd = (xs >= self.xmin) & (xs <= self.xmax)
        # Keep only those values from xs, assigning that array to xs
        xs = xs[xd]
        # Do the same for the ys array. If an x value is out of bounds, we discard its corresponding y value.
        ys = ys[xd]

        # Compute the array indices in the ys array whose values are within the global y-axis bounds. Then reset xs and ys to contain only the values at those indices.
        yd = (ys >= self.ymin) & (ys <= self.ymax)
        xs = xs[yd]
        ys = ys[yd]

        # Combine the xs and ys indices with each other and a z-axis coordinate array of all zeros to create an array of arrays where the inner arrays are 3D points consisting of x, y, and z coordinates.
        # This 3 element array format is necessary to reset the repast4py.space.DiscretePoint at variable that is used in both the Zombie step, method and the Human step method.
        return np.stack((xs, ys, np.zeros(len(ys), dtype=np.int32)), axis=-1)

agent_cache = {}


def restore_agent(agent_data: Tuple):
    """Creates an agent from the specified agent_data.

    This is used to re-create agents when they have moved from one MPI rank to another.
    The tuple returned by the agent's save() method is moved between ranks, and restore_agent
    is called for each tuple in order to create the agent on that rank. Here we also use
    a cache to cache any agents already created on this rank, and only update their state
    rather than creating from scratch.

    Args:
        agent_data: the data to create the agent from. This is the tuple returned from the agent's save() method
                    where the first element is the agent id tuple, and any remaining arguments encapsulate
                    agent state.
    """
    uid = agent_data[0]
    # 0 is id, 1 is type, 2 is rank
    if uid[1] == Neuron.TYPE:
        if uid in agent_cache:
            n = agent_cache[uid]
        else:
            n = Neuron(uid[0], uid[2])
            agent_cache[uid] = n

        # restore the agent state from the agent_data tuple
        n.is_alive = agent_data[1]
        n.is_alpha = agent_data[2]
        n.num_alpha = agent_data[3]
        n.num_misfolded = agent_data[4]
        n.alpha_ticks = agent_data[5]
        return n
    elif uid[1] == Microglia.TYPE:
        if uid in agent_cache:
            return agent_cache[uid]
        else:
            c = Microglia(uid[0], uid[2])
            agent_cache[uid] = c
            return c
    elif uid[1] == Astrocyte.TYPE:
        if uid in agent_cache:
            return agent_cache[uid]
        else:
            c = Astrocyte(uid[0], uid[2])
            agent_cache[uid] = c
            return c


@dataclass
class Neurons:
    """Dataclass used by repast4py aggregate logging to record
    the number of Humans and Zombies after each tick.
    """
    x: int = 0
    y: int = 0
    rank: int = 0
    #zombies: int = 0

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
    
    # def __init__(self, a_id: int, rank: int, is_alive: bool, is_alpha: bool, num_alpha: int, num_misfolded: int, alpha_ticks: int):
    #     super().__init__(id=a_id, type=Neuron.TYPE, rank=rank)
    #     self.is_alive = is_alive
    #     self.is_alpha = is_alpha
    #     self.num_alpha = num_alpha
    #     self.num_misfolded = num_misfolded
    #     self.alpha_ticks = alpha_ticks

    def save(self) -> Tuple:
        """Saves the state of this Human as a Tuple.

        Used to move this Human from one MPI rank to another.

        Returns:
            The saved state of this Human.
        """
        return (self.uid, self.is_alive, self.is_alpha, self.num_alpha, self.num_misfolded, self.alpha_ticks)

    def die(self):
        self.is_alive = False

    # Given the similarities with the Zombie step() method only the relevant differences will be highlighted below.
    # See the comments on the Zombie class for more informations.
    def step(self):
        # Initialize an alive variable that specifies whether or not this human is still alive (not a zombie).
        alive = self.is_alive
        alpha = self.is_alpha
        grid = model.grid
        pt = grid.get_location(self)
        at = dpt(0, 0, 0)
        count_cytos = 0
        count_levos = 0
        count_deads = 0

        nghs = model.ngh_finder.find(pt.x, pt.y)  # include_origin=True)
        for ngh in nghs:
            at._reset_from_array(ngh)                
            for obj in grid.get_agents(at):
                if obj.uid[1] == Cytokine.TYPE:
                    count_cytos += 1
                elif obj.uid[1] == Levodopa.TYPE:
                    count_levos += 1
                elif self.is_alive and self.is_alpha == False and obj.uid[1] == Neuron.TYPE and obj.is_alive == False and obj.is_alpha:
                    obj.is_alpha = False
                    self.num_alpha += obj.num_alpha
                    self.num_misfolded += obj.num_misfolded
                    obj.num_alpha = 0
                    obj.num_misfolded = 0
                    count_deads += 1

        if alive and alpha == False:
            self.alpha_ticks = 0
            grid = model.grid
            pt = grid.get_location(self)
            
            number_sup = 100 * (count_levos + 1)
            if count_cytos >= 2:
                return (not alive, alpha, pt)
            elif count_levos <= 5 and (count_deads >= 1 or random.default_rng.integers(0, number_sup) > number_sup - 2):
                return (alive, not alpha, pt)
            
        elif alive and alpha:
            
            if (count_cytos >= 2 or (self.num_misfolded / self.num_alpha) > 0.95):
                return(not alive, alpha, pt)
            elif (self.alpha_ticks > int(14/(count_levos + 1))) and (float(self.num_misfolded) / float(self.num_alpha) < 0.45):
                return(alive, not alpha, pt)

            self.num_alpha += int(self.num_alpha * random.default_rng.integers(10 ,30) / 100)

            new_misfolded = int((self.num_alpha - self.num_misfolded) * random.default_rng.integers(10, 15) / 100)
            if (self.num_misfolded + new_misfolded > self.num_alpha):
                self.num_misfolded = self.num_alpha
            else:
                self.num_misfolded += new_misfolded

            self.num_misfolded -= int(self.num_misfolded * random.default_rng.integers(10, 100) / 100)

            self.alpha_ticks += 1

        return (alive, alpha, pt)
class Model:

    def __init__(self, comm, params):
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        self.grid = space.SharedGrid('grid', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)
        self.context.add_projection(self.grid)
        
        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)

        self.neurons = Neurons()
        #loggers = logging.create_loggers(self.neurons, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.TabularLogger(self.comm, params['counts_file'], ["tick", "agent_id", "agent_type", "x", "y", "rank"], delimiter=",")

        world_size = comm.Get_size()
        self.occupied_coords = []
        random.init(self.rank)

        # Adding Neurons to the environment
        total_neuron_count = int((params['world.width'] * params['world.height']) * params['neuron.perc'] / 100)
        pp_neuron_count = int(total_neuron_count / world_size)
        if self.rank < total_neuron_count % world_size:
            pp_neuron_count += 1
        self.add_agents(pp_neuron_count, Neuron.TYPE)

        # Adding Microglia to the environment
        total_microglia_count = int((params['world.width'] * params['world.height']) * params['microglia.perc'] / 100)
        pp_microglia_count = int(total_microglia_count / world_size)
        if self.rank < total_microglia_count % world_size:
            pp_microglia_count += 1
        self.add_agents(pp_microglia_count, Microglia.TYPE)

        # Adding Astrocytes to the environment
        total_astrocyte_count = int((params['world.width'] * params['world.height']) * params['astrocyte.perc'] / 100)
        pp_astrocyte_count = int(total_astrocyte_count / world_size)
        if self.rank < total_astrocyte_count % world_size:
            pp_astrocyte_count += 1
        self.add_agents(pp_astrocyte_count, Astrocyte.TYPE)
    
    def add_agents(self, pp_count, type):
        local_bounds = self.grid.get_local_bounds()
        ag = None
        for i in range(pp_count):
            if type == Neuron.TYPE:
                ag = Neuron(i, self.rank)
            elif type == Microglia.TYPE:
                ag = Microglia(i, self.rank)
            elif type == Astrocyte.TYPE:
                ag = Astrocyte(i, self.rank)
            
            x = random.default_rng.integers(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
            y = random.default_rng.integers(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            while (x,y) in self.occupied_coords:
                x = random.default_rng.integers(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
                y = random.default_rng.integers(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)

            self.context.add(ag)
            self.occupied_coords.append((x,y))
            self.move(ag, x, y)

    def at_end(self):
        self.data_set.close()

    def move(self, agent, x, y):
        self.grid.move(agent, dpt(x, y))

    def step(self):
        tick = self.runner.schedule.tick
        self.log_counts(tick)
        self.context.synchronize(restore_agent)

        # dead_humans = []
        # for n in self.context.agents(Neuron.TYPE):
        #     dead, alpha, pt = n.step()
        #     if dead:
        #         dead_humans.append((n, pt))

        # for h, pt in dead_humans:
        #     model.remove_agent(h)
        #     model.add_zombie(pt)

    def run(self):
        self.runner.execute()
    
    def remove_agent(self, agent):
        self.context.remove(agent)

    # def add_zombie(self, pt):
    #     z = Zombie(self.zombie_id, self.rank)
    #     self.zombie_id += 1
    #     self.context.add(z)
    #     self.move(z, pt.x, pt.y)

    def log_counts(self, tick):
        #num_agents = self.context.size([Neuron.TYPE])#, Cytokine.TYPE])
        #self.counts.zombies = num_agents[Cytokine.TYPE]
        
        for n in self.context.agents():
            pt = self.grid.get_location(n)
            self.data_set.log_row(tick, n.save()[0][0], n.save()[0][1], pt.x, pt.y, self.rank)
            print("Tick: {}, x: {}, y: {}, rank: {}".format(tick, pt.x, pt.y, self.rank), flush=True)
                    
        #if tick % 10 == 0:
            #human_count = np.zeros(1, dtype='int64')
            #zombie_count = np.zeros(1, dtype='int64')
            #self.comm.Reduce(np.array([self.counts.humans], dtype='int64'), human_count, op=MPI.SUM, root=0)
            #self.comm.Reduce(np.array([self.counts.zombies], dtype='int64'), zombie_count, op=MPI.SUM, root=0)
            #if (self.rank == 0):
                #print("Tick: {}, Human Count: {}, Zombie Count: {}".format(tick, human_count[0], zombie_count[0]), flush=True)
             #   print("Tick: {}, x: {}, y: {}".format(tick, human_count[0]), flush=True)


def run(params: Dict):
    """Creates and runs the Zombies Model.

    Args:
        params: the model input parameters
    """
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.run()


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    run(params)

# The simulation is run from the command line. For example, from within the examples/zombies directory:
# mpirun -n 4 python zombies.py zombie_model.yaml