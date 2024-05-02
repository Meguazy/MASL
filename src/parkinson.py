import os
import glob
import numpy as np
from typing import Dict, Tuple
from mpi4py import MPI
from dataclasses import dataclass

import loguru

import numba
from numba import int32, int64
from numba.experimental import jitclass

from repast4py import core, space, schedule, logging, random
from repast4py import context as ctx
from repast4py.parameters import create_args_parser, init_params

from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, OccupancyType

from antigen import Antigen
from astrocyte import Astrocyte
from cytokine import Cytokine
from debris import Debris
from dopamine import Dopamine
from levodopa import Levodopa
from microglia import Microglia
from neuron import Neuron
from tcell import TCell
from th1 import TH1
from th2 import TH2

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
        self.mo = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
        self.no = np.array([1, 1, 1, 0, 0, -1, -1, -1], dtype=np.int32)
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

agent_brain_cache = {}
def restore_brain_agent(agent_data: Tuple):
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
        if uid in agent_brain_cache:
            n = agent_brain_cache[uid]
        else:
            n = Neuron(uid[0], uid[2])
            agent_brain_cache[uid] = n

        # restore the agent state from the agent_data tuple
        n.is_alive = agent_data[1]
        n.is_alpha = agent_data[2]
        n.num_alpha = agent_data[3]
        n.num_misfolded = agent_data[4]
        n.alpha_ticks = agent_data[5]
        return n
    elif uid[1] == Microglia.TYPE:
        if uid in agent_brain_cache:
            m = agent_brain_cache[uid]
        else:
            m = Microglia(uid[0], uid[2])
            agent_brain_cache[uid] = m
        
        m.is_activated = agent_data[1]
        return m
    elif uid[1] == Astrocyte.TYPE:
        if uid in agent_brain_cache:
            a = agent_brain_cache[uid]
        else:
            a = Astrocyte(uid[0], uid[2])
            agent_brain_cache[uid] = a

        a.is_activated = agent_data[1]    
        return a
    elif uid[1] == Cytokine.TYPE:
        if uid in agent_brain_cache:
            return agent_brain_cache[uid]
        else:
            c = Cytokine(uid[0], uid[2])
            agent_brain_cache[uid] = c
            return c
    elif uid[1] == Antigen.TYPE:
        if uid in agent_brain_cache:
            return agent_brain_cache[uid]
        else:
            an = Antigen(uid[0], uid[2])
            agent_brain_cache[uid] = an
        
        an.to_move = agent_data[1]
        return an
    elif uid[1] == Debris.TYPE:
        if uid in agent_brain_cache:
            return agent_brain_cache[uid]
        else:
            d = Debris(uid[0], uid[2])
            agent_brain_cache[uid] = d
            return d
    elif uid[1] == TH1.TYPE:
        if uid in agent_brain_cache:
            return agent_brain_cache[uid]
        else:
            t = TH1(uid[0], uid[2])
            agent_brain_cache[uid] = t
        
        t.to_move = agent_data[1]
        return t
    elif uid[1] == Dopamine.TYPE:
        if uid in agent_brain_cache:
            return agent_brain_cache[uid]
        else:
            d = Dopamine(uid[0], uid[2])
            agent_brain_cache[uid] = d
            return d
    elif uid[1] == Levodopa.TYPE:
        if uid in agent_brain_cache:
            return agent_brain_cache[uid]
        else:
            l = Levodopa(uid[0], uid[2])
            agent_brain_cache[uid] = l
            return l

agent_periphery_cache = {}  
def restore_periphery_agent(agent_data: Tuple):
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
    if uid[1] == TCell.TYPE:
        if uid in agent_periphery_cache:
            return agent_periphery_cache[uid]
        else:
            tc = TCell(uid[0], uid[2])
            agent_periphery_cache[uid] = tc
            return tc
    elif uid[1] == Antigen.TYPE:
        if uid in agent_periphery_cache:
            return agent_periphery_cache[uid]
        else:
            ag = Antigen(uid[0], uid[2])
            agent_periphery_cache[uid] = ag
        
        ag.to_move = agent_data[1]
        return ag
    elif uid[1] == TH1.TYPE:
        if uid in agent_periphery_cache:
            return agent_periphery_cache[uid]
        else:
            ag = TH1(uid[0], uid[2])
            agent_periphery_cache[uid] = ag
            return ag
    elif uid[1] == TH2.TYPE:
        if uid in agent_periphery_cache:
            return agent_periphery_cache[uid]
        else:
            ag = TH2(uid[0], uid[2])
            agent_periphery_cache[uid] = ag
            return ag
    elif uid[1] == Levodopa.TYPE:
        if uid in agent_periphery_cache:
            return agent_periphery_cache[uid]
        else:
            ag = Levodopa(uid[0], uid[2])
            agent_periphery_cache[uid] = ag
            return ag
    elif uid[1] == Dopamine.TYPE:
        if uid in agent_periphery_cache:
            return agent_periphery_cache[uid]
        else:
            ag = Dopamine(uid[0], uid[2])
            agent_periphery_cache[uid] = ag
            return ag


@dataclass
class Neurons:
    """Dataclass used by repast4py aggregate logging to record
    the number of Humans and Zombies after each tick.
    """
    x: int = 0
    y: int = 0
    rank: int = 0

@dataclass
class Periphery:
    """Dataclass used by repast4py aggregate logging to record
    the number of Humans and Zombies after each tick.
    """
    x: int = 0
    y: int = 0
    rank: int = 0


class BloodBrainBarrier():

    def __init__(self):
        self.retained_antigens = []
        self.retained_th1s = []
        self.retained_levodopa = []

    def retain(self, ag, pt, type):
        if type == Antigen.TYPE:
            self.retained_antigens.append((ag, pt))
        elif type == TH1.TYPE:
            self.retained_th1s.append((ag, pt))
        elif type == Levodopa.TYPE:
            self.retained_levodopa.append((ag, pt))
    
    def release(self, agent_type: int):
        to_release = []
        if agent_type == Antigen.TYPE:
            to_release = self.retained_antigens.copy()
            self.retained_antigens = []
        elif agent_type == TH1.TYPE:
            to_release = self.retained_th1s.copy()
            self.retained_th1s = []
        elif agent_type == Levodopa.TYPE:
            to_release = self.retained_levodopa.copy()
            self.retained_levodopa = []

        return to_release

    
class Model:
    contexts = {}
    def __init__(self, comm, params):
        self.comm = comm
        self.contexts["brain"] = ctx.SharedContext(comm)
        self.contexts["peripheral"] = ctx.SharedContext(comm)
        
        self.rank = self.comm.Get_rank()

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        self.BBB = BloodBrainBarrier()
        
        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        self.brain_grid = space.SharedGrid('grid', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)
        self.periphery_grid = space.SharedGrid('grid', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)
        self.contexts["brain"].add_projection(self.brain_grid)
        self.contexts["peripheral"].add_projection(self.periphery_grid)
        
        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)

        #self.neurons = Neurons()
        #loggers = logging.create_loggers(self.neurons, op=MPI.SUM, rank=self.rank)
        self.brain_data_set = logging.TabularLogger(self.comm, params['brain_file'], ["tick", "agent_id", "agent_type", "x", "y", "rank"], delimiter=",")

        # self.peripheral_logs = Periphery()
        self.periphery_data_set = logging.TabularLogger(self.comm, params['periphery_file'], ["tick", "agent_id", "agent_type", "x", "y", "rank"], delimiter=",")

        self.neuron_status_data_set = logging.TabularLogger(self.comm, params['neuron_status_file'], ["tick", "agent_id", "is_alive", "is_alpha", "num_alpha", "num_misfolded", "alpha_ticks", "rank"], delimiter=",")
        self.microglia_status_data_set = logging.TabularLogger(self.comm, params['microglia_status_file'], ["tick", "agent_id", "is_activated", "rank"], delimiter=",")
        self.astrocyte_status_data_set = logging.TabularLogger(self.comm, params['astrocyte_status_file'], ["tick", "agent_id", "is_activated", "rank"], delimiter=",")

        self.world_size = comm.Get_size()
        self.occupied_brain_coords = []
        self.occupied_periphery_coords = []
        random.init(self.rank)

        self.brain_dopamine = 0
        self.carbidopa_effectiveness = params['carbidopa.effectiveness']
        self.id_counter = 0
        # Adding Neurons to the environment
        self.setup(Neuron.TYPE, params, "neuron.perc", "BRAIN")

        # Adding Microglia to the environment
        self.setup(Microglia.TYPE, params, "microglia.perc", "BRAIN")

        # Adding Astrocytes to the environment
        self.setup(Astrocyte.TYPE, params, "astrocyte.perc", "BRAIN")

        # Adding TCells to the environment
        self.setup(TCell.TYPE, params, "tcells.perc", "PERIPHERY")

        # Adding TH1s to the environment
        self.setup(TH1.TYPE, params, "th1.perc", "PERIPHERY")

        # Adding TH2s to the environment
        self.setup(TH2.TYPE, params, "th2.perc", "PERIPHERY") 

        if self.rank == 1 or self.rank == 3:
            # Adding Levodopa to the environment
            self.setup(Levodopa.TYPE, params, "levodopa.perc", "PERIPHERY")
    
    def get_carbidopa_perc(self):
        return self.carbidopa_effectiveness

    def setup(self, agent_type, params, param, env):
        total_count = int((params['world.width'] * params['world.height']) * params[param] / 100)
        pp_count = int(total_count / self.world_size)
        if self.rank < total_count % self.world_size:
            pp_count += 1
        self.add_agents(pp_count, agent_type, env)
    
    def add_agents(self, pp_count, type, context_id):
        local_bounds = self.brain_grid.get_local_bounds() if context_id == "BRAIN" else self.periphery_grid.get_local_bounds()
        ag = None
        for _ in range(pp_count):
            if type == Neuron.TYPE:
                ag = Neuron(((self.rank + 1) * 10000000) + self.id_counter, self.rank)
            elif type == Microglia.TYPE:
                ag = Microglia(((self.rank + 1) * 10000000) + self.id_counter, self.rank)
            elif type == Astrocyte.TYPE:
                ag = Astrocyte(((self.rank + 1) * 10000000) + self.id_counter, self.rank)
            elif type == TCell.TYPE:
                ag = TCell(((self.rank + 1) * 10000000) + self.id_counter, self.rank)
            elif type == TH1.TYPE:
                ag = TH1(((self.rank + 1) * 10000000) + self.id_counter, self.rank)
            elif type == TH2.TYPE:
                ag = TH2(((self.rank + 1) * 10000000) + self.id_counter, self.rank)
            elif type == Levodopa.TYPE:
                ag = Levodopa(((self.rank + 1) * 10000000) + self.id_counter, self.rank)
            
            x, y = self.generate_coords(context_id, local_bounds.xmin, local_bounds.xmin + local_bounds.xextent, local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)

            if context_id == "BRAIN":
                self.contexts["brain"].add(ag)
                self.move(ag, x, y, "BRAIN")
                self.occupied_brain_coords.append((x,y))
            elif context_id == "PERIPHERY":
                self.contexts["peripheral"].add(ag)
                self.move(ag, x, y, "PERIPHERY")
                self.occupied_periphery_coords.append((x,y))
            
            self.id_counter += 1

    def at_end(self):
        self.brain_data_set.close()
        self.periphery_data_set.close()
        self.neuron_status_data_set.close()
        self.microglia_status_data_set.close()
        self.astrocyte_status_data_set.close()

    def move(self, agent, x, y, env):
        if(env == "BRAIN"):
            self.brain_grid.move(agent, dpt(x, y))
        elif(env == "PERIPHERY"):
            self.periphery_grid.move(agent, dpt(x, y))            

    def step(self):
        tick = self.runner.schedule.tick
        self.contexts["brain"].synchronize(restore_brain_agent)
        self.contexts["peripheral"].synchronize(restore_periphery_agent)
        self.log_counts(tick)

        to_release = self.BBB.release(Antigen.TYPE)
        for item in to_release:
            antigen, pt = item
            self.contexts["peripheral"].add(antigen)
            self.move(antigen, pt.x, 0, "PERIPHERY")

        to_release = self.BBB.release(TH1.TYPE)
        for item in to_release:
            th1, pt = item
            self.contexts["brain"].add(th1)
            self.move(th1, pt.x, 39, "BRAIN")

        #to_release = self.BBB.release(Levodopa.TYPE)
        #for item in to_release:
        #    levodopa, pt = item
        #    self.contexts["brain"].add(levodopa)
        #    self.move(levodopa, pt.x, 39, "BRAIN")
        
        for ag in self.contexts["brain"].agents(Neuron.TYPE):
            release_antigen, pt = ag.step(model, self.brain_dopamine)
            if release_antigen:
                self.coords_release_agents(pt, "ANTIGEN")

        for ag in self.contexts["brain"].agents(Microglia.TYPE):
            release_cytos, pt = ag.step(model)
            if release_cytos:
                self.coords_release_agents(pt, "CYTOKINE")

        for ag in self.contexts["brain"].agents(Astrocyte.TYPE):
            release_cytos, pt = ag.step(model)
            if release_cytos:
                self.coords_release_agents(pt, "CYTOKINE")

        try:                
            for ag in self.contexts["brain"].agents(Antigen.TYPE):
                pt = self.brain_grid.get_location(ag)
                if pt.y == 39:               
                    self.contexts["brain"].remove(ag)                    
                    self.BBB.retain(ag, pt, Antigen.TYPE)
                self.move(ag, pt.x, pt.y + 4, "BRAIN")
        except:
            pass
        
        try:
            for ag in self.contexts["peripheral"].agents(Antigen.TYPE):
                pt = self.periphery_grid.get_location(ag)
                ag.walk(pt, model)
        except:
            pass
              
        try:
            nums = self.contexts["peripheral"].size([TH1.TYPE, TH2.TYPE])
            print(nums[TH1.TYPE]/nums[TH2.TYPE], self.rank, tick)
            if (nums[TH1.TYPE] / nums[TH2.TYPE]) > 1:
                dict = list(self.contexts["peripheral"].agents(TH1.TYPE)).copy()
                for ag in dict:
                    if ag.uid[1] == TH1.TYPE:
                        pt = self.periphery_grid.get_location(ag)
                        # print(ag.save(), pt, self.rank)
                        if pt.y == 0:
                                self.contexts["peripheral"].remove(ag)
                                self.BBB.retain(ag, pt, TH1.TYPE)
                        else:
                                self.move(ag, pt.x, pt.y - 2, "PERIPHERY")
        except:
            print("Error")

        try:
            for ag in self.contexts["brain"].agents():
                if ag.save()[0][1] == TH1.TYPE:
                    pt = self.brain_grid.get_location(ag)
                    ag.walk(model, pt)
        except:
            pass

        for ag in self.contexts["peripheral"].agents(TCell.TYPE):
            pt = self.periphery_grid.get_location(ag)
            ag.step(model, pt)

        to_turn = []
        try:
            for ag in self.contexts["peripheral"].agents(Levodopa.TYPE):
                pt = self.periphery_grid.get_location(ag)
                turn = ag.step(model, pt)
                if turn:
                    to_turn.append(ag)
                elif pt.y == 0:
                    self.remove_agent(ag, "PERIPHERY")
                    self.BBB.retain(ag, pt, Levodopa.TYPE)

            for agent in to_turn:
                pt = self.periphery_grid.get_location(agent)
                self.remove_agent(agent, "PERIPHERY")
                self.spawn_dopamine(pt)
                self.brain_dopamine += 1 
        except:
            pass


    def run(self):
        self.runner.execute()
    
    def remove_agent(self, agent, env):
        if env == "PERIPHERY":
            self.contexts["peripheral"].remove(agent)
        elif env == "BRAIN":
            self.contexts["brain"].remove(agent)

    def stick_coordinates(self, coord):
        c = coord
        if c>39:
            c = 39
        elif c<0:
            c = 0
        return c

    def coords_release_agents(self, pt, agent_type):
        nghs = model.ngh_finder.find(pt.x, pt.y)
        at = dpt(0, 0)
        for ngh in nghs:
            at._reset_from_array(ngh)

            x = self.stick_coordinates(at.x)
            y = self.stick_coordinates(at.y)

            if (x, y) not in self.occupied_brain_coords:           
                ag = None
                if agent_type == "ANTIGEN":
                    if random.default_rng.integers(0, 10) >= 7:
                        ag = Antigen(((self.rank + 1) * 10000000) + self.id_counter, self.rank)
                    else:    
                        ag = Debris(((self.rank + 1) * 10000000) + self.id_counter, self.rank)                        
                elif agent_type == "CYTOKINE":
                    ag = Cytokine(((self.rank + 1) * 10000000) + self.id_counter, self.rank)

                self.contexts["brain"].add(ag)
                self.move(ag, x, y, "BRAIN")
                self.occupied_brain_coords.append((x, y))
                self.id_counter += 1

    def spawn_th1(self):
        ag = TH1(((self.rank + 1) * 10000000) + self.id_counter, self.rank)
        self.contexts["peripheral"].add(ag)
        x, y = self.generate_coords("PERIPHERY", 0, 39, 0, 39)
        self.move(ag, x, y, "PERIPHERY")
        self.occupied_periphery_coords.append((x, y))
        self.id_counter += 1
    
    def spawn_dopamine(self, pt):
        ag = Dopamine(((self.rank + 1) * 10000000) + self.id_counter, self.rank)
        self.contexts["peripheral"].add(ag)
        self.move(ag, pt.x, pt.y, "PERIPHERY")
        self.id_counter += 1

    def generate_coords(self, context_id, x_min, x_max, y_min, y_max):

        x = random.default_rng.integers(x_min, x_max)
        y = random.default_rng.integers(y_min, y_max)
        if context_id == "BRAIN":
            while (x,y) in self.occupied_brain_coords:
                x = random.default_rng.integers(x_min, x_max)
                y = random.default_rng.integers(y_min, y_max)
        else:
            while (x,y) in self.occupied_periphery_coords:
                x = random.default_rng.integers(x_min, x_max)
                y = random.default_rng.integers(y_min, y_max)

        return x, y
    
    def log_counts(self, tick):
        #num_agents = self.context.size([Neuron.TYPE])#, Cytokine.TYPE])
        #self.counts.zombies = num_agents[Cytokine.TYPE]
        #print(num_agents)
        
        for ag in self.contexts["brain"].agents():
            pt = self.brain_grid.get_location(ag)
            saved = ag.save()
            self.brain_data_set.log_row(tick, saved[0][0], saved[0][1], pt.x, pt.y, self.rank)
            if saved[0][1] == Neuron.TYPE:
                self.neuron_status_data_set.log_row(tick, saved[0][0], saved[1], saved[2], saved[3], saved[4], saved[5], self.rank)
            elif saved[0][1] == Astrocyte.TYPE:
                 self.astrocyte_status_data_set.log_row(tick, saved[0][0], saved[1], self.rank)
            elif saved[0][1] == Microglia.TYPE:
                 self.microglia_status_data_set.log_row(tick, saved[0][0], saved[1], self.rank)
        
        for ag in self.contexts["peripheral"].agents():
            pt = self.periphery_grid.get_location(ag)
            self.periphery_data_set.log_row(tick, ag.save()[0][0], ag.save()[0][1], pt.x, pt.y, self.rank)


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
