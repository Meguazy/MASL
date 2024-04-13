import sys
import math
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

# Human subclasses repast4py.core.Agent. Subclassing Agent is a requirement for all Repast4Py agent implementations.
class Human(core.Agent):
    """The Human Agent

    Args:
        a_id: a integer that uniquely identifies this Human on its starting rank
        rank: the starting MPI rank of this Human.
    """
    # TYPE is a class variable that defines the agent type id the Human agent. This is a required part of the unique agent id tuple.
    TYPE = 0

    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=Human.TYPE, rank=rank)
        self.infected = False
        self.infected_duration = 0

    def save(self) -> Tuple:
        """Saves the state of this Human as a Tuple.

        Used to move this Human from one MPI rank to another.

        Returns:
            The saved state of this Human.
        """
        return (self.uid, self.infected, self.infected_duration)

    # We saw that zombies infect humans by calling the human’s infect() method. This method simply changes the infected state from False to True.
    def infect(self):
        self.infected = True

    # Given the similarities with the Zombie step() method only the relevant differences will be highlighted below.
    # See the comments on the Zombie class for more informations.
    def step(self):
        space_pt = model.space.get_location(self)
        # Initialize an alive variable that specifies whether or not this human is still alive (not a zombie).
        alive = True
        # If the human is infected, increment its infection duration.
        # If the infection duration is 10 or more, then set alive to False, indicating that this human should become a zombie.
        if self.infected:
            self.infected_duration += 1
            alive = self.infected_duration < 10

        if alive:
            grid = model.grid
            pt = grid.get_location(self)
            nghs = model.ngh_finder.find(pt.x, pt.y)  # include_origin=True)
    
            # Initialize a list minimum that will be used to store the current minimum number of zombie agents and the location(s) containing that minimum number.
            # The first element of the list stores the location(s), and the second the current minimum.
            # We set the initial minimum number of humans as sys.maxsize, the largest integer, so that anything below that counts as the new minimum value.
            minimum = [[], sys.maxsize]
            at = dpt(0, 0, 0)
            for ngh in nghs:
                at._reset_from_array(ngh)
                count = 0
                for obj in grid.get_agents(at):
                    if obj.uid[1] == Zombie.TYPE:
                        count += 1
                # Checks if the zombie count is less than the current minimum value, updating appropriately if so.
                if count < minimum[1]:
                    minimum[0] = [ngh]
                    minimum[1] = count
                elif count == minimum[1]:
                    minimum[0].append(ngh)

            min_ngh = minimum[0][random.default_rng.integers(0, len(minimum[0]))]
            # timer.stop_timer('zombie_finder')

            # if not np.all(min_ngh == pt.coordinates):
            # if min_ngh[0] != pt.coordinates[0] or min_ngh[1] != pt.coordinates[1]:
            # if not np.array_equal(min_ngh, pt.coordinates):
            if not is_equal(min_ngh, pt.coordinates):
                # Moves this human using the same mechanism as the zombie, but twice as far, 0.5 vs 0.25.
                direction = (min_ngh - pt.coordinates) * 0.5
                model.move(self, space_pt.x + direction[0], space_pt.y + direction[1])
        # Return a tuple of alive and the human’s current location in the continuous space.
        # This is returned to the Model class calling code, which will replace the human with a zombie if the human is no longer alive.
        return (not alive, space_pt)

# Zombie subclasses repast4py.core.Agent. Subclassing Agent is a requirement for all Repast4Py agent implementations.
class Zombie(core.Agent):
    # TYPE is a class variable that defines the agent type id for the Zombie agents. This is a required part of the unique agent id tuple.
    TYPE = 1

    # In order to uniquely identify the agent across all ranks in the simulation, the repast4py.core.Agent constructor takes the following three arguments: 
    # - an integer id that uniquely identifies an agent on the process where it was created, 
    # - a non-negative integer identifying the type of the agent, 
    # - the rank on which the agent is created.
    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Zombie.TYPE, rank=rank)

    # To move our zombie agent between processes, we must save its state.
    # Because the zombie agent does not have an internal state, our save method returns only the zombie agent’s unique id tuple.
    def save(self):
        return (self.uid,)

    def step(self):
        # The Model contains both the grid and continuous space in its grid and space fields. The model variable contains the instance of the Model class.
        grid = model.grid
        # Get the location of this zombie. This location is a Discrete Point.
        pt = grid.get_location(self)
        # Use the Model’s instance of a GridNghFinder to get the Moore neighborhood coordinates of the zombie’s current location.
        nghs = model.ngh_finder.find(pt.x, pt.y)  # include_origin=True)

        # Create a temporary DiscretePoint for use in the loop over the Moore neighborhood coordinates.
        at = dpt(0, 0)
        # Initialize a list maximum that will be used to store the current maximum number of human agents and the location(s) containing that maximum number.
        # The first element of the list stores the location(s), and the second the current maximum.
        # We set the initial maximum number of humans as -(sys.maxsize - 1), the smallest negative integer.
        # Consequently, if there are 0 neighboring humans then that becomes the new maximum, and the maximum list always contains at least one location.
        maximum = [[], -(sys.maxsize - 1)]
        # Iterate through all the neighboring locations to find the location(s) with the maximum number of humans.
        # For each neighbor location, we count the number of humans at that location, and if the total count is equal to or greater than the current maximum, update or reset the maximum list appropriately.
        for ngh in nghs:
            # Reset the the at DiscretePoint to the current neighbor coordinates. 
            # This will be used in the get_agents call to come, which takes a DiscretePoint argument and this converts the ngh numpy array to a DiscretePoint.
            at._reset_from_array(ngh)
            count = 0
            # Get all the agents at the current neighbor location, and iterate through those agents to count the number of humans.
            # Humans are those agents where the type component of their unique id tuple is equal to Human.ID.
            for obj in grid.get_agents(at):
                if obj.uid[1] == Human.TYPE:
                    count += 1
            # If the count is greater than the current maximum count, reset the maximum list to the current location, and maximum count.
            if count > maximum[1]:
                maximum[0] = [ngh]
                maximum[1] = count
            # If the count is equals to the current maximum count, then append the current location to the maximum list.
            elif count == maximum[1]:
                maximum[0].append(ngh)

        # Select one of the maximum neighbor locations at random using Repast4Py’s default random number generator.
        # See the docs for more details: https://repast.github.io/repast4py.site/apidoc/source/repast4py.random.html
        max_ngh = maximum[0][random.default_rng.integers(0, len(maximum[0]))]

        # Check if the maximum neighbor location is the zombie’s current location, using the is_equal function.
        # If not, move the zombie toward the selected location.
        if not is_equal(max_ngh, pt.coordinates):
            # Calculate the direction to move by subtracting the zombie’s current location from its desired location.
            # The zombie is only able to move a distance of 0.25 spaces per step (i.e., its speed is 0.25 spaces/tick), and so we multiply the direction vector by 0.25.
            direction = (max_ngh - pt.coordinates[0:3]) * 0.25
            # Get the zombie’s current location in the continuous space. As with the grid, the Model class instance model contains the continuous space over which the agents move.
            cpt = model.space.get_location(self)
            # timer.start_timer('zombie_move')
            # Move the zombie using the Model’s move() method to the location computed by adding the current location to the direction vector.
            model.move(self, cpt.x + direction[0], cpt.y + direction[1])
            # timer.stop_timer('zombie_move')

        # Get the zombie’s current location in grid space and infect any humans found at that location.
        pt = grid.get_location(self)
        for obj in grid.get_agents(pt):
            if obj.uid[1] == Human.TYPE:
                obj.infect()
                break


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
    if uid[1] == Human.TYPE:
        if uid in agent_cache:
            h = agent_cache[uid]
        else:
            h = Human(uid[0], uid[2])
            agent_cache[uid] = h

        # restore the agent state from the agent_data tuple
        h.infected = agent_data[1]
        h.infected_duration = agent_data[2]
        return h
    else:
        # note that the zombie has no internal state
        # so there's nothing to restore other than
        # the Zombie itself
        if uid in agent_cache:
            return agent_cache[uid]
        else:
            z = Zombie(uid[0], uid[2])
            agent_cache[uid] = z
            return z


@dataclass
class Counts:
    """Dataclass used by repast4py aggregate logging to record
    the number of Humans and Zombies after each tick.
    """
    humans: int = 0
    zombies: int = 0


class Model:

    def __init__(self, comm, params):
        self.comm = comm
        # Create a context to hold the agents and the network projection
        self.context = ctx.SharedContext(comm)
        # Get the rank that is executing this code, the current process rank
        self.rank = self.comm.Get_rank()

        # Initialize schedule runner
        self.runner = schedule.init_schedule_runner(comm)
        # Schedule the repeating event of Model.step, beginning at tick 1 and repeating every tick thereafter
        self.runner.schedule_repeating_event(1, 1, self.step)
        # Schedule the tick at which the simulation should stop, and events will no longer be executed
        self.runner.schedule_stop(params['stop.at'])
        # Schedule a simulation end event to occur after events have stopped
        self.runner.schedule_end_event(self.at_end)

        # Create a BoundingBox to initialize the size of the Cartesian spaces.
        # Its arguments are the minimum x coordinate, the extent of the x dimension, and then the same for the y and z dimensions.
        # Here we create a 2D box (the z extent is 0) starting at (0,0) and extending for params['world.width'] in the x dimension and params['world.height'] in the y dimension.
        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        # Create the grid projection. repast4py.space.SharedGrid takes a name, its bounds, its border, and occupancy types, as well as a buffer size, and a MPI communicator as arguments.
        # SharedGrid API: https://repast.github.io/repast4py.site/apidoc/source/repast4py.space.html#repast4py.space.SharedGrid
        # The concept of a buffer is described here: https://repast.github.io/repast4py.site/guide/user_guide.html#_distributed_simulation
        self.grid = space.SharedGrid('grid', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)
        # Add the grid to the context so that it can be properly synchronized across processes
        self.context.add_projection(self.grid)
        # Create the space projection. repast4py.space.SharedCSpace takes a name, its bounds, its border, and occupancy types, as well as a buffer size, a MPI communicator, and a tree threshold as arguments.
        # SharedCSpace APIs: https://repast.github.io/repast4py.site/apidoc/source/repast4py.space.html#repast4py.space.SharedCSpace
        self.space = space.SharedCSpace('space', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                        buffer_size=2, comm=comm, tree_threshold=100)
        # Add the space to the context so that it can be properly synchronized across processes
        self.context.add_projection(self.space)
        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)

        # Create the Counts instance that we use to record the number of humans and zombies on each rank
        self.counts = Counts()
        # Create a list of loggers that use self.counts as the source of the data to log, and that perform a cross process rank summation of that data.
        # The names argument is not specified, so the Counts field names will be used as column headers.
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        # Create a logging.ReducingDataSet from the list of loggers. params['counts_file'] is the name of the file to log to.
        self.data_set = logging.ReducingDataSet(loggers, self.comm, params['counts_file'])

        # Get the number of process ranks over which the simulation is distributed
        world_size = comm.Get_size()

        # Get the total number of Humans to create from the input parameters dictionary
        total_human_count = params['human.count']
        # Compute the number of Human agents per processor
        pp_human_count = int(total_human_count / world_size)
        # Increment the number of agents to create on this rank, if this rank’s id is less than the number of remaining agents to create.
        # This will assign each rank, starting with 0, an additional agent in order to reach the total when the total number of agents cannot be evenly divided among all the process ranks.
        if self.rank < total_human_count % world_size:
            pp_human_count += 1

        # Get the local bounds of the continuous space. Each rank is responsible for some part of the total area defined by the space’s bounding box.
        # For example, assuming 4 process ranks, each rank would be responsible for some quadrant of the space.
        # get_local_bounds returns the area that the calling rank is responsible for as a BoundingBox.
        local_bounds = self.space.get_local_bounds()
        # Iterate through the number of humans to be assigned to each rank
        for i in range(pp_human_count):
            # Create a human agent
            h = Human(i, self.rank)
            # Add the new human agent to the context
            self.context.add(h)
            # Choose a random x and y location within the current local bounds using repast4py’s default random number generator.
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            # Move the new human agent to that location, using Model.move.
            self.move(h, x, y)

        # The code for creating the zombie agents is nearly identical, except that the the zombie.count input parameter is used as the total number of agents to create, and a zombie agent is created rather than a human.
        total_zombie_count = params['zombie.count']
        pp_zombie_count = int(total_zombie_count / world_size)
        if self.rank < total_zombie_count % world_size:
            pp_zombie_count += 1

        for i in range(pp_zombie_count):
            zo = Zombie(i, self.rank)
            self.context.add(zo)
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(zo, x, y)
        
        # Set the next integer id for newly created zombies to the number of zombies created on this rank.
        # When a human becomes a zombie, this zombie_id is used as the id of that new zombie, and then incremented for the next time a human becomes a zombie.
        self.zombie_id = pp_zombie_count

    # Model.at_end runs when the simulation reaches its final tick and ends.
    # This method closes the data_set log, ensuring that any remaining unwritten data is written to the output file.
    def at_end(self):
        self.data_set.close()

    # Pass the move method the x and y coordinates in the space projection that the agent argument is moving to.
    def move(self, agent, x, y):
        # Move the agent to the specified point in the continuous space, creating a new ContinuousPoint from the x and y coordinates.
        self.space.move(agent, cpt(x, y))
        # Move the agent to the corresponding location in the grid space.
        # The grid takes a DiscretePoint as its location argument. To create one, we take the floor of the x and y coordinates, convert those to ints, and create a DiscretePoint from those ints.
        self.grid.move(agent, dpt(int(math.floor(x)), int(math.floor(y))))

    def step(self):
        # Get the current tick value from the schedule runner
        tick = self.runner.schedule.tick
        # Log the current number of humans and zombies by calling the log_counts method
        self.log_counts(tick)
        # Synchronize the state of the simulation across processes using the restore_agent function to restore any agents (Zombies and Humans) that have moved processes.
        self.context.synchronize(restore_agent)

        # Iterate over all the Zombie agents in the model, calling step on each one.
        for z in self.context.agents(Zombie.TYPE):
            z.step()

        # Create an empty list for collecting the dead humans and their current location. This is used later in step to replace the humans with zombies.
        dead_humans = []
        # Iterate over all the human agents in the model, calling step on each one.
        # Human.step returns a boolean that indicates whether or not the Human has died (and thus should become a Zombie), and the current location of that human.
        for h in self.context.agents(Human.TYPE):
            dead, pt = h.step()
            # If the human has died, then append it and its current location to the dead_humans list.
            if dead:
                dead_humans.append((h, pt))

        # Iterate over the dead human data, removing the human from the model, and replacing it with a zombie at its former location.
        for h, pt in dead_humans:
            model.remove_agent(h)
            model.add_zombie(pt)

    def run(self):
        # Start the simulation by executing the schedule which calls the scheduled methods at the appropriate times and frequency
        self.runner.execute()
    
    # Remove the agent from the context.
    def remove_agent(self, agent):
        self.context.remove(agent)

    # The final location of the human agent that just died is passed into the add_zombie method
    def add_zombie(self, pt):
        # Create a new zombie agent, using the zombie_id field instantiated in the constructor
        z = Zombie(self.zombie_id, self.rank)
        # Increment the zombie_id to create the id for the next created zombie
        self.zombie_id += 1
        # Add the newly created zombie to the Model’s context
        self.context.add(z)
        # Move the zombie to the location of the dead human the zombie is replacing
        self.move(z, pt.x, pt.y)

    def log_counts(self, tick):
        # Get the current number of zombies and humans and log
        # Get the number of agents of the specified types currently in the context.
        # context.size takes a list of agent type ids and returns a dictionary where the type ids are the keys and the values are the number of agents of that type.
        num_agents = self.context.size([Human.TYPE, Zombie.TYPE])
        # Set the self.counts.humans to the number of humans
        self.counts.humans = num_agents[Human.TYPE]
        # Set the self.counts.zombies to the number of zombies
        self.counts.zombies = num_agents[Zombie.TYPE]
        # Log the values for the specified tick. This will sum the values in self.counts across all the ranks and log the results.
        self.data_set.log(tick)

        # Do the cross-rank reduction manually and print the result
        if tick % 10 == 0:
            human_count = np.zeros(1, dtype='int64')
            zombie_count = np.zeros(1, dtype='int64')
            self.comm.Reduce(np.array([self.counts.humans], dtype='int64'), human_count, op=MPI.SUM, root=0)
            self.comm.Reduce(np.array([self.counts.zombies], dtype='int64'), zombie_count, op=MPI.SUM, root=0)
            if (self.rank == 0):
                print("Tick: {}, Human Count: {}, Zombie Count: {}".format(tick, human_count[0], zombie_count[0]),
                      flush=True)


def run(params: Dict):
    """Creates and runs the Zombies Model.

    Args:
        params: the model input parameters
    """
    # Use the global keyword to indicate that model refers to the module level model variable and not a local variable
    global model
    # Create the model instance, passing the Model constructor the MPI world communicator and the input parameters dictionary
    model = Model(MPI.COMM_WORLD, params)
    model.run()


if __name__ == "__main__":
    # Create the default command line argument parser
    parser = create_args_parser()
    # Parse the command line into its arguments using that default parser
    args = parser.parse_args()
    # Create the model input parameters dictionary from those arguments using parameters.init_params
    params = init_params(args.parameters_file, args.parameters)
    # Call the run function to run the simulation
    run(params)

# The simulation is run from the command line. For example, from within the examples/zombies directory:
# mpirun -n 4 python zombies.py zombie_model.yaml