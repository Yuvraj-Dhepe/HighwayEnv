import math
import random

import numpy as np


class Coordinate:
    """The Coordinate class represents spacial coordinates x and y"""

    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y

    def __add__(self, other):
        return Coordinate(x=self.x + other.x, y=self.y + other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"x: {self.x}, y: {self.y}"


class Direction(Coordinate):
    """
    The Direction class represents a direction on the current road space used by the road generator
    x and y can be eiter 1, 0 or -1 representing right/up (1) or left/down (-1) while 0 represents no movement in that
    direction
    """

    def __init__(self, x: int, y: int):
        super().__init__(x, y)

    def update_position(self, x=None, y=None):
        """
        Update direction in x or y direction if a value is given otherwise dont update
        :param x: new x direction
        :param y: new y direction
        :return: None
        """
        self.x = x if x is not None else self.x
        self.y = y if y is not None else self.y


class Randomizer:
    """
        Chose random actions based on the given
    """

    def __init__(self):
        self.seed = 6969
        np.random.seed(self.seed)
        self.p_straight: float = 0.5
        self.p_turn: float = 0.5

        assert self.p_turn + self.p_straight == 1

    def update_probabilities(self, new_p_straight: float):
        """
        During the process of randomizing tracks, the probabilities of straights and turns will differ depending on
        whether the last piece was a straight or a turn
        :param new_p_straight:
        """
        self.p_straight = new_p_straight
        self.p_turn = 1 - new_p_straight

    def get_random_action(self) -> int:
        """
        Make a bernoulli experiment Ber(self.straight). If the outcome is 1, then a straight line as next action is
        chosen and when its 0 a turn is chosen
        :return: 1 = next action is a straight line 0 = next action is a turn
        """
        choice: int = np.random.choice([0, 1], p=[self.p_turn, self.p_straight])

        # adjust probabilities
        new_p_straight = self.p_straight - 0.1 if choice else self.p_straight + 0.1
        self.update_probabilities(new_p_straight=new_p_straight)

        return choice

    def get_random_turn(self, p: list[float]) -> int:
        """
        get a random turn based on p
        :param p: a list of probabilities for the different choices
        """
        return np.random.choice([1, -1], p=p)


class RoadGenerator:
    """
        The main class of the road generator
    """

    def __init__(self, size: int):
        """
        Constructor
        :param size: the size of the space where the road is constructed

        Using the size, a matrix with dims=(size x size) will be constructed. Each element of that space represents
        a 10 by 10 pixels field for the actual road-network.

        e.g. space[0][0] represents the area from (x, y) = (0, 0) up tp (10, 10).
        """
        self.space_size: int = size - 1  # since space starts from 0
        self.space: np.array = np.zeros(shape=(size, size))
        self.initial_coordinate: Coordinate = Coordinate(x=2, y=2)

        self.current_position: Coordinate = Coordinate(x=2, y=2)
        # initialize the field at the initial coordinate
        self.space[self.initial_coordinate.y][self.initial_coordinate.y] = 1

        # defines the current direction of the movement where 1 is right and -1 is left
        # since the start is in the top left corner we start with going towards the right
        # for y coordinate 1 is up and - is down
        self.current_direction: Direction = Direction(x=1, y=0)
        self.preferred_direction: Direction = Direction(x=1, y=0)

        # define the corners of the field
        self._top_l: Coordinate = Coordinate(0, 0)
        self._top_r: Coordinate = Coordinate(0, self.space_size)
        self._bottom_l: Coordinate = Coordinate(self.space_size, 0)
        self._bottom_r: Coordinate = Coordinate(self.space_size, self.space_size)

        self.corners: list = [self._top_l, self._top_r, self._bottom_l, self._bottom_r]

        # define the quadrant borders
        center: int = int(size / 2)
        self.horizontal_border: list[Coordinate] = [Coordinate(x=i, y=center) for i in range(size)]
        self.vertical_border: list[Coordinate] = [Coordinate(x=center, y=i) for i in range(size)]

        # the active border is the one that is used when checking whether the road is close to crash
        self.active_border = self.horizontal_border

        # make general borders
        self.top_border: list[Coordinate] = [Coordinate(x=i, y=0) for i in range(self.space_size)]
        self.left_border: list[Coordinate] = [Coordinate(x=0, y=i) for i in range(self.space_size)]
        self.right_border: list[Coordinate] = [Coordinate(x=self.space_size, y=i) for i in range(self.space_size)]
        self.bottom_border: list[Coordinate] = [Coordinate(x=i, y=self.space_size) for i in range(self.space_size)]
        self.space_border: list[Coordinate] = self.top_border + self.right_border + self.left_border + self.bottom_border

        # instantiate the random generator
        self.random_step: Randomizer = Randomizer()

        self.counter: int = 2

    def create_map(self):
        """
        create a map with n steps
        :return:
        """
        # as a first step make a straight line
        self.update_space()
        # get the distance to the right wall
        max_dist: int = int(math.fabs(self.current_position.x - self.space_size))

        # compute the random number of steps to go towards that direction
        steps: range = range(random.randint(max_dist, max_dist + 5))
        for _step in steps:
            print(self.current_position)
            self.step()

    def step(self):
        """
        Perform one forward step and insert a new piece
        """
        free_direction: bool = True

        # check the current set of actions
        if self.is_border_ahead():
            print("border ahead")
            action: int = 0  # action for turn
            free_direction = False  # since we are forced to take a turn in a specific direction

        else:
            action: int = self.random_step.get_random_action()

        # perform action
        if action == 1:
            # add a straight
            self.add_straight_line()
        else:
            # add a curve
            self.add_curve(free_direction=free_direction)

    def update_space(self):
        """
        Take the update the current position with the current direction and add a 1 to the new position
        :return: none
        """
        self.current_position += self.current_direction
        self.space[self.current_position.y][self.current_position.x] = self.counter
        self.counter += 1

    def switch_active_border(self):
        """
            Switch the current border from vertical to horizontal or vice versa
        """
        self.active_border = self.horizontal_border if self.active_border == self.vertical_border else self.vertical_border

    def is_border_ahead(self) -> bool:
        """
        Check whether the next step in the current direction will crash with the border or not.
        The border is the end of the space that is either 0 or self.size
        :return:
        """
        next_position: Coordinate = self.current_direction + self.current_direction + self.current_position
        if next_position.x == 0 or next_position.y == 0:
            a = 1
        return next_position in self.space_border or next_position in self.active_border

    def is_corner_ahead(self) -> bool:
        """
        Return true if the next piece is a corner
        :return: bool
        """
        next_position: Coordinate = self.current_direction + self.current_position
        return next_position in self.corners

    def add_straight_line(self):
        """
        Add a straight line in the current direction to the space.
        Currently this function only applies a space update but later on it will construct the whole straight lane
        """
        self.update_space()

    def add_curve(self, free_direction: bool):
        """
        Make a turn. Currently only 90 degree turns are supported. This means, we go one step forward and one step
        towards the direction of the curve.
        Since in the racetrack env, a 90 degree curve modifies the x and y coordinate, a curve will be displayed as one
        step forward and the one step in the direction of the curve.
        :return:
        """

        if free_direction:
            # determine_position if we move in x direction we chose upwards or downwards turn
            # if we move in y direction we chose left or right turn
            turn_direction: int = self.random_step.get_random_turn(p=[.5, .5])
            self.update_space()
            if np.abs(self.current_direction.x) == 1:
                # we move in x direction
                self.current_direction.update_position(x=0, y=turn_direction)
        else:
            self.update_space()
            self.current_direction.update_position(x=self.preferred_direction.x, y=self.preferred_direction.y)

        self.update_space()

    def make_turn(self, random_direction: bool = True, random_degree: bool = True, degree: int = 90) -> Direction:
        """
        Make a turn.
        Different modes are possible:
            if random_degree is True, the curve can be either 90 or 180 degree if it is set to false, the degree
            parameter is used instead.

            if random direction is Ture the direction will be randomly chosen. Otherwise the turn will go to the
            side with the most space. So if left side is a wall, it will be a right turn.
        :param degree: either 90 or 180
        :param random_degree: randomize degree between 90 and 180
        :param random_direction: chose turn direction randomly or with respect to the most space available
        :return: new direction
        """

        if random_degree:
            degree = 90 if np.random.randint(0, 1) else 180

        if random_direction:
            pass
        else:
            # check the current direction
            if self.current_direction.x == 1 or self.current_direction.x == -1:
                # we move forward on the x axis
                if self.current_position.x > self.space_size / 2:
                    # we are in the lower half of the space -> turn upwards
                    return Direction(x=0, y=1)
                else:
                    # we are in the upper half of the space -> turn downwards
                    return Direction(x=0, y=-1)
            else:
                # we move on the y axis
                if self.current_position.y > self.space_size / 2:
                    # we are the right side of the space -> turn left
                    return Direction(x=-1, y=0)
                else:
                    # we are on the left side of the space -> turn right
                    return Direction(x=1, y=0)

    def draw_active_border(self):
        """
            draw the current active border
        """
        for coordinate in self.active_border:
            self.space[coordinate.y][coordinate.x] = -1

    def draw_space_border(self):
        """
            Draw the space borders
        """
        for coordinate in self.space_border:
            self.space[coordinate.y][coordinate.x] = -1


if __name__ == "__main__":
    r = RoadGenerator(17)
    r.draw_active_border()
    r.draw_space_border()
    r.create_map()
    print(r.space)