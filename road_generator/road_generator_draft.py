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


class RoadGenerator:
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

        # define the corners of the field
        self._top_l: Coordinate = Coordinate(0, 0)
        self._top_r: Coordinate = Coordinate(0, self.space_size)
        self._bottom_l: Coordinate = Coordinate(self.space_size, 0)
        self._bottom_r: Coordinate = Coordinate(self.space_size, self.space_size)

        self.corners: list = [self._top_l, self._top_r, self._bottom_l, self._bottom_r]

        self.counter: int = 2

    def create_map(self, steps: int):
        """
        create a map with n steps
        :param steps:
        :return:
        """
        # as a first step make a straight line
        self.update_space()
        for _ in range(steps):
            self.step()

    def step(self):
        """
        Perform one forward step and insert a new piece
        """
        if self.is_corner_ahead():
            self.current_direction.update_position()
        elif self.is_border_ahead():
            new_direction: Direction = self.make_turn(random_direction=False)
            self.current_direction.update_position(x=new_direction.x, y=new_direction.y)
        else:
            # everything normal do a forward step
            pass

        self.update_space()

        # generate random action

    def update_space(self):
        """
        Take the update the current position with the current direction and add a 1 to the new position
        :return: none
        """
        self.current_position += self.current_direction
        self.space[self.current_position.y][self.current_position.x] = self.counter
        self.counter += 1

    def is_border_ahead(self) -> bool:
        """
        Check whether the next step in the current direction will crash with the border or not.
        The border is the end of the space that is either 0 or self.size
        :return:
        """
        next_position: Coordinate = self.current_direction + self.current_position
        return next_position.x in [0, self.space_size] or next_position.y in [0, self.space_size]

    def is_corner_ahead(self) -> bool:
        """
        Return true if the next piece is a corner
        :return: bool
        """
        next_position: Coordinate = self.current_direction + self.current_position
        return next_position in self.corners

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


if __name__ == "__main__":
    r = RoadGenerator(10)
    r.create_map(25)
    print(r.space)