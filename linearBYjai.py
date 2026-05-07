from typing import Iterable
from typing_extensions import Generator


class LinearRegression:
    """This uses mostly the same flow, with different internals"""

    def fit(
        self, x: list[float], y: list[float]
    ):  ##y is depedent variable $ x is independent variable
        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        """Use sum instead of the equivalent for loop
        
        sum is implemented in C and is more effient
        """
        add_x = sum(x)
        add_y = sum(y)

        mean_x: float = add_x / len(x)
        mean_y: float = add_y / len(y)
        deviation_x: list[float] = [i - mean_x for i in x]
        deviation_y: list[float] = [i - mean_y for i in y]

        """zip allows you to iterate over 2 iterables (in this case lists) at the same time, meaning you don't
        need to have the extra count variable.

        Here since you don't need product_deviation more than once you can use a generator, this way instead of 
        creating a massive list just to call iterate over it once you create something that you consume as you go.
        In this case it makes a difference memory-wise but not a huge difference speed wise.

        The generator would make a big difference if instead you had something like:

        for dev in product_deviation:
            if dev > 15:
                return
        
        if the second item of product_deviation is 15, the generator would have computed only 2 items while the list would
        have computed len(product_deviation) and only used 2.
        """
        product_deviation: Generator[float] = (
            d_x * d_y for d_x, d_y in zip(deviation_x, deviation_y)
        )
        sum_product_deviation: float = sum(product_deviation)

        """The old solution iterated over deviation_x 2 times instead of once 
        (** is the power operator in python)

        square_devation_x = list(map(lambda x: x*x,deviation_x)) # once here
        sum_square_devation_x = sum(square_devation_x) # once on the result of the previous
        """
        sum_square_devation_x = sum(x**2 for x in deviation_x)
        if sum_square_devation_x == 0:
            raise ValueError(" The deviation is 0, Can't move further")

        self.slope = sum_product_deviation / sum_square_devation_x
        self.intercept = mean_y - (self.slope * mean_x)

    def predict(self, x):
        ##Using formula Y = mx + c    where m is slope and c is intercept

        if isinstance(x, (list, tuple)):
            return [self.slope * i + self.intercept for i in x]

        else:
            return (self.slope * x) + self.intercept


def _mean(items: Iterable[float]) -> float:
    return sum(items) / len(items)


def _deviation(mean: float, items: Iterable[float]) -> Generator[float]:
    return (i - mean for i in items)


def _deviation_alternative_syntax(
    mean: float, items: Iterable[float]
) -> Generator[float]:
    """This is the same as the previous one but with a different syntax, its 100% the same"""
    for i in items:
        yield i - mean


class LinearRegression2:
    def __init__(self, x: Iterable[float], y: Iterable[float]):
        """This is a more object oriented approach, where you calculate the slope and intercept in the constructor,
        this way you can make sure that they are always calculated and you don't have to worry about calling fit before
        predict, since you can't call predict without creating an instance of the class first."""
        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        self.slope, self.intercept = self._calculate_fit(x, y)

    def _calculate_fit(
        self, independant: Iterable[float], dependant: Iterable[float]
    ) -> tuple[float, float]:
        mean_x = _mean(independant)
        deviation_x = list(
            _deviation(mean_x, independant)
        )  # This one needs to be a list since we need more than once

        # Do sum_square first since if it gets 0, all other calculations are useless
        sum_square_devation_x = sum(x**2 for x in deviation_x)
        if sum_square_devation_x == 0:
            raise ValueError(" The deviation is 0, Can't move further")

        mean_y = _mean(dependant)
        deviation_y = _deviation(mean_y, dependant)
        sum_product_deviation = sum(
            d_x * d_y for d_x, d_y in zip(deviation_x, deviation_y)
        )

        slope = sum_product_deviation / sum_square_devation_x
        intercept = mean_y - (slope * mean_x)
        return slope, intercept

    def predict(self, *inputs: float) -> list[float]:
        """Using the calculated slope and intercept and formula:

        Y = slope * x + intercept
        """
        # *inputs colects anything and makes your input/output more consitent

        # Instead of having it in a comment make it the docstring and make sure the
        # formula matches the variable names
        return [self.intercept + (self.slope * i) for i in inputs]
