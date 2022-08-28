import numpy as np
from typing import List, Iterable
from dataclasses import dataclass
from scipy.integrate import odeint


@dataclass
class PopChange:
    delta_t: float
    delta_s: float
    delta_i: float
    delta_r: float
    
    @classmethod
    def combine_pop_changes(cls, pop_changes: Iterable["PopChange"]) -> "PopChange":
        delta_t = 0.0
        delta_s = 0.0
        delta_i = 0.0
        delta_r = 0.0
        for pop_change in pop_changes:
            delta_t += pop_change.delta_t
            delta_s += pop_change.delta_s
            delta_i += pop_change.delta_i
            delta_r += pop_change.delta_r
        return cls(delta_t, delta_s, delta_i, delta_r)


class Population:
    def __init__(self, t: float, S: float, I: float, R: float) -> None:
        self.t = t
        self.S = S
        self.I = I
        self.R = R

    @property
    def N(self) -> float:
        return self.S + self.I + self.R

    @property
    def y(self) -> List[float]:
        return [self.S, self.I, self.R]
    
    def get_new_pop(self, pop_change: PopChange) -> "Population":
        return self.__class__(t=(self.t + pop_change.delta_t),
                              S=(self.S + pop_change.delta_s),
                              I=(self.I + pop_change.delta_i),
                              R=(self.R + pop_change.delta_r))


class Variant:
    def __init__(self) -> None:
        self.t_start: float = 0
        self.Rv: float = 0
        self.beta: float = 0.1
        self.gamma: float = 0.01
        self.mu: float = 0.05
        self.t_steps: int = 10

    def calc_modified_S(self, pop: Population) -> float:
        return pop.S #+ self.mu * pop.R - self.Rv

    def dS(self, pop: Population) -> float:
        return -self.beta * pop.I * self.calc_modified_S(pop) / (pop.N)

    def dI(self, pop: Population) -> float:
        return -1 * (self.dS(pop) + self.dR(pop))

    def dR(self, pop: Population) -> float:
        return self.gamma * pop.I

    def dydt(
        self, t: float, y: np.ndarray
    ) -> List[float]:  # List[Callable[[], float]]:
        pop = Population(t, *y)
        return [fun(pop) for fun in [self.dS, self.dI, self.dR]]
        # return [partial(fun, pop=pop) for fun in [self.dS, self.dI, self.dR]]

    def calc_change(self, pop: Population, t_end: float) -> PopChange:
        t: np.ndarray = np.linspace(pop.t, t_end, self.t_steps)
        y = odeint(self.dydt, pop.y, t, tfirst=True)
        change: np.ndarray = y[-1, :] - y[0, :]
        return PopChange(t_end - pop.t, *change.tolist())


class Pandemic:
    def __init__(self) -> None:
        self.variants: List[Variant] = [Variant()]
        self.populations: List[Population] = [Population(0, 100, 1, 0)]
        self.timestep = 1

    def update(self):
        pop = self.populations[-1]
        pop_change = PopChange.combine_pop_changes([variant.calc_change(pop, pop.t + self.timestep) for variant in self.variants])
        self.populations.append(pop.get_new_pop(pop_change))
    
    @property
    def current_population(self) -> Population:
        return self.populations[-1]
    
    @property
    def current_time(self) -> float:
        return self.current_population.t
    
    def run(self, t_end: float):
        while self.current_time < t_end:
            self.update()
    
    @property
    def t(self) -> np.ndarray:
        return np.array([pop.t for pop in self.populations])

    @property
    def S(self) -> np.ndarray:
        return np.array([pop.S for pop in self.populations])

    @property
    def I(self) -> np.ndarray:
        return np.array([pop.I for pop in self.populations])

    @property
    def R(self) -> np.ndarray:
        return np.array([pop.R for pop in self.populations])