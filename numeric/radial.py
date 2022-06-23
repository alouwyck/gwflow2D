import numpy as np
from ._base import Grid as BaseGrid
from ._base import HorizontalFlowParameters as BaseHorizontalFlowParameters
from ._base import StressPeriod as BaseStressPeriod
from ._base import Model as BaseModel


class Grid(BaseGrid):

    def __init__(self, rb, D, constant=None, inactive=None, connected=None):
        BaseGrid.__init__(self, rb=rb, D=D, constant=constant, inactive=inactive, connected=connected)

    def _set_r_hs_vol(self):
        self.r = np.sqrt(self.rb[:-1] * self.rb[1:])  # (nr, )
        rb2 = self.rb ** 2  # (nr+1, )
        self.hs = np.pi * (rb2[1:] - rb2[:-1])  # (nr, )
        self.vol = np.outer(self.D, self.hs)  # (nl, nr)


class HorizontalFlowParameters(BaseHorizontalFlowParameters):

    def __init__(self, grid, k=None, c=None):
        BaseHorizontalFlowParameters.__init__(self, grid, k, c)

    def _calculate_qc(self):
        # must return qc with shape (nl, nr-1), i.e. without inner and outer boundary
        if self.k is None:
            qc = 2 * np.pi * np.outer(self.D, self.rb[1:-1]) / self.c  # (nl, nr-1)
        else:
            rbc = np.log(self.rb[1:] / self.rb[:-1]) / self.k / 2  # (nl, nr)
            qc = 2 * np.pi * self.D[:, np.newaxis] / (rbc[:, :-1] + rbc[:, 1:])  # (nl, nr-1)
            if self.c is not None:
                irow, icol = (~np.isnan(self.c)).nonzero()
                qc[irow, icol] = 2 * np.pi * self.D[irow] * self.rb[icol + 1] / self.c[irow, icol]
        return qc


class StressPeriod(BaseStressPeriod):

    def __init__(self, grid, timesteps=None, previous=None):
        BaseStressPeriod.__init__(self, grid, timesteps, previous)
        self._horizontal_flow_constructor = HorizontalFlowParameters


class Model(BaseModel):

    def __init__(self):
        BaseModel.__init__(self)
        self._grid_constructor = Grid
        self._period_constructor = StressPeriod
