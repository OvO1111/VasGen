import numpy as np
from typing import List, Dict
from functools import reduce
from scipy.interpolate import splprep, splev


def parse_vessels(vs: Dict) -> List[List]:
    # (1, 2, 3), (3, 4), (3, 5) -> (1, 2, 3, 4), (1, 2, 3, 5)
    paths = reduce(lambda x, y: x + y, [vs[k] for k in vs.keys()])
    levels = reduce(lambda x, y: x + y, [[k] * len(v) for k, v in vs.items()])
    vessels = sorted(list(zip(paths, levels)), key=lambda x: x[1], reverse=True)
    vessels = [[*vessels[i]] for i in range(len(vessels))]

    while vessels[-1][-1] < vessels[0][-1]:
        vessel_parent = vessels[-1][0]
        for i, v in enumerate(vessels[:-1]):
            path = v[0]
            if np.all(path[0][0] == vessel_parent[-1][0]): 
                vessels[i][0] = vessel_parent + path[1:]
            
        vessels.pop(-1)
    return vessels


class Bspline:
    def __init__(self, vessels):
        self.raw_vessels = parse_vessels(vessels)
        self.processed_vessels = []
        
    def connect(self):
        for vessel, level in self.raw_vessels:
            v, r = list(zip(*vessel))
            tck, _ = splprep(np.array(v).T, s=len(v) + np.sqrt(len(v)))
            spline = np.round(splev(np.linspace(0, 1, len(v)), tck)).T.astype(int)
            self.processed_vessels.append((spline, r, level))
            
        return self.processed_vessels