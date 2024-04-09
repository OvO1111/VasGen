import numpy as np


class LateralConnect:
    def __init__(self):
        self.path = None
    
    def connect(self, start, end):
        self.path = []
        
        dx, dy, dz = end - start
        dd_max = np.abs([dx, dy, dz]).max()
        for d in range(dd_max):
            self.path.append([round(start[0] + dx / dd_max * d),
                              round(start[1] + dy / dd_max * d),
                              round(start[2] + dz / dd_max * d)])
        return self.path
        
        