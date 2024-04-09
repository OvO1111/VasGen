import numpy as np
import functools as fct
from v3.utils import bounding_box

ADJACENT_NODES_3D = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
ADJACENT_NODES_3D.remove((0, 0, 0))


def norm_(d, **kwargs):
    return d.astype(np.float32) / np.linalg.norm(d, **kwargs)


class Node:
    # __slots__ = 'pos', 'parent', 'f', 'g', 'h', 'd', "carry_dir"
    
    def __init__(self, parent, pos, g=0):
        self.pos = np.round(pos).astype(int)
        self.parent = parent
        """if self.parent is not None:
            if self.parent.carry_dir is None:
                self.carry_dir = norm_(self.pos - self.parent.pos) / g
            else:
                self.carry_dir = (self.parent.carry_dir * (g - 1) + norm_(self.pos - self.parent.pos)) / g
        else:
            self.carry_dir = None"""
        
        self.f = 0
        self.g = g
        self.h = 0
        self.d = 0
        
    def __eq__(self, other):
        return all([(self.pos[_] - other.pos[_]) == 0 for _ in range(len(self.pos))])
    
    def __getitem__(self, index):
        return self.pos[index]


class Astar:
    def __init__(self, mask, **cls_kwargs):
        self.path = None
        self.mask = mask
        self.p = cls_kwargs.get("p", 0.8)
        
        self.away = cls_kwargs.get("away", 1)
        self.exit_step = cls_kwargs.get("exit_step", 2000)
        self.preserve_step = cls_kwargs.get("preserve_step", 3)
        
    def connect(self, start, end, max_step, **trace_kwargs):
        self.path = []
        open_list = []
        close_list = []
        
        max_preserve_step = 15
        _is_tracing_done = lambda x, y: x == y
        _away = trace_kwargs.get("away", self.away)
        _exit_step = trace_kwargs.get("exit_step", self.exit_step)
        _preserve_step = trace_kwargs.get("preserve_step", self.preserve_step)
        if trace_kwargs.__contains__("mask"): self.mask = trace_kwargs["mask"]

        start_node = Node(None, start)
        end_node = Node(None, end)
        open_list.append(start_node)
        
        for i in range(_preserve_step, max_preserve_step + 1):
            bbox = bounding_box(self.mask, start, i)
            if (self.mask[*bbox.cropper] == 0).sum() > 0:
                _preserve_step = max(_preserve_step, i)
                break
        else:
            raise RuntimeError(f"[ERROR] A-star trapped in a local area around {start_node.pos} after {max_preserve_step} steps")
        
        for i in range(10):
            bbox = bounding_box(self.mask, end, i)
            if (self.mask[*bbox.cropper] == 0).sum() > 0:
                _away = max(_away, i)
                break
        else:
            _away = max(_away, 10)
        
        while len(open_list) > 0:
            current_node: Node = open_list[0]
            # 1) if inside bifurcation area, then exit and do bifurcation
            # if bifurcation is not None and bifurcation[*current_node.pos]: 
            #     _is_tracing_done = lambda x, y: True
                    
            open_list.pop(0)
            close_list.append(current_node)
            
            # 2) cap search space to limit seach time
            if len(close_list) > _exit_step:
                all_list = open_list + close_list
                best_node = np.array([np.linalg.norm(np.array(p.pos) - np.array(end_node.pos)) for p in all_list])
                backtrack_node = all_list[best_node.argmin()]
                while backtrack_node is not None:
                    self.path.append(backtrack_node.pos)
                    backtrack_node = backtrack_node.parent
                # print(f"A-star unable to converge on time, starting at {start_node.pos}, best try at {all_list[best_node.argmin()].pos}, end is {end_node.pos}")
                break
            
            if current_node.g > _preserve_step and current_node.h < _away ** 2:
                backtrack_node = current_node
                while backtrack_node is not None:
                    self.path.append(backtrack_node.pos)
                    backtrack_node = backtrack_node.parent
                # print(f"[INFO] A-star has found node {current_node.pos}, {np.sqrt(current_node.h):.2f} away from destination {end_node.pos}")
                break
            
            if _is_tracing_done(current_node, end_node):
                backtrack_node = current_node
                while backtrack_node is not None:
                    self.path.append(backtrack_node.pos)
                    backtrack_node = backtrack_node.parent
                break
            
            children = []
            for deltapos in ADJACENT_NODES_3D:
                # 1) check whether in the range of `self.mask`
                current_pos = tuple(current_node.pos[_] + deltapos[_] for _ in range(len(current_node.pos)))
                if fct.reduce(lambda x, y: x | y, [current_pos[_] < 0 for _ in range(len(current_pos))] +\
                    [current_pos[_] >= self.mask.shape[_] for _ in range(len(current_pos))]):
                    continue
                # 2) check whether hit a valid label (organ, auxiliary spheres, etc.)
                if current_node.g > _preserve_step and self.mask[current_pos] > 0:
                    continue
                # 3) if not, append self.end to candidate points
                children.append(Node(current_node, current_pos, current_node.g + 1))
            
            for child in children:
                for closed_child in close_list:
                    if child == closed_child:
                        break
                else:
                    for index, open_node in enumerate(open_list):
                        if child == open_node:
                            if child.g >= open_node.g:
                                break
                            else:
                                open_list.pop(index)
                    else:
                        index = 0
                        child.h = sum([(child.pos[_] - end_node.pos[_]) ** 2 for _ in range(len(current_pos))])
                        
                        child.f = child.g + child.h
                        # insertion sort
                        while index < len(open_list) and open_list[index].f < child.f:
                            index += 1
                        open_list.insert(index, child)
        
        return self.path
