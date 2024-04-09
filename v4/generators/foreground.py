import os, sys
sys.path.append(__file__)
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import enum
import torch
import numpy as np
import v4.utils as utils

from typing import Iterable
from functools import partial
from numpy.linalg import norm
from numpy import array, argwhere, einsum
from numpy.random import choice, randint, normal, uniform
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label

from v4.macros import LABEL_MAPPING
from v4.trace_algorithms.astar import Astar
from v4.trace_algorithms.bspline import Bspline


def rad_by_deg(deg):
    return deg * np.pi / 180


def get_norm(x, axis=None):
    if axis is None:
        return x / norm(x)
    return x / (np.expand_dims(norm(x, axis=axis), axis) + 1e-8)


def get_min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())


def expand_to_radius(p, r, grid_by_mm=lambda x: x):
    return np.mgrid[*[slice(p[_] - grid_by_mm(r, _), p[_] + grid_by_mm(r, _) + 1) for _ in range(3)]]


def numel(x):
    if hasattr(x, "shape"):
        return np.prod(x.shape)
    elif isinstance(x, Iterable):
        return len(x)
    return 1


class AutoRegression(torch.nn.Module):
    def __init__(self, n, r, R, alpha) -> None:
        super(AutoRegression, self).__init__()
        self.R = torch.tensor(R)
        self.Y = torch.nn.Parameter(torch.randn(n, 3, dtype=torch.float32), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.ones(n, 1, dtype=torch.float32), requires_grad=False) 
        self.gamma = torch.nn.Parameter(torch.randn(1, dtype=torch.float32), requires_grad=True)
        self.I = torch.eye(n, n)
        self.r = torch.from_numpy(r).to(torch.float32)
        self.alpha = torch.from_numpy(alpha).to(torch.float32)
                
    def forward(self):
        # Y = self.Y * self.gamma
        X = (self.beta * self.gamma @ self.alpha.T + self.Y @ (torch.eye(3, 3) - self.alpha @ self.alpha.T)).T
        r_sq = self.r ** 2
        r_inv = r_sq.T / torch.linalg.norm(r_sq.flatten()) ** 2
        L = (self.R ** 2 * self.alpha @ r_inv + X @ (self.I - r_sq @ r_inv))
        L_norm = torch.linalg.norm(L, dim=0)
        return L, (L_norm - 1).abs().max()


def solve_children_bifurcation(alpha, R, n):
    itr, lr, loss_min, max_step = 0, 3e-4, 1, 2e5
    current_best = None
    
    model = AutoRegression(R=R, n=n, r=array([TerminalRadius] * n)[:, None], alpha=alpha[:, None])
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    
    while True:
        L, loss = model()
        if loss < 2e-7 or itr > max_step: 
            break
        if torch.isnan(loss).item():
            break
        current_best = current_best if loss >= loss_min else L
        loss_min = min(loss.item(), loss_min)
        optim.zero_grad()
        loss.backward()
        optim.step()
        itr += 1

    L = current_best.cpu().data.numpy()
    IL = np.diag(norm(L, axis=0))
    L = L @ np.linalg.inv(IL)
    print(f"best loss: {loss_min}")
        
    return L


class print_verbose:
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def __call__(self, msg, *args, **kwargs):
        print(msg, *args, **kwargs)

    
NodeTypes = enum.Enum("NodeTypes", ["UNASSIGNED", "ROOT", "INTERMEDIATE", "BIFURCATION", "LEAF"])
LayerTypes = enum.IntEnum("LayerTypes", ["UNASSIGNED", "PRIMARY", "SECONDARY", "TERTIARY", "EXTENDED"])

Perception_distance = 10                            # in grid
Perception_leaf_angle = rad_by_deg(30)
Perception_intermediate_angle = rad_by_deg(75)

March_distance = 5                                  # in mm
Distance_decay = 0.8

TerminalDistance = 3                                # in mm
TerminalRadius = 1                                  # in mm
TerminalIntensity = 80                              # in HU
Murray_k = 2.5
VesselTracingEMA = .5
    
    
class VesselNode:
    # __slots__ = []
    def __init__(self, 
                 parent=None,
                 r=None, intensity=None, 
                 position=None, direction=None,
                 nodetype=NodeTypes.UNASSIGNED, layertype=LayerTypes.UNASSIGNED):
        self.parent: VesselNode
        
        self.child = []
        self.parent = parent
        self.local_radius = r
        self.position = position
        self.intensity = intensity
        self.direction = direction
        self.layer = layertype
        self.nodetype = nodetype
            
    def get_perception_cone(self, outline):
        mc = [slice(outline[_, 0], outline[_, 1]) for _ in range(3)]
        mgrid = get_norm(np.mgrid[*mc], axis=0)
        perception_sphere = np.sqrt((mgrid ** 2).sum(0)) <= Perception_distance ** 2
        if self.nodetype == NodeTypes.LEAF or self.nodetype == NodeTypes.ROOT:
            perception_cone = einsum("n,nijk->ijk", self.direction, mgrid) > np.cos(Perception_leaf_angle)
        elif self.nodetype != NodeTypes.UNASSIGNED:
            perception_cone = np.abs(einsum("n,nijk->ijk", self.direction, mgrid)) < np.cos(Perception_intermediate_angle)
        perceptive_field = perception_cone & perception_sphere
        return perceptive_field
    
    def get_bifurcation(self, n):
        if n == 0:
            return []
        assert self.nodetype == NodeTypes.BIFURCATION
        dirs = solve_children_bifurcation(self.direction, self.local_radius, n)
        return dirs
    
    def broadcast_radius(self, r_in=TerminalRadius):
        self.local_radius = r_in
        if self.parent:
            if self.parent.nodetype == NodeTypes.BIFURCATION:
                r_out = (len(self.parent.child) * r_in ** Murray_k) ** (1 / Murray_k)
            else:
                r_out = r_in * uniform(.9, 1.2)
            self.parent.broadcast_radius(r_out)
            
    def __repr__(self):
        return "  ".join([f"position={self.position}",
                          f"children={len(self.child)}",
                          f"radius={self.local_radius}",
                          f"nodetype={self.nodetype}",
                          f"layertype={self.layer}"])


class Vasculature:
    def __init__(self, name=None, spacing=None):
        self.name = name
        self.root = None
        self.node_group = {}
        self.spacing = spacing
        
    def convert_to_global_map(self, inv_translation):
        for i, node in self.node_group.items():
            node.position = inv_translation(node.position)
        
    def get_vasculature(self, foreground, background):
        _fill_box = partial(utils.bounding_box, array=foreground)
        foreground_of_background = np.zeros_like(background)
        self.root.nodetype = NodeTypes.ROOT
        bifurcations = [] + [self.root]
        
        def _local_fill(pt, r, lyr, hu=0):
            _local_box = _fill_box(pt=pt, outline=r + 1)
            _local_edt = distance_transform_edt(foreground[*_local_box.cropper] > 0, sampling=self.spacing) < r
            foreground[*_local_box.cropper][_local_edt] = lyr
            foreground_of_background[*_local_box.cropper][_local_edt] = hu
            
        while len(bifurcations) > 0:
            parent: VesselNode = bifurcations.pop(0)
            while len(parent.child) == 1:
                parent_pos, child_pos = array(parent.position), array(parent.child[0].position)
                parent_radius, child_radius = parent.local_radius, parent.child[0].local_radius
                parent_intensity, child_intensity = parent.intensity, parent.child[0].intensity
                
                _local_fill(parent_pos, parent_radius, parent.layer, parent_intensity)
                dx, dy, dz = child_pos - parent_pos
                d = np.abs(child_pos - parent_pos).max()
                for dd in range(d):
                    _local_fill([parent_pos[0] + round(dd / d * dx),
                                 parent_pos[1] + round(dd / d * dy),
                                 parent_pos[2] + round(dd / d * dz)],
                                parent_radius + (child_radius - parent_radius) * (dd / d),
                                parent.layer,
                                parent_intensity + (child_intensity - parent_intensity) * (dd / d))
                parent = parent.child[0]
                    
            if len(parent.child) > 1:
                parent_radius, parent_pos, parent_intensity = parent.local_radius, array(parent.position), parent.intensity
                _local_fill(parent_pos, parent_radius, parent.layer, parent_intensity)
                
                for child in parent.child:
                    child: VesselNode
                    child_pos, child_radius, child_intensity = array(child.position), child.local_radius, child.intensity
                    dx, dy, dz = child_pos - parent_pos
                    d = np.abs(child_pos - parent_pos).max()
                    for dd in range(d):
                        _local_fill([parent_pos[0] + round(dd / d * dx),
                                 parent_pos[1] + round(dd / d * dy),
                                 parent_pos[2] + round(dd / d * dz)],
                                parent_radius + (child_radius - parent_radius) * (dd / d),
                                child.layer,
                                parent_intensity + (child_intensity - parent_intensity) * (dd / d))
                bifurcations = bifurcations + parent.child
        
        background[foreground > 0] = foreground_of_background
        return foreground, background
            
    def __setitem__(self, node_position, node: VesselNode):
        node_position = tuple(map(int, node_position))
        # assign position
        node.position = node_position
        # assign for parent
        parent = getattr(node, "parent")
        # assign intensity
        if getattr(node, "intensity") is None:
            pass
        assert not self.node_group.__contains__(node_position)
        self.node_group[node_position] = node
        if parent is not None: parent.child.append(node)
        if node.nodetype == NodeTypes.ROOT: self.root = node
        
    def __getitem__(self, key) -> VesselNode:
        key = tuple(map(int, key))
        return self.node_group[key]
    
    def __iter__(self):
        for item in self.node_group:
            yield item


class ForegroundGenerator:
    def __init__(
        self,
        start,          # start point of this vessel
        end,            # (dummy) end point of this vessel
        mask,           # mask for organs (totalseg)
        bifur,          # bifurcation vector of this vessel tree
        spacing,        # spacing of the original ct
        poi=None,       # places of interest for this vessel
        start_dir=None, # start direction of vessel tree
        
    ):
        self.poi = poi
        self.mask = mask
        self.mask[...] = 0
        self.bifur = bifur
        self.start_dir = start_dir
        self.start_point = array(start, dtype=int)
        self.end_point = array(end, dtype=int)
        
        self.spacing = np.array(spacing)
        self.print = print_verbose(verbose=True)
        self.xyz_avg = lambda x: x / np.mean(self.spacing)
        self.grid_by_mm = lambda x, i: max(1, round(x / spacing[i]))
        
        # get local map and oxygen sinks
        sp_ = np.zeros_like(self.mask, dtype=bool)
        ep_ = np.zeros_like(self.mask, dtype=bool)
        sp_[*self.start_point] = True
        ep_[*self.end_point] = True
        if self.poi is None: self.poi = binary_dilation(ep_, iterations=1)
        self.poi_box = utils.bounding_box((self.poi | sp_), outline=5)
        
        self._poi = self.poi[*self.poi_box.cropper]
        self._mask = self.mask[*self.poi_box.cropper]
        self._start_point = self.poi_box.transformer(self.start_point)
        self._end_point = self.poi_box.transformer(self.end_point)
        self._distance_map = distance_transform_edt(self._poi == 0, sampling=self.spacing)
        
        self._local_map = self.__get_local_map()
        self._oxygen_sink = self.__get_oxygen_sinks()
        self._vasculature = Vasculature(spacing=self.spacing)
        
        self.next_nodes = []
        self.astar = Astar(self._local_map)
        
        self.foreground = np.zeros_like(self.mask)
        self.background_prototype = np.zeros_like(self.mask)
        
        self.trace()
        
    def __get_local_map(self, candidate_threshold=1e3):
        iterations = 0
        local_map = utils.find_largest_connected_components(self._mask == 0, n=1)
        local_erosible = ~self._poi #& (self._distance_map > TerminalDistance)
        labeled_map, past_labeled_map = None, label(local_map)
        # can be optimized using disjoint set
        while True:
            local_map = binary_erosion(local_map, mask=local_erosible)
            labeled_map = label(local_map)
            if labeled_map[0].sum() == past_labeled_map[0].sum() or \
                labeled_map[1] > 1 or (labeled_map[0] & local_erosible).sum() < candidate_threshold: break
            iterations += 1
            past_labeled_map = labeled_map
            
        self.print(f"local map erosion continued for {iterations} iterations")
        local_map = past_labeled_map[0] & local_erosible
        
        track = Astar(local_map).connect(self._start_point, self._end_point, max_step=1000)
        if len(track) == 0: raise RuntimeError(f"no direct path to endpoint")
        if local_map.sum() == 0: raise RuntimeError(f"not enough valid candidates")
        return local_map
    
    def __get_oxygen_sinks(self):
        oxygen_sinks = np.zeros_like(self._poi)
        gridpoints = argwhere(self._local_map) # n3
        
        final_destinations = argwhere(self._poi & ~binary_erosion(self._poi))[:, None]  # m3
        viable_gridpoints = np.repeat(gridpoints[None],
                                      final_destinations.shape[0],
                                      axis=0)                       # mn3
        distances = np.sqrt(((viable_gridpoints - final_destinations) ** 2).sum(-1)).min(0)
        distances = distances ** 3
        dec_distances = np.clip(1 - get_min_max_norm(distances), a_min=.1, a_max=.5)
        p = randint(0, 101, dec_distances.shape) / 100 < dec_distances
        oxygen_sinks[*gridpoints[p].T] = True
        # oxygen_sinks[*gridpoints[randint(0, len(gridpoints), (round(.1 * self._local_map.sum()),))].T] = 1
        
        # simple_visual = oxygen_sinks.astype(np.uint8)
        # simple_visual[self._poi > 0] = 2
        # simple_visual[(self._local_map > 0) & (oxygen_sinks == 0)] = 3
        # simple_visual[*self._start_point] = 4
        # simple_visual[*self._end_point] = 5
        # simple_visual[self._mask > 0] = 255
        # utils.simple_visualize(simple_visual)
        return oxygen_sinks
    
    def __check_bifurcation(self, node: VesselNode):
        if self.bifur is None: return False
        track = self.astar.connect(node.position, self._end_point)
        is_tracable = len(track) < self.bifur["p"] and len(track) > 0
        if not is_tracable: return False
        
        if node.layer == LayerTypes.PRIMARY:
            next_bifurs = randint(2, 4)
        else:
            next_bifurs = randint(2, 5)
        return next_bifurs
    
    def __get_feasible_end(self, bbox, start_node: VesselNode, advance_direction: np.ndarray):
        current_position = start_node.position
        advance_length = self.xyz_avg(March_distance) * Distance_decay ** start_node.layer
        _vacancy_map = self._mask[*bbox.cropper] == 0
        _vacancy_map[*(array(current_position) - array(bbox.bbox)[:, 0])] = True
        vacancy = argwhere(_vacancy_map) - (array(current_position) - array(bbox.bbox)[:, 0])[None]

        length_dev = -get_min_max_norm(np.abs((vacancy ** 2).sum(1) ** (1 / 2) - advance_length))            # (-1, 0) close to 0 -> √
        direction_dev = einsum("nd,d->n", vacancy, advance_direction) / (norm(vacancy, axis=1) + 1e-8)   # (-1, 1) close to 1 -> √
        next_position = vacancy[np.argmax(length_dev + direction_dev)] + array(current_position)
        return next_position

    def __trace(self):
        start_node: VesselNode = self.next_nodes.pop(0)
        perceptive_bbox = utils.bounding_box(self._oxygen_sink,
                                             start_node.position,
                                             outline=(self.grid_by_mm(Perception_distance, 0),
                                                      self.grid_by_mm(Perception_distance, 1),
                                                      self.grid_by_mm(Perception_distance, 2)))
        _get_next_target = partial(self.__get_feasible_end, perceptive_bbox, start_node)
        _get_perception_cone = partial(start_node.get_perception_cone, array(perceptive_bbox.bbox) - array(start_node.position)[:, None])
        _get_direction = lambda percp: get_norm(argwhere(percp).mean(0) + (array(perceptive_bbox.bbox) - array(start_node.position)[:, None])[:, 0])
        
        perception_cone = _get_perception_cone()
        perception_field = self._oxygen_sink[*perceptive_bbox.cropper]
        advance_direction = get_norm(start_node.direction * VesselTracingEMA + _get_direction(perception_cone & (perception_field > 0)) * (1 - VesselTracingEMA))
        self.print(f"start {self._start_point} end {self._end_point} current {start_node.position} propose {np.round(_get_direction(perception_cone & (perception_field > 0)), 2)} dir {start_node.direction}")
        
        if np.isnan(advance_direction).sum() > 0:   # no oxygen sinks in perceptive field
            advance_direction = start_node.direction
            
        if self._distance_map[*start_node.position] < self.xyz_avg(March_distance) * Distance_decay ** start_node.layer:
            start_node.broadcast_radius(TerminalRadius)
            return
        
        if n := self.__check_bifurcation(start_node):
            start_node.nodetype = NodeTypes.BIFURCATION
            children = start_node.get_bifurcation(n).T
            for i, child in enumerate(children):
                child_advance_direction = get_norm(child + advance_direction)
                next_target_position = _get_next_target(child_advance_direction)
                next_target_direction = get_norm(start_node.direction * VesselTracingEMA + (next_target_position - start_node.position) * (1 - VesselTracingEMA))
                self._vasculature[tuple(next_target_position)] = VesselNode(parent=start_node,
                                                                            nodetype=NodeTypes.LEAF,
                                                                            layertype=start_node.layer + 1,
                                                                            direction=next_target_direction)
                
                self.next_nodes.append(self._vasculature[tuple(next_target_position)])
                if i == 0: self._vasculature[tuple(next_target_position)].broadcast_radius()
        else:
            start_node.nodetype = NodeTypes.INTERMEDIATE
            next_target_position = _get_next_target(advance_direction)
            next_target_direction = get_norm(start_node.direction * VesselTracingEMA + (next_target_position - start_node.position) * (1 - VesselTracingEMA))
            self._vasculature[tuple(next_target_position)] = VesselNode(parent=start_node,
                                                                        nodetype=NodeTypes.LEAF,
                                                                        layertype=start_node.layer,
                                                                        direction=next_target_direction)
            self.next_nodes.append(self._vasculature[tuple(next_target_position)])
        
        perception_field[_get_perception_cone()] = 0
        return
    
    def trace(self):
        if self.start_dir is None:
            self.start_dir = self._end_point - self._start_point
        if not isinstance(self.start_dir, np.ndarray):
            self.start_dir = array(self.start_dir)
        self.start_dir = get_norm(self.start_dir).astype(np.float32) 
        
        self._vasculature[self._start_point] = VesselNode(direction=self.start_dir, 
                                                          nodetype=NodeTypes.ROOT, 
                                                          layertype=LayerTypes.PRIMARY)
        self._vasculature[self._start_point].broadcast_radius()
        self.next_nodes.append(self._vasculature[self._start_point])
        while len(self.next_nodes) > 0: self.__trace()
        
        self._vasculature.convert_to_global_map(self.poi_box.inv_transformer)
        self._vasculature.get_vasculature(self.foreground, self.background_prototype)
                
                
if __name__ == "__main__":
    import SimpleITK as sitk
    sample_mask = "/mnt/data/oss_beijing/dailinrui/data/cmu/splitall/mask/Dataset001_00005_0000.nii.gz"
    sample_mask = sitk.ReadImage(sample_mask)
    spacing = sample_mask.GetSpacing()
    sample_mask = sitk.GetArrayFromImage(sample_mask)
    
    generator = ForegroundGenerator(sample_mask, sample_mask == 106, 
                                    argwhere(sample_mask == 7)[1234] + array([5, 2, 5]), 
                                    argwhere(sample_mask == 106)[1234], 
                                    {"p": 30, "n": (2, 5), "b": {"p": 10, "n": (2, 3), "b": None}},
                                    spacing, None)