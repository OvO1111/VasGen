import numpy as np

import utils
from typing import Iterable, List
from trace_algorithms.astar import Astar
from trace_algorithms.bspline import Bspline
from trace_algorithms.simple import LateralConnect

from scipy.ndimage import binary_dilation
from numpy.linalg import norm, svd

targeted_region = None
reader = utils.NiftiReader()


class VesselTree(object):
    def __init__(self, base_node, base_dirc, targets, mask, 
                 base_scale=.9, base_beta=.5, base_k=2.55, base_d=10, base_phi=0.7, base_delta=20, base_gamma=-40):
        """
        generate a complete vessel tree; from the primal endpoint to the distal endpoint(s)
        """
        self.vessel: List[Vessel] = []
        
        self.k = base_k
        
        self.base_node = base_node
        self.base_direction = base_dirc
        self.base_artery_targets = targets
        self.totalsegmentor_organ_mask = mask
        self.base_hyper_parameters = dict(d=base_d,
                                          phi=base_phi,
                                          beta=base_beta,
                                          delta=base_delta,
                                          gamma=base_gamma)
        self.global_scale = base_scale
        self.local_scale = base_scale
        
        self._vessel_path = []
        self._percepted_targets = []
        self.meshgrid = np.asarray(
            np.meshgrid(*[np.arange(self.totalsegmentor_organ_mask.shape[i]) for i in range(3)], indexing="ij")
        )
        
    def load_subalgo_kwargs(self, astar_kwargs=None, bspline_kwargs=None):
        self.astar_kwargs = astar_kwargs
        self.bspline_kwargs = bspline_kwargs
        
    def _astar_conn(self, _vessel_vertices, connected=True):
        if connected:
            _masked = self.totalsegmentor_organ_mask[*_vessel_vertices.T] == 0
            _vessel_vertices = _vessel_vertices[_masked]
        self._vessel_path = []
        conn = Astar(self.totalsegmentor_organ_mask)
        indices = np.linspace(0, len(_vessel_vertices)-1, endpoint=True, num=5, dtype=int)
        for i in range(len(indices) - 1):
            self._vessel_path.extend(conn.connect(_vessel_vertices[indices[i]],\
                _vessel_vertices[indices[i+1]])[:-1])
        self._vessel_path = np.array(self._vessel_path)
        
    def _bspine_conn(self):
        pass
    
    def _simple_conn(self, _vessel_vertices):
        self._vessel_path = []
        conn = LateralConnect()
        for i in range(len(_vessel_vertices) - 1):
            self._vessel_path.extend(conn.connect(_vessel_vertices[i], _vessel_vertices[i+1])[:-1])
        self._vessel_path = np.array(self._vessel_path)
        """if len(self._vessel_path) > 1:
            self._astar_conn(self._vessel_path)"""

    def _generate(self, vessel_dic: dict, vessel_r: int | float, vessel_level: int=0):
        global targeted_region, reader
        if vessel_level > 7: return []
        
        self.step_(vessel_dic, self.base_hyper_parameters, level=vessel_level)
        self.vessel.append(dict(path=self._vessel_path,
                                dic=vessel_dic, r=vessel_r, level=vessel_level))
        child_vessels, rs = self.next_(vessel_r)
        levels = [vessel_level + 1] * len(child_vessels)
        
        if len(self._vessel_path) > 0:
            targeted_region[*np.array(self._vessel_path).T] = 100
            targeted_region[binary_dilation(targeted_region == 100)] = vessel_level + 2
            reader.save_to_disk(targeted_region)
        
        return list(zip(child_vessels, rs, levels))
        
    def generate(self):
        raw_vessel_diameter = 4
        primary_vessel = dict(root=self.base_node,
                            dirc=self.base_direction,
                            tgt=self.base_artery_targets)

        vessel_bfs_queue = [(primary_vessel, raw_vessel_diameter, 0)]
        while len(vessel_bfs_queue) > 0:
            vessel_dic, vessel_r, level = vessel_bfs_queue.pop(0)
            vessel_bfs_queue.extend(self._generate(vessel_dic, vessel_r, level))
            
    def next_(self, r0):
        
        if len(self._vessel_path) < 2: return [], []
        _target = self._percepted_targets
        _end_point = np.array(self._vessel_path[-1])
        _end_direction = utils.get_direction(self._vessel_path)
  
        cos_theta1 = lambda x, y: (r0 ** 4 + x ** 4 - y ** 4) / (2 * x ** 2 * r0 ** 2)
        cos_theta2 = lambda x, y: (r0 ** 4 + y ** 4 - x ** 4) / (2 * r0 ** 2 * y ** 2)
        unit_radius_ball_around_endpoint = np.asarray(
            np.meshgrid(*[np.arange
                          (_end_point[i] - 2, _end_point[i] + 3) for i in range(3)], indexing="ij")
        ).reshape(3, -1)
        sub = (unit_radius_ball_around_endpoint - _end_point[:, None]).T
        
        u, s, _ = svd(_target @ _target.T / len(_target))
        target_primal_axis = u[np.diag(s).argmax()]
        t1 = np.dot(self.target.T - _end_point[None], target_primal_axis) > 0
        t2 = np.dot(self.target.T - _end_point[None], target_primal_axis) <= 0
        target_along_primal_axis1 = np.dot(_target.T - _end_point[None], target_primal_axis) > 0
        target_along_primal_axis2 = np.dot(_target.T - _end_point[None], target_primal_axis) <= 0
        q1 = _target[:, target_along_primal_axis1]
        q2 = _target[:, target_along_primal_axis2]
        q1_to_q2_ratio = q1.shape[1] / q2.shape[1] if q2.shape[1] != 0 else np.inf
        q2_to_q1_ratio = q2.shape[1] / q1.shape[1] if q1.shape[1] != 0 else np.inf
        # r0 ** k = r1 ** k + r2 ** k
        # q1 / q2 = r1 ** 2 / r2 ** 2
        r1 = (r0 ** self.k / (1 + q2_to_q1_ratio ** (self.k / 2))) ** (1 / self.k)
        r2 = (r0 ** self.k / (1 + q1_to_q2_ratio ** (self.k / 2))) ** (1 / self.k)
        
        rs = []
        children_vessel_kwargs = []
        if r1 > 1.:
            cos1 = cos_theta1(r1, r2)
            angular_compatible_unit_ball_candidates1 = np.abs(np.dot(sub, _end_direction) / norm(sub, axis=1) - cos1) < 0.1
            orient_along_primal_axis1 = np.dot(sub[angular_compatible_unit_ball_candidates1], target_primal_axis).argmax()
            next_direction = sub[angular_compatible_unit_ball_candidates1][orient_along_primal_axis1].astype(np.float32)
            next_direction /= norm(next_direction)
            children_vessel_kwargs.append(dict(root=_end_point,
                                            dirc=next_direction,
                                            tgt=self.target[:, t1]))
            rs.append(r1)
        if r2 > 1.:
            cos2 = cos_theta2(r1, r2)
            angular_compatible_unit_ball_candidates2 = np.abs(np.dot(sub, _end_direction) / norm(sub, axis=1) - cos2) < 0.1
            orient_along_primal_axis2 = np.dot(sub[angular_compatible_unit_ball_candidates2], -target_primal_axis).argmax()
            next_direction = sub[angular_compatible_unit_ball_candidates2][orient_along_primal_axis2].astype(np.float32)
            next_direction /= norm(next_direction)
            children_vessel_kwargs.append(dict(root=_end_point,
                                            dirc=next_direction,
                                            tgt=self.target[:, t2]))
            rs.append(r2)
        return children_vessel_kwargs, rs
            
    def step_(self, _vessel_setter: dict, _params_setter: dict, **kwargs):
        global reader, targeted_region
        
        _start_point = _vessel_setter.get("root")
        _start_direction = _vessel_setter.get("dirc")
        _target = _vessel_setter.get("tgt")
        
        _global_scale = self.global_scale ** kwargs.get("level", 0)
        
        _raw_d = _d = _params_setter.get("d", 10) * _global_scale
        _phi = _params_setter.get("phi", 10) * (2 - _global_scale)
        _beta = _params_setter.get("beta", 0.3)
        _alpha = _params_setter.get("alpha", 30) * np.pi / 180
        _delta = _params_setter.get("delta", 30) * _global_scale
        _raw_gamma = _gamma = (90 - _params_setter.get("gamma", 10)) * np.pi / 180 * _global_scale
        
        _vessel_vertices = [_start_point]
        target_matrix_glob = np.zeros(self.totalsegmentor_organ_mask.shape, dtype=bool)
        target_matrix_glob[*_target] = True
        
        _end_point = _start_point
        _end_direction = _start_direction
        _percepted_target_along_discrete_path = np.zeros((0, 3), dtype=int)
        _percepted_orientation_to_discrete_path = np.zeros((0,), dtype=int)
        
        while True:
            
            bounding_box = utils.bounding_box(self.totalsegmentor_organ_mask, _end_point, outline=round(2 * _delta))
            _meshgrid = self.meshgrid[:, *bounding_box.cropper]
            _vessel_endpoint = bounding_box.transformer(_end_point)
            _target_matrix_glob = target_matrix_glob[*bounding_box.cropper]
            
            sub1 = (_meshgrid - _end_point[:, None, None, None]) * reader.spacing[:, None, None, None]
            distance_to_current_endpoint = norm(sub1, axis=0)
            search_sphere = (distance_to_current_endpoint > 1) & (distance_to_current_endpoint < _delta)
            dot1 = np.dot(sub1.transpose(1, 2, 3, 0), _end_direction) / distance_to_current_endpoint
            search_cone = dot1 > np.cos(_gamma)
            search_region = search_sphere  # search_cone & search_sphere
            targets_within_horizon = np.argwhere(search_region & _target_matrix_glob)
            
            if len(targets_within_horizon) == 0: break
            
            percept_ring = (np.cos(_alpha + _gamma / 2) < dot1) & (dot1 < np.cos(_alpha - _gamma / 2)) 
            percept_region = percept_ring & search_sphere
            targets_within_perception = np.argwhere(percept_region & _target_matrix_glob)
            targeted_vector_within_horizon = targets_within_horizon - _vessel_endpoint[None]
            targeted_vector_within_perception = targets_within_perception - _vessel_endpoint[None]
            
            orientation_within_horizon = np.dot(targeted_vector_within_horizon, _end_direction) /\
                norm(targeted_vector_within_horizon, axis=1)
            orientation_within_perception = np.dot(targeted_vector_within_perception, _end_direction) /\
                norm(targeted_vector_within_perception, axis=1)
            
            bifurcation_ind = np.std(orientation_within_horizon) 
            is_bifurcation = (len(_vessel_vertices) > 10) or (bifurcation_ind > _phi)
            _percepted_orientation_to_discrete_path = np.concatenate([_percepted_orientation_to_discrete_path,
                                                                      orientation_within_perception])
            _percepted_target_along_discrete_path = np.concatenate([_percepted_target_along_discrete_path,
                                                                    targets_within_perception])

            if is_bifurcation: break
            
            _endpoint_search_space = (distance_to_current_endpoint < _d * 1.5) & (dot1 > 0.5)
            _endpoint_search_w = np.exp(-np.abs(distance_to_current_endpoint + _d) + 10 * dot1)[_endpoint_search_space]
            _endpoint_search_w /= _endpoint_search_w.sum()
            _endpoint_candidates = np.argwhere(_endpoint_search_space)
            
            if len(_endpoint_candidates) == 0: break
            
            _vessel_endpoint_choice = np.random.choice(np.arange(len(_endpoint_candidates)), size=(1,), p=_endpoint_search_w)
            self.end_point = bounding_box.inv_transformer(_endpoint_candidates[_vessel_endpoint_choice][0])
            _end_point = np.round(_end_point + _end_direction * _d).astype(int)
            
            _traversed_direction = _end_point - _vessel_vertices[-1]
            _end_direction = (_beta * _traversed_direction / norm(_traversed_direction) +\
                (1 - _beta) * targeted_vector_within_horizon.mean(0) / norm(targeted_vector_within_horizon.mean(0)))

            _beta = _beta * self.local_scale
            _d = max(_d * self.local_scale, _raw_d / 2)
            _gamma = max(_gamma * self.local_scale, _raw_gamma / 2)
            _vessel_vertices.append(_end_point)
            
            # targeted_region[*bounding_box.cropper][:][search_region] = 255 - len(_vessel_vertices)
            # reader.save_to_disk(targeted_region)

        self._percepted_targets = bounding_box.inv_transformer(np.argwhere(search_sphere & _target_matrix_glob)).T
        target_matrix_glob[*_percepted_target_along_discrete_path.T] = False
        self.target = np.argwhere(target_matrix_glob).T
        self._simple_conn(_vessel_vertices)


class Vessel(object):
    def __init__(self,
                 root: Iterable[int],
                 dirc: np.ndarray,
                 tgt: np.ndarray,
                 mask: np.ndarray,
                 **hyper_kwargs
        ):
        self.start_point = root
        self.start_direction = dirc
        
        self.hyper_kwargs = hyper_kwargs
        self.local_scale = hyper_kwargs.get("local_scale", 0.85)
        self.global_scale = hyper_kwargs.get("global_scale", 1)
        
        self.d = hyper_kwargs.get("d", 10) * self.global_scale
        self.k = hyper_kwargs.get("kappa", 2.55)
        self.phi = hyper_kwargs.get("phi", 10) * self.global_scale
        self.beta = hyper_kwargs.get("beta", 0.5)
        self.delta = hyper_kwargs.get("delta", 2) * self.global_scale
        self.alpha = hyper_kwargs.get("alpha", 30) * np.pi / 180
        self.gamma = (90 - hyper_kwargs.get("gamma", 30)) * np.pi / 180
        
        self.mask = mask
        self.target = tgt  # 3 * N
        self.vessel_path = []
        self.raw_d = self.d
        self.raw_gamma = self.gamma
        
        self.generate()
        
    def __getitem__(self, index):
        return self.vessel_path[index]
    
    def _astar_conn(self):
        self.vessel_path = [tuple(self.start_point)]
        conn = Astar(np.zeros(self.mask.shape))
        for i in range(len(self.vessel_vertices) - 1):
            self.vessel_path.extend(conn.connect(self.vessel_vertices[i], self.vessel_vertices[i+1])[1:])
        
    def _lateral_conn(self):
        conn = LateralConnect(self.start_point, self.end_point)
        self.vessel_path = conn.connect()
        
    def _validify_pt(self, pt, follow_dir=None):
        if self.mask[*pt] == 0: return pt
        search_fn = lambda i: np.round(pt + i * follow_dir).astype(int)
        for i in np.linspace(-2, 20, 22):
            new_pt = search_fn(i)
            if self.mask[*new_pt] == 0: return new_pt
        return
        
    def callback(self, r0):
  
        cos_theta1 = lambda x, y: (r0 ** 4 + x ** 4 - y ** 4) / (2 * x ** 2 * r0 ** 2)
        cos_theta2 = lambda x, y: (r0 ** 4 + y ** 4 - x ** 4) / (2 * r0 ** 2 * y ** 2)
        unit_radius_ball_around_endpoint = np.asarray(
            np.meshgrid(*[np.arange
                          (self.end_point[i] - 2, self.end_point[i] + 3) for i in range(3)], indexing="ij")
        ).reshape(3, -1)
        sub = (unit_radius_ball_around_endpoint - self.end_point[:, None]).T
        
        u, s, _ = svd(self.target @ self.target.T / len(self.target))
        target_primal_axis = u[np.diag(s).argmax()]
        target_along_primal_axis1 = np.dot(self.target.T - self.end_point[None], target_primal_axis) > 0
        target_along_primal_axis2 = np.dot(self.target.T - self.end_point[None], target_primal_axis) <= 0
        q1 = self.target[:, target_along_primal_axis1]
        q2 = self.target[:, target_along_primal_axis2]
        q1_to_q2_ratio = q1.shape[1] / q2.shape[1] if q2.shape[1] != 0 else np.inf
        q2_to_q1_ratio = q2.shape[1] / q1.shape[1] if q1.shape[1] != 0 else np.inf
        # r0 ** k = r1 ** k + r2 ** k
        # q1 / q2 = r1 ** 2 / r2 ** 2
        r1 = (r0 ** self.k / (1 + q2_to_q1_ratio ** (self.k / 2))) ** (1 / self.k)
        r2 = (r0 ** self.k / (1 + q1_to_q2_ratio ** (self.k / 2))) ** (1 / self.k)
        
        rs = []
        children_vessel_kwargs = []
        if r1 > 1:
            cos1 = cos_theta1(r1, r2)
            angular_compatible_unit_ball_candidates1 = np.abs(np.dot(sub, self.end_direction) / norm(sub, axis=1) - cos1) < 0.1
            orient_along_primal_axis1 = np.dot(sub[angular_compatible_unit_ball_candidates1], target_primal_axis).argmax()
            angular_compatible_unit_ball_orient_along_primal_axis1 = sub[angular_compatible_unit_ball_candidates1][orient_along_primal_axis1].astype(np.float32)
            angular_compatible_unit_ball_orient_along_primal_axis1 /= norm(angular_compatible_unit_ball_orient_along_primal_axis1)
            children_vessel_kwargs.append(dict(root=self.end_point,
                                            dirc=angular_compatible_unit_ball_orient_along_primal_axis1,
                                            tgt=q1,
                                            mask=self.mask))
            rs.append(r1)
        if r2 > 1:
            cos2 = cos_theta2(r1, r2)
            angular_compatible_unit_ball_candidates2 = np.abs(np.dot(sub, self.end_direction) / norm(sub, axis=1) - cos2) < 0.1
            orient_along_primal_axis2 = np.dot(sub[angular_compatible_unit_ball_candidates2], -target_primal_axis).argmax()
            angular_compatible_unit_ball_orient_along_primal_axis2 = sub[angular_compatible_unit_ball_candidates2][orient_along_primal_axis2].astype(np.float32)
            angular_compatible_unit_ball_orient_along_primal_axis2 /= norm(angular_compatible_unit_ball_orient_along_primal_axis2)
            children_vessel_kwargs.append(dict(root=self.end_point,
                                            dirc=angular_compatible_unit_ball_orient_along_primal_axis2,
                                            tgt=q2,
                                            mask=self.mask))
            rs.append(r2)
        return children_vessel_kwargs, rs
    
    def generate(self):
        global reader, targeted_region
        self.vessel_vertices = [self.start_point]
        mask_meshgrid = np.asarray(np.meshgrid(*[np.arange(self.mask.shape[i]) for i in range(3)], indexing="ij"))
        target_matrix_glob = np.zeros(self.mask.shape, dtype=bool)
        target_matrix_glob[*self.target] = True
        
        self.end_point = self.start_point
        self.end_direction = self.start_direction
        _percepted_target_along_discrete_path = np.zeros((0, 3), dtype=int)
        _percepted_orientation_to_discrete_path = np.zeros((0,), dtype=int)
        
        while True:
            
            bounding_box = utils.bounding_box(self.mask, self.end_point, outline=round(2 * self.delta))
            __mask_meshgrid = mask_meshgrid[:, *bounding_box.cropper]
            __vessel_endpoint = bounding_box.transformer(self.end_point)
            __target_matrix_glob = target_matrix_glob[*bounding_box.cropper]
            
            sub1 = (__mask_meshgrid - self.end_point[:, None, None, None]) * reader.spacing[:, None, None, None]
            distance_to_current_endpoint = norm(sub1, axis=0)
            search_sphere = distance_to_current_endpoint < self.delta
            dot1 = np.dot(sub1.transpose(1, 2, 3, 0), self.end_direction) / distance_to_current_endpoint
            dot1[np.isnan(dot1)] = 0
            search_cone = dot1 > np.cos(self.gamma)
            search_region = search_cone & search_sphere
            targets_within_horizon = np.argwhere(search_region & __target_matrix_glob)
            
            """_valid_endpoint_search_space = (distance_to_current_endpoint < self.d * 1.6) & (distance_to_current_endpoint > self.d * 0.8) & (dot1 > 0.5)
            _valid_endpoint_search_w = np.exp(-np.abs(distance_to_current_endpoint + self.d) + 10 * dot1)[_valid_endpoint_search_space]
            _valid_endpoint_search_w /= _valid_endpoint_search_w.sum()
            _valid_endpoint_candidates = np.argwhere(_valid_endpoint_search_space)
            if len(_valid_endpoint_candidates) == 0:
                break
            _vessel_endpoint_choice = np.random.choice(np.arange(len(_valid_endpoint_candidates)), size=(1,), p=_valid_endpoint_search_w)
            self.end_point = bounding_box.inv_transformer(_valid_endpoint_candidates[_vessel_endpoint_choice][0])"""
            # self.end_point = bounding_box.inv_transformer(np.mean(targets_within_horizon, axis=0))
            
            if len(targets_within_horizon) == 0: break
            
            percept_ring = (np.cos(self.alpha + self.gamma / 2) < dot1) & (dot1 < np.cos(self.alpha - self.gamma / 2)) 
            percept_region = percept_ring & search_sphere
            targets_within_perception = np.argwhere(percept_region & __target_matrix_glob)
            targeted_vector_within_horizon = targets_within_horizon - __vessel_endpoint[None]
            targeted_vector_within_perception = targets_within_perception - __vessel_endpoint[None]
            
            orientation_within_horizon = np.dot(targeted_vector_within_horizon, self.end_direction)
            orientation_within_perception = np.dot(targeted_vector_within_perception, self.end_direction)
            # orientation_within_horizon = orientation_within_horizon / norm(targeted_vector_within_horizon)
            # orientation_within_perception = orientation_within_perception / norm(targeted_vector_within_perception)
            
            bifurcation_ind = np.std(orientation_within_horizon)
            is_bifurcation = bifurcation_ind > self.phi
            _percepted_orientation_to_discrete_path = np.concatenate([_percepted_orientation_to_discrete_path, orientation_within_perception])
            _percepted_target_along_discrete_path = np.concatenate([_percepted_target_along_discrete_path, targets_within_perception])

            if is_bifurcation: break
            
            self.end_point = np.round(self.end_point + self.end_direction * self.d).astype(int)
            _traversed_direction = self.end_point - self.vessel_vertices[-1]
            self.end_direction = (self.beta * _traversed_direction / norm(_traversed_direction) +\
                (1 - self.beta) * targeted_vector_within_horizon.mean(0) / norm(targeted_vector_within_horizon.mean(0)))

            self.beta = self.beta * self.local_scale
            # self.d = max(self.d * local_scale, self.raw_d / 2)
            self.gamma = max(self.gamma * self.local_scale, self.raw_gamma / 2)
            self.vessel_vertices.append(self.end_point)
            
            targeted_region[*bounding_box.cropper][:][search_region] = 255 - len(self.vessel_vertices)
            reader.save_to_disk(targeted_region)

        self._astar_conn()
        target_matrix_glob[*_percepted_target_along_discrete_path.T] = False
        self.target = np.argwhere(target_matrix_glob).T
            
                
if __name__ == "__main__":
    import SimpleITK as sitk
    from scipy.ndimage import distance_transform_edt
    base_focus_target_num = 100000
    base_non_focus_target_density = 100
    
    reader.load("/nas/dailinrui/dataset/cmu/v2/mask/Dataset000CMU_00001_0000.nii.gz", 
                "/nas/dailinrui/targets.nii.gz")
    mask = reader.array
    
    """colon_label = 107
    tgt = np.argwhere(mask == colon_label)
    masked_shape = np.zeros(mask.shape)
    masked_shape[*tgt[tgt[:, 2] > tgt[:, 2].mean()].T] = 1
    masked_shape[mask != colon_label] = 1
    focused_targets = np.argwhere(np.random.random((mask.shape)) < np.exp(-distance_transform_edt(masked_shape) / 20)).T
    focused_targets = focused_targets[:, (mask[*focused_targets] == 0) | (mask[*focused_targets] == colon_label)]
    # focused_targets = focused_targets[:, np.random.choice(focused_targets.shape[1], base_focus_target_num)]
    random_targets = np.array(np.meshgrid(*[np.linspace(0, mask.shape[i], num=base_non_focus_target_density, endpoint=False, dtype=int) for i in range(3)], indexing="ij")).reshape(3, -1)
    # targets = np.hstack([random_targets, focused_targets])
    targets = focused_targets"""
    
    reader.load("/nas/dailinrui/targets.nii.gz", "/nas/dailinrui/label.nii.gz")
    shape = binary_dilation(reader.array == 1, iterations=4)
    masked_shape = reader.array == 0
    targets = np.argwhere(np.random.random((mask.shape)) < np.exp(-distance_transform_edt(masked_shape) / 7)).T
    targets[mask > 0] = 0

    targeted_region = reader.zeros(dtype=np.uint8)
    targeted_region[*targets] = 1
    targeted_region[(mask >= 105) & (mask <= 110)] = 255
    reader.save_to_disk(targeted_region)
    
    vessel_tree = VesselTree(base_node=np.asarray([272, 272, 296]), base_dirc=np.asarray([0, 0, -1]), targets=targets, mask=mask)
    vessel_tree.generate()