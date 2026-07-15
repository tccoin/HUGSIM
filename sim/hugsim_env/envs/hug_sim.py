import torch
import numpy as np
from copy import deepcopy
import gymnasium
from gymnasium import spaces
from sim.utils.sim_utils import create_cam, rt2pose, pose2rt, load_camera_cfg, dense_cam_poses
from scipy.spatial.transform import Rotation as SCR
from sim.utils.score_calculator import create_rectangle, bg_collision_det
import os
import pickle
import json
from sim.utils.plan import planner, UnifiedMap
from omegaconf import OmegaConf
import math
from gaussian_renderer import GaussianModel
from scene.obj_model import ObjModel
from gaussian_renderer import render
import open3d as o3d

from gs_world.planning_agents.gaia_e2e.camera_geometry import (
    FLU_TO_RDF,
    pose_sv_to_hugsim_cam_to_vehicle,
)
from gs_world.utils.rotation import quaternion_wxyz_to_matrix, rotvec_to_matrix


def _focal2fov(focal, pixels):
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def _validate_sg_native_rig_config(use_gaia_rig, use_dataset_cam_to_vehicle):
    if use_gaia_rig and use_dataset_cam_to_vehicle:
        raise ValueError(
            'rig_mode=gaia cannot use dataset cam-to-vehicle extrinsics'
        )


def _load_gaia_pose_sv(camera_info_dir, camera_name):
    path = os.path.join(str(camera_info_dir), f"{camera_name}.json")
    with open(path, "r") as f:
        data = json.load(f)
    ext = data["calibration"]["extrinsics"]["transform_VS"]
    quaternion = ext["so3"]
    rot = quaternion_wxyz_to_matrix(
        [quaternion["w"], quaternion["x"], quaternion["y"], quaternion["z"]]
    )
    rot = rot @ rotvec_to_matrix([0.0, 0.0, math.pi / 2.0]) @ rotvec_to_matrix(
        [0.0, -math.pi / 2.0, 0.0]
    )
    pose_vs = np.eye(4, dtype=np.float64)
    pose_vs[:3, :3] = rot
    pose_vs[:3, 3] = np.asarray(ext["translation"]["matrix"][0], dtype=np.float64)
    return FLU_TO_RDF @ np.linalg.inv(pose_vs)


def _load_gaia_pinhole_intrinsic(camera_info_dir, camera_name, image_size):
    path = os.path.join(str(camera_info_dir), f"{camera_name}.json")
    with open(path, "r") as f:
        data = json.load(f)
    matrix = data["calibration"]["intrinsics"]["camera_model"]["pinhole_parameters"]["matrix_image_camera"]["matrix"]
    target_w, target_h = int(image_size[0]), int(image_size[1])
    sx = target_w / max(float(data["width"]), 1.0)
    sy = target_h / max(float(data["height"]), 1.0)
    fx = float(matrix[0][0]) * sx
    fy = float(matrix[1][1]) * sy
    cx = float(matrix[2][0]) * sx
    cy = float(matrix[2][1]) * sy
    return {
        'H': target_h,
        'W': target_w,
        'fovx': _focal2fov(fx, target_w),
        'fovy': _focal2fov(fy, target_h),
        'cx': cx,
        'cy': cy,
    }


def fg_collision_det(ego_box, objs):
    ego_x, ego_y, _, ego_w, ego_l, ego_h, ego_yaw = ego_box
    ego_poly = create_rectangle(ego_x, ego_y, ego_w, ego_l, ego_yaw)
    for obs in objs:
        obs_x, obs_y, _, obs_w, obs_l, _, obs_yaw = obs
        obs_poly = create_rectangle(
            obs_x, obs_y, obs_w, obs_l, obs_yaw)
        if ego_poly.intersects(obs_poly):
            return True
    return False

class HUGSimEnv(gymnasium.Env):
    def __init__(self, cfg, output):
        super().__init__()

        plan_list = cfg.scenario.plan_list
        for control_param in plan_list:
            control_param[5] = os.path.join(cfg.base.realcar_path, control_param[5])

        # When cfg.sg.enabled=True we skip the HUGSIM-trained scene entirely:
        # there is no ground_param.pkl / scene.pth / cfg.yaml to load, and the
        # renderer is Street Gaussians instead of HUGSIM's gaussian_renderer.
        # SG-derived equivalents (ground_model, collision points, ply exports)
        # are built below from the SG training cameras + gaussian state.
        sg_cfg = cfg.get('sg', None) if hasattr(cfg, 'get') else None
        self.sg_enabled = bool(sg_cfg is not None and sg_cfg.get('enabled', False))
        self.sg_backend = None
        self._use_exact_ego_pose = bool(sg_cfg.get('use_exact_ego_pose', False)) if self.sg_enabled else False
        self._use_alpasim_controller_absolute_pose = (
            bool(sg_cfg.get('use_controller_absolute_pose', True))
            if self.sg_enabled else True
        )
        self._ground_pose_correction = (
            str(sg_cfg.get('ground_pose_correction', 'none')).lower()
            if self.sg_enabled else 'none'
        )
        self._exact_ego_pose = None

        if not self.sg_enabled:
            # ── Legacy HUGSIM scene path ─────────────────────────────────
            with open(os.path.join(cfg.model_path, 'ground_param.pkl'), 'rb') as f:
                #numpy.ndarray, float, list
                cam_poses, cam_heights, commands = pickle.load(f)
                cam_poses, commands = dense_cam_poses(cam_poses, commands)
                self.ground_model = (cam_poses, cam_heights, commands)
        else:
            # ground_model derived from SG (built after sg_backend init below)
            self.ground_model = None

        if cfg.scenario.load_HD_map:
            unified_map = UnifiedMap(cfg.base.HD_map.path, cfg.base.HD_map.version, cfg.scenario.scene_name)
        else:
            unified_map = None

        self.kinematic = OmegaConf.to_container(cfg.kinematic)
        self.kinematic['min_steer'] = -math.radians(cfg.kinematic.min_steer)
        self.kinematic['max_steer'] = math.radians(cfg.kinematic.max_steer)
        self.kinematic['start_vr']= np.array(cfg.scenario.start_euler) / 180 * np.pi
        self.kinematic['start_vab'] = np.array(cfg.scenario.start_ab)
        self.kinematic['start_velo'] = cfg.scenario.start_velo
        self.kinematic['start_steer'] = cfg.scenario.start_steer

        if not self.sg_enabled:
            self.gaussians = GaussianModel(cfg.model.sh_degree, affine=cfg.affine)
        else:
            self.gaussians = None  # SG-only mode owns its own gaussians

        if self.sg_enabled:
            # Initialise external render backend; build derived ground_model.
            # Lazy import so the legacy hugsim env doesn't pay renderer deps at import time.
            import sys
            sys.path.insert(0, os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', '..', '..', '..', '..')))
            backend_name = str(sg_cfg.get('backend', 'sg')).lower()
            if backend_name == 'alpasim':
                from gs_world.simulation.alpasim_render_backend import AlpaSimRenderBackend
                self.sg_backend = AlpaSimRenderBackend(
                    renderer_address=str(sg_cfg.renderer_address),
                    scene_id=str(sg_cfg.scene_id),
                    usdz_path=str(sg_cfg.usdz_path),
                    alpasim_src=str(sg_cfg.get('alpasim_src', '')),
                    insert_ego_mask=bool(sg_cfg.get('insert_ego_mask', True)),
                    ego_mask_rig_config_id=str(
                        sg_cfg.get('ego_mask_rig_config_id', 'hyperion_8')),
                    render_intrinsics=str(sg_cfg.get('render_intrinsics', 'native')),
                    time_origin_offset_s=float(sg_cfg.get('time_origin_offset_s', 0.0)),
                    physics_address=str(sg_cfg.get('physics_address', '')),
                    height_correction=str(sg_cfg.get('height_correction', 'recorded')),
                    timeout_s=float(sg_cfg.get('timeout_s', 60.0)),
                )
            else:
                from gs_world.simulation.sg_render_backend import SGRenderBackend
                self.sg_backend = SGRenderBackend(
                    sg_root=str(sg_cfg.root),
                    sg_config=str(sg_cfg.config),
                    sg_model_path=str(sg_cfg.model_path),
                    sg_data_path=str(sg_cfg.data_path),
                    include_sky=bool(sg_cfg.get('include_sky', False)),
                )
            if hasattr(self.sg_backend, 'build_ground_model_hugsim'):
                self.ground_model = self.sg_backend.build_ground_model_hugsim()
            else:
                self.ground_model = self._build_ground_model_from_sg()

        """
        plan_list: a, b, height, yaw, v, model_path, controller, params
        Yaw is based on ego car's orientation. 0 means same direction as ego.
        Right is positive and left is negative.
        """
        _scene_path = cfg.model_path if not self.sg_enabled else (
            sg_cfg.data_path if hasattr(sg_cfg, 'data_path') else None)
        self.planner = planner(plan_list, scene_path=_scene_path, unified_map=unified_map, ground=self.ground_model, dt=cfg.kinematic.dt)

        dynamic_gaussians = {}
        if not self.sg_enabled:
            (model_params, iteration) = torch.load(os.path.join(cfg.model_path, "scene.pth"), weights_only=False)
            self.gaussians.restore(model_params, None)

            for plan_id in self.planner.ckpts.keys():
                dynamic_gaussians[plan_id] = ObjModel(cfg.model.sh_degree, feat_mutable=False)
                (model_params, iteration) = torch.load(self.planner.ckpts[plan_id], weights_only=False)
                model_params = list(model_params)
                dynamic_gaussians[plan_id].restore(model_params, None)

            semantic_idx = torch.argmax(self.gaussians.get_full_3D_features, dim=-1, keepdim=True)
            ground_xyz = self.gaussians.get_full_xyz[(semantic_idx == 0)[:, 0]].detach().cpu().numpy()
            scene_xyz = self.gaussians.get_full_xyz[((semantic_idx > 1) & (semantic_idx != 10))[:, 0]].detach().cpu().numpy()
        else:
            # SG mode: collision points are ALL SG gaussian centers above a
            # confidence threshold; ground points come from the ground_model
            # cam poses (driving surface) instead of a semantic gaussian split,
            # because SG doesn't tag points with HUGSIM's semantic classes.
            if hasattr(self.sg_backend, 'extract_pointclouds_hugsim'):
                ground_xyz, scene_xyz = self.sg_backend.extract_pointclouds_hugsim()
            else:
                ground_xyz, scene_xyz = self._extract_pointclouds_from_sg()

        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(ground_xyz.astype(float))
        o3d.io.write_point_cloud(os.path.join(output, 'ground.ply'), ground_pcd)
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_xyz.astype(float))
        o3d.io.write_point_cloud(os.path.join(output, 'scene.ply'), scene_pcd)

        if cfg.scenario.load_HD_map:
            self.planner.update_agent_route()
        
        if self.sg_enabled and bool(cfg.camera.get('sg_native', False)):
            self.cam_params, cam_align, self.cam_rect = self._load_sg_native_camera_cfg(cfg.camera)
        else:
            self.cam_params, cam_align, self.cam_rect = load_camera_cfg(cfg.camera)
        
        self.ego_verts = np.array([[0.5, 0, 0.5], [0.5, 0, -0.5], [0.5, 1.0,  0.5], [0.5, 1.0, -0.5],
                    [-0.5, 0, -0.5], [-0.5, 0, 0.5], [-0.5, 1.0, -0.5], [-0.5, 1.0, 0.5]])
        self.whl = np.array([1.6, 1.5, 3.0])
        self.ego_verts *= self.whl
        self.data_type = cfg.data_type

        self.action_space = spaces.Dict(
            {
                "steer_rate": spaces.Box(self.kinematic['min_steer'], self.kinematic['max_steer'], dtype=float),
                "acc": spaces.Box(self.kinematic['min_acc'], self.kinematic['max_acc'], dtype=float)
            }
        )
        self.observation_space = spaces.Dict(
            {
                'rgb': spaces.Dict({
                    cam_name: spaces.Box(
                        low=0, high=255, 
                        shape=(params['intrinsic']['H'], params['intrinsic']['W'], 3), dtype=np.uint8
                    ) for cam_name, params in self.cam_params.items()
                }),
                'semantic': spaces.Dict({
                    cam_name: spaces.Box(
                        low=0, high=50, 
                        shape=(params['intrinsic']['H'], params['intrinsic']['W']), dtype=np.uint8
                    ) for cam_name, params in self.cam_params.items()
                }),
                'depth': spaces.Dict({
                    cam_name: spaces.Box(
                        low=0, high=1000, 
                        shape=(params['intrinsic']['H'], params['intrinsic']['W']), dtype=np.float32
                    ) for cam_name, params in self.cam_params.items()
                }),
            }
        )
        self.fric = self.kinematic['fric']

        self.start_vr = self.kinematic['start_vr']
        self.start_vab = self.kinematic['start_vab']
        self.start_velo = self.kinematic['start_velo']
        self.vr = deepcopy(self.kinematic['start_vr'])
        self.vab = deepcopy(self.kinematic['start_vab'])
        self.velo = deepcopy(self.kinematic['start_velo'])
        self.steer = deepcopy(self.kinematic['start_steer'])
        self.dt = self.kinematic['dt']

        if not self.sg_enabled:
            bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
            self.render_fn = render
            self.render_kwargs = {
                "pc": self.gaussians,
                "bg_color": torch.tensor(bg_color, dtype=torch.float32, device="cuda"),
                "dynamic_gaussians": dynamic_gaussians,
                "unicycles": {} # dummy input, unicycle planner is used for unicycle models
            }
            gaussians = self.gaussians
            semantic_idx = torch.argmax(gaussians.get_3D_features, dim=-1, keepdim=True)
            opacities = gaussians.get_opacity[:, 0]
            mask = ((semantic_idx > 1) & (semantic_idx != 10))[:, 0] & (opacities > 0.8)
            self.points = gaussians.get_xyz[mask]
        else:
            # SG mode: rendering goes through self.sg_backend.render(). The
            # planner's update path still expects render_kwargs['planning']
            # to be set on reset/step, so keep an empty dict here.
            self.render_fn = None
            self.render_kwargs = {
                "planning": ({}, {}, {}),  # plan_traj output for empty plan_list
                "dynamic_gaussians": {},
                "unicycles": {},
            }
            # Collision points from SG gaussians (SG world frame → HUGSIM world).
            if hasattr(self.sg_backend, 'extract_collision_points_hugsim'):
                collision_points = self.sg_backend.extract_collision_points_hugsim()
                self.points = torch.from_numpy(collision_points.astype(np.float32)).cuda()
            else:
                self.points = self._extract_collision_points_sg()

        self.last_accel = 0
        self.last_steer_rate = 0

        self.timestamp = 0
    
    def ground_height(self, u, v):
        cam_poses, cam_height, _ = self.ground_model
        cam_dist = np.sqrt(
            (cam_poses[:, 0, 3] - u)**2 + (cam_poses[:, 2, 3] - v)**2
        )
        nearest_cam_idx = np.argmin(cam_dist, axis=0)
        nearest_c2w = cam_poses[nearest_cam_idx]

        nearest_w2c = np.linalg.inv(nearest_c2w)
        uhv_local = nearest_w2c[:3, :3] @ np.array([u, 0, v]) + nearest_w2c[:3, 3]
        uhv_local[1] = 0
        uhv_world = nearest_c2w[:3, :3] @ uhv_local + nearest_c2w[:3, 3]
        
        return uhv_world[1]

    @staticmethod
    def _fit_plane(points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3:
            return None
        ctr = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - ctr, full_matrices=False)
        normal = vh[-1]
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            return None
        return ctr, normal / norm

    def _apply_local_ground_pose_correction(self, pose):
        if self._ground_pose_correction != 'local_plane':
            return pose
        if self.ground_model is None or not hasattr(self, 'whl'):
            return pose

        width, _, length = [float(x) for x in self.whl]
        grid = np.linspace(0.0, 1.0, 4)
        xs, zs = np.meshgrid(grid, grid)
        bottom_local = np.column_stack([
            xs.ravel() * width - width * 0.5,
            np.zeros(xs.size, dtype=np.float64),
            zs.ravel() * length - length * 0.5,
            np.ones(xs.size, dtype=np.float64),
        ])
        bottom_world = (np.asarray(pose, dtype=np.float64) @ bottom_local.T).T[:, :3]
        ground_world = bottom_world.copy()
        for i, point in enumerate(bottom_world):
            ground_world[i, 1] = self.ground_height(float(point[0]), float(point[2]))

        bottom_plane = self._fit_plane(bottom_world)
        ground_plane = self._fit_plane(ground_world)
        if bottom_plane is None or ground_plane is None:
            return pose
        bottom_ctr, bottom_normal = bottom_plane
        ground_ctr, ground_normal = ground_plane
        if float(np.dot(bottom_normal, ground_normal)) < 0.0:
            bottom_normal = -bottom_normal

        displacement = ground_ctr - bottom_ctr
        if np.linalg.norm(displacement) > 1.5:
            return pose
        try:
            rot = SCR.align_vectors(
                ground_normal.reshape(1, 3),
                bottom_normal.reshape(1, 3),
            )[0]
            rot_mat = rot.as_matrix()
        except Exception:
            return pose
        if np.abs(rot.as_euler('xyz', degrees=True)).max() > 10.0:
            return pose

        updated = np.asarray(pose, dtype=np.float64).copy()
        t = np.eye(4, dtype=np.float64)
        t[:3, 3] = updated[:3, 3]
        inv_t = np.eye(4, dtype=np.float64)
        inv_t[:3, 3] = -updated[:3, 3]
        local_rot = np.eye(4, dtype=np.float64)
        local_rot[:3, :3] = rot_mat
        trans = np.eye(4, dtype=np.float64)
        trans[:3, 3] = displacement
        corrected = trans @ t @ local_rot @ inv_t @ updated
        corrected[0, 3] = updated[0, 3]
        corrected[2, 3] = updated[2, 3]
        return corrected
    
    @property
    def route_completion(self):
        cam_poses, _, _ = self.ground_model
        cam_dist = np.sqrt(
            (cam_poses[:, 0, 3] - self.vab[0])**2 + (cam_poses[:, 2, 3] - self.vab[1])**2
        )
        nearest_cam_idx = np.argmin(cam_dist, axis=0)
        completion = (nearest_cam_idx + 1) / max(float(cam_poses.shape[0]), 1.0)
        return float(np.clip(completion, 0.0, 1.0)), cam_dist[nearest_cam_idx]
        

    @property
    def vt(self):
        vt = np.zeros(3)
        vt[[0, 2]] = self.vab
        vt[1] = self.ground_height(self.vab[0], self.vab[1])
        return vt
    
    @property
    def ego(self):
        if self._use_exact_ego_pose and self._exact_ego_pose is not None:
            return np.asarray(self._exact_ego_pose, dtype=np.float64)
        return self._apply_local_ground_pose_correction(rt2pose(self.vr, self.vt))
    
    @property
    def ego_state(self):
        return torch.tensor([self.vab[0], self.vab[1], self.vr[1], self.velo])
    
    @property
    def ego_box(self):
        if self._use_exact_ego_pose and self._exact_ego_pose is not None:
            ego_pose = np.asarray(self._exact_ego_pose, dtype=np.float64)
            pos = ego_pose[:3, 3]
            fwd = ego_pose[:3, 2]
            yaw = np.arctan2(float(fwd[0]), float(fwd[2]))
            return [pos[2], -pos[0], -pos[1], self.whl[0], self.whl[2], self.whl[1], -yaw]
        return [self.vt[2], -self.vt[0], -self.vt[1], self.whl[0], self.whl[2], self.whl[1], -self.vr[1]]

    @property
    def objs_list(self):
        obj_boxes = []
        objs = self.render_kwargs['planning'][0]
        for obj_id, obj_b2w in objs.items():
            yaw = SCR.from_matrix(obj_b2w[:3, :3].detach().cpu().numpy()).as_euler('YXZ')[0]
            # X, Y, Z in IMU, w, l, h
            wlh = self.planner.wlhs[obj_id]
            obj_boxes.append([obj_b2w[2, 3].item(), -obj_b2w[0, 3].item(), -obj_b2w[1, 3].item(), wlh[0], wlh[1], wlh[2], -yaw-0.5*np.pi])
        return obj_boxes

    def _get_obs(self):
        if self.sg_backend is not None:
            return self._get_obs_sg()

        rgbs, semantics, depths = {}, {}, {}
        v2front = self.cam_params['CAM_FRONT']["v2c"]
        for cam_name, params in self.cam_params.items():
            intrinsic, v2c = params['intrinsic'], params['v2c']
            c2front = v2front @ np.linalg.inv(v2c) @ self.cam_rect
            c2w = self.ego @ c2front
            viewpoint = create_cam(intrinsic, c2w)
            with torch.no_grad():
                render_pkg = self.render_fn(viewpoint=viewpoint, prev_viewpoint=None, **self.render_kwargs)
            rgb = (torch.permute(render_pkg['render'].clamp(0, 1), (1,2,0)).detach().cpu().numpy() * 255).astype(np.uint8)
            smt = torch.argmax(render_pkg['feats'], dim=0).detach().cpu().numpy().astype(np.uint8)
            depth = render_pkg['depth'][0].detach().cpu().numpy()
            if (self.data_type == 'waymo' or self.data_type == 'kitti360') and 'BACK' in cam_name:
                rgbs[cam_name] = np.zeros_like(rgb)
                semantics[cam_name] = np.zeros_like(smt)
                depths[cam_name] = np.zeros_like(depth)
            else:
                rgbs[cam_name] = rgb
                semantics[cam_name] = smt
                depths[cam_name] = depth

        return {
                'rgb': rgbs,
                'semantic': semantics,
                'depth': depths,
                }

    # ── SG-mode helpers ────────────────────────────────────────────────────
    def _build_ground_model_from_sg(self):
        """Reconstruct (cam_poses, cam_heights, commands) from SG's training cameras.

        - cam_poses: (N,4,4) c2w in HUGSIM world. We take only CAM_FRONT
          training poses, sorted by timestamp, and convert from
          SG world to HUGSIM world via inv(T_hugsim_to_sg).
        - cam_heights: scalar — height of the camera above the road. Estimated
          from cam_pose translation Y (HUGSIM Y points DOWN, so positive Y is
          below ground). We use the median Y of the front-cam poses.
        - commands: per-pose driving command. We don't have the upstream
          HUGSIM command stream; default everything to STRAIGHT (= 1) which
          the LTF/UniAD wrappers also default to.
        """
        sgb = self.sg_backend
        T_h2s = sgb.T_hugsim_to_sg
        T_s2h = np.linalg.inv(T_h2s)

        front_cam_id = sgb.cam_id_for_name('CAM_FRONT')
        front_cams = []
        for cam in sgb.train_cams:
            if cam.meta.get('cam', -1) == front_cam_id:
                front_cams.append(cam)
        if not front_cams:
            front_cams = list(sgb.train_cams)
        front_cams.sort(key=lambda c: c.meta.get('timestamp', c.meta.get('frame_idx', 0)))

        c2w_hugsim_list = []
        cam_y_list = []
        for cam in front_cams:
            R = np.asarray(cam.R, dtype=np.float64)  # SG: c2w rotation
            T = np.asarray(cam.T, dtype=np.float64)  # SG: w2c translation
            c2w_sg = np.eye(4, dtype=np.float64)
            c2w_sg[:3, :3] = R
            c2w_sg[:3, 3] = -R @ T  # camera position in SG world
            c2w_h = T_s2h @ c2w_sg
            c2w_hugsim_list.append(c2w_h)
            cam_y_list.append(c2w_h[1, 3])

        cam_poses = np.stack(c2w_hugsim_list, axis=0).astype(np.float64)  # (N, 4, 4)
        # cam_heights: HUGSIM Y points downward; ground_height() adds cam_height
        # to the camera-derived ground point, so this is the (positive) drop
        # from camera centre to road. We assume frame-0's camera Y == camera
        # height; with HUGSIM frame-0 cam = identity, the relative drop is 0,
        # so use a constant typical sensor height (~1.5 m for waymo).
        cam_heights = 1.5
        commands = [1] * cam_poses.shape[0]  # STRAIGHT for every keyframe
        return cam_poses, cam_heights, commands

    def _extract_pointclouds_from_sg(self):
        """Return (ground_xyz, scene_xyz) in HUGSIM world derived from SG.

        SG doesn't tag points with HUGSIM's semantic classes, so we use a
        rough heuristic: high-opacity gaussians within a Y-window centred on
        the camera path are scene-structure; everything below is treated as
        ground. Used only for ground.ply / scene.ply (eval inputs).
        """
        sgb = self.sg_backend
        T_h2s = sgb.T_hugsim_to_sg
        T_s2h = np.linalg.inv(T_h2s)

        # Use the background gaussian sub-model directly — the
        # top-level get_xyz / get_opacity properties only resolve after a
        # camera is set, which we haven't done at env-init time.
        bkgd = sgb.gaussians.background
        with torch.no_grad():
            xyz_sg = bkgd.get_xyz.detach().cpu().numpy().astype(np.float64)
            opa = bkgd.get_opacity.detach().cpu().numpy().reshape(-1)
        # Threshold matches HUGSIM's own collision-point opacity filter
        # (>0.8). 0.3 was too permissive — lots of low-opacity floaters
        # around parked cars / curbs registered as "solid" and fired
        # bg_collision when the ego just passed near them.
        mask = opa > 0.8
        xyz_sg = xyz_sg[mask]
        # SG → HUGSIM world
        xyz_h = (T_s2h[:3, :3] @ xyz_sg.T + T_s2h[:3, 3:]).T

        # Heuristic ground / scene split on HUGSIM Y (positive = below cam)
        y_med = float(np.median(xyz_h[:, 1]))
        ground = xyz_h[xyz_h[:, 1] > y_med + 0.5]
        scene  = xyz_h[xyz_h[:, 1] < y_med + 0.5]
        return ground, scene

    def _extract_collision_points_sg(self):
        """Return a torch tensor of high-confidence collision points (HUGSIM
        world). Used by bg_collision_det in step()."""
        _, scene_xyz = self._extract_pointclouds_from_sg()
        return torch.from_numpy(scene_xyz.astype(np.float32)).cuda()

    def _sg_cam_c2w_hugsim(self, cam):
        sgb = self.sg_backend
        T_sg_to_hugsim = np.linalg.inv(sgb.T_hugsim_to_sg)
        c2w_sg = cam.get_extrinsic()
        return T_sg_to_hugsim @ c2w_sg

    def _find_sg_train_cam(self, frame_idx, cam_id):
        sgb = self.sg_backend
        best, best_d = None, 1e9
        for cam in sgb.train_cams:
            if cam.meta.get('cam', -1) != cam_id:
                continue
            d = abs(int(cam.meta.get('frame_idx', -1)) - int(frame_idx))
            if d < best_d:
                best, best_d = cam, d
        return best

    def _resolve_sg_image_path(self, image_path):
        if os.path.isabs(str(image_path)):
            return str(image_path)
        sgb = self.sg_backend
        source_path = sgb.metadata.get('source_path', '') if sgb is not None else ''
        if isinstance(source_path, (list, tuple, np.ndarray)):
            source_path = source_path[0] if len(source_path) else ''
        source_path = str(source_path).strip()
        if source_path.startswith('[') and source_path.endswith(']'):
            source_path = source_path.strip("[]'\" ")
        model_path = str(getattr(self.sg_backend._cfg, 'model_path', ''))
        scene_token = os.path.basename(model_path).replace('eval_waymo_training_', '').split('_')[0]
        processed_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '..', '..',
            'data', 'processed', 'street_gaussians', 'waymo', 'training',
            scene_token,
        ))
        candidates = [
            os.path.join(str(source_path), str(image_path)) if source_path else '',
            os.path.join(processed_root, str(image_path)) if scene_token else '',
            os.path.join(model_path, str(image_path)),
            str(image_path),
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
            root, ext = os.path.splitext(candidate)
            for alt_ext in ('.png', '.jpg', '.jpeg'):
                alt = root + alt_ext
                if ext.lower() != alt_ext and os.path.exists(alt):
                    return alt
        return next((c for c in candidates if c), str(image_path))

    def _read_sg_log_image(self, cam, intrinsic):
        import cv2
        path = self._resolve_sg_image_path(cam.image_path)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read GT image for open-loop mode: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        K = cam.K.detach().cpu().numpy() if hasattr(cam.K, 'detach') else np.asarray(cam.K)
        src_h, src_w = rgb.shape[:2]
        crop_w = min(src_w, int(round(2.0 * float(K[0, 0]) * math.tan(float(intrinsic['fovx']) / 2.0))))
        crop_h = min(src_h, int(round(2.0 * float(K[1, 1]) * math.tan(float(intrinsic['fovy']) / 2.0))))
        if crop_w > 0 and crop_h > 0 and (crop_w < src_w or crop_h < src_h):
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            x0 = max(0, min(src_w - crop_w, int(round(cx - crop_w / 2.0))))
            y0 = max(0, min(src_h - crop_h, int(round(cy - crop_h / 2.0))))
            rgb = rgb[y0:y0 + crop_h, x0:x0 + crop_w]
        H, W = int(intrinsic['H']), int(intrinsic['W'])
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        return rgb.astype(np.uint8)

    def _sg_log_obs_at_time(self, t_sec, fps=10.0):
        """Observation built from original logged images at the nearest frame."""
        assert self.sg_backend is not None, "GT log observations require SG mode"
        frame_idx = int(round(float(t_sec) * float(fps)))
        rgbs, semantics, depths = {}, {}, {}
        for cam_name, params in self.cam_params.items():
            intrinsic = params['intrinsic']
            H, W = int(intrinsic['H']), int(intrinsic['W'])
            rgb = None
            if params.get('sg_render', True):
                sg_cam_id = self.sg_backend.cam_id_for_name(cam_name)
                if sg_cam_id is not None:
                    cam = self._find_sg_train_cam(frame_idx, sg_cam_id)
                    if cam is not None:
                        rgb = self._read_sg_log_image(cam, intrinsic)
            if rgb is None:
                rgb = np.zeros((H, W, 3), dtype=np.uint8)
            rgbs[cam_name] = rgb
            semantics[cam_name] = np.zeros((H, W), dtype=np.uint8)
            depths[cam_name] = np.zeros((H, W), dtype=np.float32)
        return {'rgb': rgbs, 'semantic': semantics, 'depth': depths}

    def _set_sg_state_at_time(self, t_sec, fps=10.0):
        pose_at = getattr(self.sg_backend, 'ego_pose_at_time_hugsim', None)
        if self._use_exact_ego_pose and callable(pose_at):
            ego_pose = np.asarray(pose_at(float(t_sec)), dtype=np.float64)
            self._exact_ego_pose = ego_pose.copy()
            pos = ego_pose[:3, 3]
            fwd = ego_pose[:3, 2]
            vab = np.array([pos[0], pos[2]], dtype=np.float64)
            vr = np.array(
                [0.0, math.atan2(float(fwd[0]), float(fwd[2])), 0.0],
                dtype=np.float64,
            )
        else:
            self._exact_ego_pose = None
            vab, vr = self.sg_start_pose_at_time(t_sec, fps=fps)
        self.vab = np.array(vab, dtype=np.float64)
        self.vr = np.array(vr, dtype=np.float64)
        self.velo = float(self.sg_log_speed_at_time(t_sec, fps=fps))
        self.timestamp = float(t_sec)
        if self.planner is not None:
            self.render_kwargs['planning'] = self.planner.plan_traj(self.timestamp, self.ego_state)

    def sg_log_time_bounds(self, fps=10.0):
        assert self.sg_backend is not None, "sg_log_time_bounds requires SG mode"
        front_cam_id = self.sg_backend.cam_id_for_name('CAM_FRONT')
        frames = [
            int(cam.meta.get('frame_idx', 0))
            for cam in self.sg_backend.train_cams
            if cam.meta.get('cam', -1) == front_cam_id
        ]
        if not frames:
            return 0.0, 0.0
        return min(frames) / float(fps), max(frames) / float(fps)

    def open_loop_observation_at_time(self, t_sec, fps=10.0):
        """Set ego to logged state and return original-image observation/info."""
        self._set_sg_state_at_time(t_sec, fps=fps)
        observation = self._sg_log_obs_at_time(t_sec, fps=fps)
        info = self._get_info()
        rc, dist = self.route_completion
        info['rc'] = rc
        info['collision'] = False
        info['bg_collision'] = False
        info['fg_collision'] = False
        off_route_threshold = float(getattr(self, 'off_route_threshold', 30.0))
        info['off_route'] = bool(dist > off_route_threshold)
        info['route_complete'] = False
        info['open_loop'] = True
        return observation, info

    def _intrinsic_from_sg_cam(self, cam, image_size):
        target_w, target_h = int(image_size[0]), int(image_size[1])
        native_w, native_h = int(cam.image_width), int(cam.image_height)
        K = cam.K.detach().cpu().numpy() if hasattr(cam.K, 'detach') else np.asarray(cam.K)
        fx = float(K[0, 0]) * target_w / native_w
        fy = float(K[1, 1]) * target_h / native_h
        cx = float(K[0, 2]) * target_w / native_w
        cy = float(K[1, 2]) * target_h / native_h
        return {
            'H': target_h,
            'W': target_w,
            'fovx': _focal2fov(fx, target_w),
            'fovy': _focal2fov(fy, target_h),
            'cx': cx,
            'cy': cy,
        }

    def _front_tele_intrinsic(self, image_size, hfov_degrees=30.0):
        target_w, target_h = int(image_size[0]), int(image_size[1])
        hfov = math.radians(float(hfov_degrees))
        focal = target_w / (2.0 * math.tan(hfov / 2.0))
        return {
            'H': target_h,
            'W': target_w,
            'fovx': hfov,
            'fovy': _focal2fov(focal, target_h),
            'cx': target_w / 2.0,
            'cy': target_h / 2.0,
        }

    def _requested_sg_native_camera_names(self, camera_cfg):
        camera_names = list(camera_cfg.get('camera_names', []))
        if not camera_names:
            camera_names = list(camera_cfg.get('cameras', []))
        if not camera_names:
            for row in camera_cfg.get('cam_align', []):
                camera_names.extend(list(row))
        if not camera_names:
            camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

        unique = []
        for cam_name in camera_names:
            cam_name = str(cam_name)
            if cam_name not in unique:
                unique.append(cam_name)
        return unique

    def _sg_native_cam_align(self, camera_cfg, camera_names, cam_params):
        rows = camera_cfg.get('cam_align', [])
        cam_align = [
            [str(cam_name) for cam_name in row if str(cam_name) in cam_params]
            for row in rows
        ]
        cam_align = [row for row in cam_align if row]
        if not cam_align:
            cam_align = [[cam_name for cam_name in camera_names if cam_name in cam_params]]
        return cam_align

    def _load_sg_native_camera_cfg(self, camera_cfg):
        """Build camera params from the active SG scene's frame-0 training rig.

        HUGSim SG mode uses CAM_FRONT as the reference frame. Therefore v2c is
        chosen so ``v2front @ inv(v2c)`` equals each SG camera's frame-0
        cam-to-front transform. This makes the first rendered front camera
        identical to the SG training front camera; side cameras follow the SG
        training extrinsics.
        """
        image_size = camera_cfg.get('image_size', [1152, 672])
        frame_idx = int(camera_cfg.get('frame_idx', 0))
        camera_names = self._requested_sg_native_camera_names(camera_cfg)
        if 'CAM_FRONT' not in camera_names:
            raise ValueError('SG native camera config requires CAM_FRONT as the reference camera')
        front_sg_id = self.sg_backend.cam_id_for_name('CAM_FRONT')
        front_cam = self._find_sg_train_cam(frame_idx, front_sg_id)
        assert front_cam is not None, 'SG native camera config requires CAM_FRONT'
        front_c2w = self._sg_cam_c2w_hugsim(front_cam)
        agent_sg_native = camera_cfg.get('agent_sg_native', {})
        use_dataset_cam_to_vehicle = bool(agent_sg_native.get('use_dataset_cam_to_vehicle', False))
        legacy_render_gaia_rig = bool(agent_sg_native.get('render_gaia_rig', False))
        rig_mode = str(agent_sg_native.get('rig_mode', '')).strip().lower()
        if not rig_mode:
            rig_mode = 'gaia' if legacy_render_gaia_rig else 'hugsim'
        if rig_mode not in ('hugsim', 'gaia'):
            raise ValueError("agent_sg_native.rig_mode must be 'hugsim' or 'gaia'")
        use_gaia_rig = rig_mode == 'gaia'
        _validate_sg_native_rig_config(use_gaia_rig, use_dataset_cam_to_vehicle)
        intrinsics_mode = str(agent_sg_native.get('intrinsics_mode', '')).strip().lower()
        if intrinsics_mode not in ('', 'hugsim', 'gaia'):
            raise ValueError("agent_sg_native.intrinsics_mode must be 'hugsim' or 'gaia'")
        render_gaia_pinhole_intrinsics = bool(agent_sg_native.get('render_gaia_pinhole_intrinsics', False))
        gaia_pinhole_intrinsic_cameras = agent_sg_native.get('gaia_pinhole_intrinsic_cameras', None)
        if gaia_pinhole_intrinsic_cameras is not None:
            gaia_pinhole_intrinsic_cameras = {str(cam_name) for cam_name in gaia_pinhole_intrinsic_cameras}
        gaia_camera_info_dir = agent_sg_native.get('gaia_camera_info_dir', None)
        gaia_camera_map = agent_sg_native.get('gaia_camera_map', {})
        if intrinsics_mode == 'gaia':
            uses_gaia_intrinsics = True
            gaia_pinhole_intrinsic_cameras = None
        elif intrinsics_mode == 'hugsim':
            uses_gaia_intrinsics = False
        else:
            uses_gaia_intrinsics = render_gaia_pinhole_intrinsics
        if use_gaia_rig or uses_gaia_intrinsics:
            if gaia_camera_info_dir is None:
                raise ValueError(
                    'rig_mode=gaia/intrinsics_mode=gaia require '
                    'agent_sg_native.gaia_camera_info_dir'
                )
            gaia_camera_map = {
                'CAM_FRONT': 'front_wide',
                'CAM_FRONT_TELE': 'front_tele',
                'CAM_FRONT_LEFT': 'left_side_forward',
                'CAM_FRONT_RIGHT': 'right_side_forward',
                **dict(gaia_camera_map),
            }
            if use_gaia_rig:
                gaia_pose_sv = {
                    gaia_name: _load_gaia_pose_sv(gaia_camera_info_dir, gaia_name)
                    for gaia_name in set(gaia_camera_map.values())
                }
                front_pose_sv = gaia_pose_sv[gaia_camera_map['CAM_FRONT']]
            else:
                gaia_pose_sv = {}
                front_pose_sv = None
        else:
            gaia_camera_map = {}
            gaia_pose_sv = {}
            front_pose_sv = None

        cam_params = {}
        for cam_name in camera_names:
            if cam_name == 'CAM_BACK':
                continue
            cam = None
            sg_id = self.sg_backend.cam_id_for_name(cam_name)
            synthesize_front_tele = (
                cam_name == 'CAM_FRONT_TELE'
                and (sg_id is None or sg_id == front_sg_id)
            )
            if synthesize_front_tele:
                cam = front_cam
                cam_c2w = front_c2w
                intrinsic = self._front_tele_intrinsic(image_size)
            else:
                if sg_id is None:
                    available = ', '.join(self.sg_backend.available_camera_names())
                    raise ValueError(
                        f'SG native camera config requested unsupported camera {cam_name}. '
                        f'Available SG camera names: {available}'
                    )
                cam = self._find_sg_train_cam(frame_idx, sg_id)
                if cam is None:
                    available = ', '.join(self.sg_backend.available_camera_names())
                    raise ValueError(
                        f'SG native camera config requested {cam_name}, but the SG scene '
                        f'has no matching training camera. Available SG camera names: {available}'
                    )
                cam_c2w = self._sg_cam_c2w_hugsim(cam)
                intrinsic = self._intrinsic_from_sg_cam(cam, image_size)
            gaia_name = gaia_camera_map.get(cam_name)
            use_gaia_pinhole_intrinsic = (
                uses_gaia_intrinsics
                and gaia_name is not None
                and (
                    gaia_pinhole_intrinsic_cameras is None
                    or cam_name in gaia_pinhole_intrinsic_cameras
                )
            )
            if use_gaia_pinhole_intrinsic:
                intrinsic = _load_gaia_pinhole_intrinsic(gaia_camera_info_dir, gaia_name, image_size)
            if use_gaia_rig and gaia_name is not None:
                cam_to_front = front_pose_sv @ np.linalg.inv(gaia_pose_sv[gaia_name])
            else:
                cam_to_front = np.linalg.inv(front_c2w) @ cam_c2w
            v2c = np.linalg.inv(cam_to_front)
            cam_params[cam_name] = {
                'intrinsic': intrinsic,
                'v2c': v2c,
                'l2c': v2c.copy(),
                'sg_render': True,
            }
            if use_gaia_rig and gaia_name is not None:
                pose_sv = gaia_pose_sv[gaia_name].copy()
                cam_params[cam_name]['pose_SV'] = pose_sv
                cam_params[cam_name]['vehicle_to_camera'] = (
                    pose_sv_to_hugsim_cam_to_vehicle(pose_sv)
                )
                cam_params[cam_name]['agent_cam_to_vehicle'] = np.linalg.inv(pose_sv)
            if use_gaia_pinhole_intrinsic:
                cam_params[cam_name]['camera_model'] = 'PINHOLE'
                cam_params[cam_name]['distortion'] = np.zeros(0, dtype=np.float32)
            is_alpasim_scene = getattr(self.sg_backend, 'dataset_type', '') == 'alpasim'
            if use_dataset_cam_to_vehicle and cam is not None:
                render_vehicle_to_camera = np.asarray(cam.extrinsic, dtype=np.float64).copy()
                if is_alpasim_scene:
                    cam_params[cam_name]['vehicle_to_camera'] = render_vehicle_to_camera.copy()
                agent_cam_to_vehicle = render_vehicle_to_camera.copy()
                if is_alpasim_scene:
                    hugsim_to_navsim = np.eye(4, dtype=np.float64)
                    hugsim_to_navsim[:3, :3] = np.array([
                        [0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                    ], dtype=np.float64)
                    agent_cam_to_vehicle = hugsim_to_navsim @ agent_cam_to_vehicle
                cam_params[cam_name]['agent_cam_to_vehicle'] = agent_cam_to_vehicle

        # Waymo perception SG scenes have no rear training camera. Keep a
        # black rear slot with a valid placeholder K/extrinsic so drivoR's
        # fixed four-camera input contract stays satisfied.
        if 'CAM_BACK' in camera_names and bool(camera_cfg.get('include_black_back', True)):
            back_v2c = np.eye(4, dtype=np.float64)
            back_v2c[:3, :3] = SCR.from_euler('Y', 180.0, degrees=True).as_matrix()
            cam_params['CAM_BACK'] = {
                'intrinsic': dict(cam_params['CAM_FRONT']['intrinsic']),
                'v2c': back_v2c,
                'l2c': back_v2c.copy(),
                'sg_render': False,
            }

        cam_align = self._sg_native_cam_align(camera_cfg, camera_names, cam_params)
        return cam_params, cam_align, np.eye(4, dtype=np.float64)

    def sg_start_pose_at_time(self, t_sec, fps=10.0):
        """Look up the SG front-cam at time t (Waymo fps=10Hz by default) and
        return HUGSIM ego state (vab, vr) that places the ego where the log
        vehicle was at that moment.

        Returns (vab[2], vr[3]) with HUGSIM conventions:
            vab[0] = world +x (right), vab[1] = world +z (forward)
            vr[1]  = yaw, CW-positive from +z toward +x
        """
        assert self.sg_backend is not None, "sg_start_pose_at_time requires SG mode"
        pose_at = getattr(self.sg_backend, 'ego_pose_at_time_hugsim', None)
        sgb = self.sg_backend
        is_alpasim_scene = getattr(sgb, 'dataset_type', '') == 'alpasim'
        if (self._use_exact_ego_pose or is_alpasim_scene) and callable(pose_at):
            ego_pose = np.asarray(pose_at(float(t_sec)), dtype=np.float64)
            pos = ego_pose[:3, 3]
            fwd = ego_pose[:3, 2]
            return (
                np.array([pos[0], pos[2]], dtype=np.float64),
                np.array([0.0, math.atan2(float(fwd[0]), float(fwd[2])), 0.0],
                         dtype=np.float64),
            )
        if is_alpasim_scene:
            raise RuntimeError(
                'AlpaSim scene requires ego_pose_at_time_hugsim to derive '
                'the vehicle pose from the recorded rig trajectory'
            )
        front_cam_id = sgb.cam_id_for_name('CAM_FRONT')
        frame_idx = int(round(float(t_sec) * float(fps)))
        front_cam = None
        for cam in sgb.train_cams:
            if cam.meta.get('frame_idx', -1) == frame_idx and cam.meta.get('cam', -1) == front_cam_id:
                front_cam = cam
                break
        if front_cam is None:
            # fall back to nearest available front-cam frame
            best, best_d = None, 1e9
            for cam in sgb.train_cams:
                if cam.meta.get('cam', -1) != front_cam_id:
                    continue
                d = abs(int(cam.meta.get('frame_idx', -1)) - frame_idx)
                if d < best_d:
                    best, best_d = cam, d
            assert best is not None, "No SG CAM_FRONT frames available"
            print(f"[hug_sim] sg_start_pose_at_time({t_sec}s, frame={frame_idx}): "
                  f"no exact match, using nearest frame_idx={best.meta['frame_idx']}")
            front_cam = best

        c2w_sg = front_cam.get_extrinsic()  # (4,4) SG world
        T_sg_to_hugsim = np.linalg.inv(sgb.T_hugsim_to_sg)
        c2w_hugsim = T_sg_to_hugsim @ c2w_sg

        # Ego origin = front cam projected onto the ground. We only use (x, z)
        # for vab, and the ground height is recomputed by the env from the
        # ground_model — so the camera's y offset doesn't matter here.
        pos = c2w_hugsim[:3, 3]
        vab = np.array([pos[0], pos[2]], dtype=np.float64)

        # Yaw: angle of cam +Z axis (camera forward) in the HUGSIM x-z plane,
        # measured CW from +Z toward +X. matches the convention in
        # kinematic_step / step (vab updates use sin(θ) for +x, cos(θ) for +z).
        fwd = c2w_hugsim[:3, 2]
        yaw = math.atan2(float(fwd[0]), float(fwd[2]))
        vr = np.array([0.0, yaw, 0.0], dtype=np.float64)
        return vab, vr

    def sg_log_speed_at_time(self, t_sec, fps=10.0):
        """Estimate logged ego speed at ``t_sec`` from adjacent SG front poses."""
        assert self.sg_backend is not None, "sg_log_speed_at_time requires SG mode"
        pose_at = getattr(self.sg_backend, 'ego_pose_at_time_hugsim', None)
        if (self._use_exact_ego_pose or getattr(self.sg_backend, 'dataset_type', '') == 'alpasim') and callable(pose_at):
            t = float(t_sec)
            dt = 1.0 / float(fps)
            t0 = max(0.0, t - dt)
            t1 = t + dt
            p0 = np.asarray(pose_at(t0), dtype=np.float64)[:3, 3]
            p1 = np.asarray(pose_at(t1), dtype=np.float64)[:3, 3]
            return float(np.linalg.norm(p1[[0, 2]] - p0[[0, 2]]) / max(t1 - t0, 1e-6))
        target_idx = int(round(float(t_sec) * float(fps)))
        sgb = self.sg_backend
        front_cam_id = sgb.cam_id_for_name('CAM_FRONT')
        front = [cam for cam in sgb.train_cams if cam.meta.get('cam', -1) == front_cam_id]
        assert front, "No SG CAM_FRONT frames available"
        front.sort(key=lambda cam: int(cam.meta.get('frame_idx', -1)))

        best_i = min(
            range(len(front)),
            key=lambda i: abs(int(front[i].meta.get('frame_idx', -1)) - target_idx),
        )
        i0 = max(0, best_i - 1)
        i1 = min(len(front) - 1, best_i + 1)
        if i0 == i1:
            return float(self.kinematic.get('start_velo', 0.0))

        T_sg_to_hugsim = np.linalg.inv(sgb.T_hugsim_to_sg)
        c0 = T_sg_to_hugsim @ front[i0].get_extrinsic()
        c1 = T_sg_to_hugsim @ front[i1].get_extrinsic()
        p0 = c0[:3, 3]
        p1 = c1[:3, 3]
        f0 = int(front[i0].meta.get('frame_idx', 0))
        f1 = int(front[i1].meta.get('frame_idx', f0))
        dt = max(abs(f1 - f0) / float(fps), 1e-6)
        return float(np.linalg.norm(p1[[0, 2]] - p0[[0, 2]]) / dt)

    def kinematic_step(self, plan_first_local):
        """Teleport the ego to the first waypoint of the agent's plan + its
        heading, advance the clock by ``self.dt``, return new obs/info.

        Used by run_closed_loop's --kinematic_only mode to bypass iLQR and
        the bicycle integrator entirely. Useful for inspecting whether the
        agent's plan itself is reasonable, independent of controller
        tracking error.

        plan_first_local: (x_right_lidar, y_forward_lidar, heading_navsim)
            — first waypoint of the plan, in ego-local lidar coordinates
            with NAVSIM-convention heading (CCW from forward, +left).
        """
        self.timestamp += self.dt
        if self.planner is not None:
            self.render_kwargs['planning'] = self.planner.plan_traj(self.timestamp, self.ego_state)

        x_right, y_fwd, h_navsim = (float(plan_first_local[0]),
                                    float(plan_first_local[1]),
                                    float(plan_first_local[2]))

        # HUGSIM world frame is OpenCV-style at θ=0 (vab[0] = world +x = right,
        # vab[1] = world +z = forward). drivoR's lidar x_right ↔ ego-local +x
        # directly (no sign flip). drivoR's NAVSIM heading is +CCW=+left, but
        # HUGSIM yaw is +CW=+right (positive yaw rotates forward toward +x =
        # right), so the two yaw conventions are *opposite sign*.
        theta = float(self.vr[1])
        c, s = math.cos(theta), math.sin(theta)
        # R_y(θ) applied to (x_right, 0, y_fwd):
        #   world_x = x_right*cos(θ) + y_fwd*sin(θ)
        #   world_z = -x_right*sin(θ) + y_fwd*cos(θ)
        d_vab0 = x_right * c + y_fwd * s
        d_vab1 = -x_right * s + y_fwd * c

        self.vab[0] += d_vab0
        self.vab[1] += d_vab1
        # Flip sign: NAVSIM CCW-positive heading → HUGSIM CW-positive yaw.
        self.vr[1] = theta - h_navsim

        # Update velo / steer from this kinematic jump so future ego_status
        # inputs to the agent reflect the motion. Bicycle inverse-kinematics
        # for steer: θ̇ = v·tan(δ)/L → δ = atan(θ̇·L / max(v, ε)).
        # Sign also flipped on theta_dot (NAVSIM → HUGSIM).
        distance = math.sqrt(d_vab0 * d_vab0 + d_vab1 * d_vab1)
        self.velo = distance / max(self.dt, 1e-6)
        L = self.kinematic['Lr'] + self.kinematic['Lf']
        theta_dot = -h_navsim / max(self.dt, 1e-6)
        self.steer = math.atan2(theta_dot * L, max(self.velo, 1e-3))
        self.last_accel = 0.0
        self.last_steer_rate = 0.0

        # Same collision / off-route / route-complete probes as step().
        # Per latest user spec, bg + fg collisions DO terminate even in
        # kinematic-only mode — teleporting into background is a signal
        # that the plan is poor, not just controller error.
        terminated = False
        reward = 0
        verts = (self.ego[:3, :3] @ self.ego_verts.T).T + self.ego[:3, 3]
        verts = torch.from_numpy(verts.astype(np.float32)).cuda()
        bg_collision = bg_collision_det(self.points, verts)
        if bg_collision:
            terminated = True
            print('Collision with background (kinematic_only)')
        fg_collision = fg_collision_det(self.ego_box, self.objs_list)
        if fg_collision:
            terminated = True
            print('Collision with foreground (kinematic_only)')

        rc, dist = self.route_completion
        off_route = False
        off_route_threshold = float(getattr(self, 'off_route_threshold', 30.0))
        if dist > off_route_threshold:
            terminated = True
            off_route = True
            print('Far from preset trajectory')

        route_complete = bool(rc >= 1.0 - 1e-6)
        if route_complete:
            terminated = True
            print('Route complete (kinematic_only)')
        try:
            _, max_log_time = self.sg_log_time_bounds()
            log_complete = bool(float(self.timestamp) >= float(max_log_time) - 1e-6)
        except Exception:
            log_complete = False

        observation = self._get_obs()
        info = self._get_info()
        info['rc'] = rc
        info['collision']    = bg_collision or fg_collision
        info['bg_collision'] = bool(bg_collision)
        info['fg_collision'] = bool(fg_collision)
        info['off_route']    = off_route
        info['route_complete'] = bool(route_complete)
        info['log_complete'] = bool(log_complete)
        terminated = bool(terminated or log_complete)
        info['kinematic_only'] = True
        return observation, reward, terminated, False, info

    def render_debug_frame(self, fpv_image=None, plan_traj_ego=None,
                           plan_traj_world=None, trajectory_overlays=None,
                           past_traj_world=None, term_reason='',
                           view_mode='pullback'):
        """Pullback + FPV side-by-side debug view, with ego bbox + trajectory
        overlay. Thin wrapper over SGRenderBackend.render_debug_view that
        gathers the per-step state the backend needs (ego pose, intrinsics,
        bbox vertices, foreground actors).
        """
        if self.sg_backend is None:
            return None
        # Front camera intrinsics drive the debug view's output resolution
        # (the backend renders at SG native res and resizes to this).
        intrinsic = self.cam_params['CAM_FRONT']['intrinsic']
        # Ego bbox vertices in HUGSIM world frame (same transform the env
        # uses for bg_collision_det in step()).
        ego_verts_world = (self.ego[:3, :3] @ self.ego_verts.T).T + self.ego[:3, 3]
        return self.sg_backend.render_debug_view(
            ego_c2w_hugsim=self.ego,
            intrinsic=intrinsic,
            timestamp=self.timestamp,
            ego_verts_hugsim=ego_verts_world,
            collision_pts_hugsim=None,
            bg_pt_count=0,
            fg_actors=self.objs_list,
            fpv_image=fpv_image,
            plan_traj_ego=plan_traj_ego,
            plan_traj_world=plan_traj_world,
            trajectory_overlays=trajectory_overlays,
            past_traj_world=past_traj_world,
            term_reason=term_reason,
            # Suppress the backend's own bg_pts / pullback / DONE overlays —
            # run_closed_loop draws its own (larger, single set) labels on the
            # combined frame instead.
            draw_overlay_text=False,
            view_mode=view_mode,
        )

    def _get_obs_sg(self):
        """SG-backed observation: SGRenderBackend renders each cam from its
        own c2w (after the make_sg_camera fix). HUGSIM's cam_rect offset is
        a rendering-time rectification that SG's training data does not
        share — we drop it from the c2w we hand to SG so the resulting view
        sits at the actual cam pose, not 30 cm offset in the rect axis."""
        rgbs, semantics, depths = {}, {}, {}
        v2front = self.cam_params['CAM_FRONT']['v2c']
        for cam_name, params in self.cam_params.items():
            intrinsic = params['intrinsic']
            H, W = int(intrinsic['H']), int(intrinsic['W'])
            v2c = params['v2c']
            if params.get('sg_render', True):
                # NO cam_rect for SG — SG was trained on raw cam poses.
                vehicle_to_camera = params.get('vehicle_to_camera')
                if vehicle_to_camera is not None:
                    c2w = self.ego @ np.asarray(vehicle_to_camera, dtype=np.float64)
                else:
                    c2front_no_rect = v2front @ np.linalg.inv(v2c)
                    c2w = self.ego @ c2front_no_rect

                rgb = self.sg_backend.render(
                    c2w_hugsim=c2w,
                    ego_c2w_hugsim=self.ego,
                    intrinsic=intrinsic,
                    cam_name=cam_name,
                    timestamp=self.timestamp,
                )
            else:
                rgb = None
            if rgb is None:
                rgb = np.zeros((H, W, 3), dtype=np.uint8)
            rgbs[cam_name] = rgb
            # SG doesn't expose drivable-area semantics or metric depth in
            # the public render path, so zero these out — drivoR / LTF /
            # UniAD agents only read 'rgb'.
            semantics[cam_name] = np.zeros((H, W), dtype=np.uint8)
            depths[cam_name] = np.zeros((H, W), dtype=np.float32)
        return {'rgb': rgbs, 'semantic': semantics, 'depth': depths}
    
    def _get_info(self):
        wego_r, wego_t = pose2rt(self.ego)
        cam_poses, _, commands = self.ground_model
        ego_pos = np.asarray(self.ego, dtype=np.float64)[:3, 3]
        dist = np.sum((cam_poses[:, :3, 3] - ego_pos) ** 2, axis=-1)
        nearest_cam_idx = np.argmin(dist)
        command = commands[nearest_cam_idx]
        return {
            'ego_pos'  : wego_t.tolist(),
            'ego_rot'  : wego_r.tolist(),
            'ego_pose_hugsim': np.asarray(self.ego, dtype=np.float64).tolist(),
            'ego_velo' : self.velo,
            'ego_steer': self.steer,
            'accelerate': self.last_accel,
            'steer_rate': self.last_steer_rate,
            'timestamp': self.timestamp,
            'command': command,
            'ego_box': self.ego_box,
            'obj_boxes': self.objs_list,
            'cam_params': self.cam_params,
            # 'ego_verts': verts,
        }

    def close(self):
        if self.sg_backend is not None and hasattr(self.sg_backend, 'close'):
            self.sg_backend.close()
        self.sg_backend = None
        self.gaussians = None
        return super().close()
    
    def reset(self, seed=None, options=None):
        self._exact_ego_pose = None
        self.vr = deepcopy(self.start_vr)
        self.vab = deepcopy(self.start_vab)
        self.velo = deepcopy(self.start_velo)
        self.steer = deepcopy(self.kinematic['start_steer'])
        self.last_accel = 0
        self.last_steer_rate = 0
        self.timestamp = 0

        if self.planner is not None:
            self.render_kwargs['planning'] = self.planner.plan_traj(self.timestamp, self.ego_state)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        self.timestamp += self.dt
        if self.planner is not None:
            self.render_kwargs['planning'] = self.planner.plan_traj(self.timestamp, self.ego_state)
        steer_rate, acc = action['steer_rate'], action['acc']
        self.last_steer_rate, self.last_accel = steer_rate, acc
        L = self.kinematic['Lr'] + self.kinematic['Lf']
        self.velo += acc * self.dt
        self.steer += steer_rate * self.dt
        theta = self.vr[1]
        # print(theta / np.pi * 180, self.steer / np.pi * 180)
        self.vab[0] = self.vab[0] + self.velo * np.sin(theta) * self.dt
        self.vab[1] = self.vab[1] + self.velo * np.cos(theta) * self.dt
        self.vr[1] = theta + self.velo * np.tan(self.steer) / L * self.dt

        terminated = False
        reward = 0
        verts = (self.ego[:3, :3] @ self.ego_verts.T).T + self.ego[:3, 3]
        verts = torch.from_numpy(verts.astype(np.float32)).cuda()
        
        bg_collision = bg_collision_det(self.points, verts)
        if bg_collision:
            terminated = True
            print('Collision with background')
            reward = -100

        fg_collision = fg_collision_det(self.ego_box, self.objs_list)
        if fg_collision:
            terminated = True
            print('Collision with foreground')
            reward = -100

        rc, dist = self.route_completion
        off_route = False
        off_route_threshold = float(getattr(self, 'off_route_threshold', 30.0))
        if dist > off_route_threshold:
            terminated = True
            off_route = True
            print('Far from preset trajectory')
            reward = -50

        route_complete = bool(rc >= 1.0 - 1e-6)
        if route_complete:
            terminated = True
            print('Route complete')
        try:
            _, max_log_time = self.sg_log_time_bounds()
            log_complete = bool(float(self.timestamp) >= float(max_log_time) - 1e-6)
        except Exception:
            log_complete = False

        observation = self._get_obs()
        info = self._get_info()
        info['rc'] = rc
        info['collision']    = bg_collision or fg_collision
        info['bg_collision'] = bool(bg_collision)
        info['fg_collision'] = bool(fg_collision)
        info['off_route']    = off_route
        info['route_complete'] = bool(route_complete)
        info['log_complete'] = bool(log_complete)
        terminated = bool(terminated or log_complete)

        return observation, reward, terminated, False, info
