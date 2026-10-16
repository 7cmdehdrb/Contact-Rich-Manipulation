"""Planar spawn/goal geometry, direction preservation, and reset isolation."""

from types import SimpleNamespace

import pytest
import torch


def _set_shelf_pose(env, position=None, rotation=None):
    count = env.num_envs
    if position is None:
        position = env.scene.env_origins + torch.tensor([-0.7, 0.0, 0.0])
    if rotation is None:
        rotation = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(count, 4).clone()
    env.scene["shelf"].data.root_pos_w = position
    env.scene["shelf"].data.root_quat_w = rotation


def test_spawn_is_uniform_by_area_in_exact_centered_two_cm_disk(load_mdp, make_env):
    events = load_mdp("goal_events")
    count = 20_000
    env = make_env(count)
    _set_shelf_pose(env)
    torch.manual_seed(461)
    events.randomize_goal_object_scene(env, None)
    state = env.scene["object_collection"].data.object_link_state_w[:, 0]
    local = state[:, :3] - env.scene["shelf"].data.root_pos_w
    radii_squared = local[:, :2].square().sum(dim=-1)
    assert radii_squared.max().item() <= 0.02**2 + 1e-8
    # Uniform area puts half the probability inside R/sqrt(2), unlike uniform R.
    assert (radii_squared < 0.02**2 / 2).float().mean().item() == pytest.approx(0.5, abs=0.012)
    assert torch.allclose(local[:, :2].mean(0), torch.zeros(2), atol=0.00025)
    assert torch.allclose(local[:, 2], torch.full((count,), 1.05))
    assert state[:, 7:].count_nonzero() == 0
    assert torch.allclose(torch.linalg.vector_norm(state[:, 3:7], dim=-1), torch.ones(count), atol=1e-6)
    goals = local[:, :2] + env.sweep_dir[:, :2]
    assert torch.all(goals.abs() <= torch.tensor([0.14, 0.44]) + 1e-6)
    distances = torch.linalg.vector_norm(env.sweep_dir, dim=-1)
    assert distances.max().item() <= 0.15 + 1e-6
    assert torch.any(distances < 0.149)
    assert torch.any(distances > 0.14999)
    # Clipping distance must not distort the distribution of sampled azimuths.
    angle = torch.atan2(env.sweep_dir[:, 1], env.sweep_dir[:, 0])
    histogram = torch.histc(angle, bins=16, min=-torch.pi, max=torch.pi)
    assert torch.all((histogram - count / 16).abs() < 150)


def test_shortening_preserves_exact_requested_azimuth_and_spawn_disk(load_mdp, make_env, monkeypatch):
    events = load_mdp("goal_events")
    env = make_env(4)
    _set_shelf_pose(env)
    spawn = torch.tensor([[0.02, 0.0], [-0.02, 0.0], [0.0, 0.02], [0.0, 0.0]])
    direction = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    monkeypatch.setattr(events, "_sample_uniform_disk", lambda count, radius, device: spawn.clone())
    monkeypatch.setattr(events, "_sample_planar_unit_vectors", lambda count, device: direction.clone())
    events.randomize_goal_object_scene(env, torch.arange(4))
    actual_spawn = env.scene["object_collection"].data.object_pos_w[:, 0, :2] - env.scene["shelf"].data.root_pos_w[:, :2]
    assert torch.allclose(actual_spawn, spawn, atol=1e-7)
    displacement = env.sweep_dir[:, :2]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    assert torch.allclose(distance, torch.tensor([0.12, 0.12, 0.15, 0.15]), atol=1e-7)
    assert torch.allclose(displacement / distance[:, None], direction, atol=1e-7)
    assert torch.allclose((actual_spawn + displacement)[:2, 0], torch.tensor([0.14, -0.14]), atol=1e-7)


def test_reset_tracks_current_shelf_pose_and_preserves_unselected_rows(load_mdp, make_env, tensor_math):
    events = load_mdp("goal_events")
    env = make_env(3)
    env.scene.env_origins[2] = torch.tensor([2.0, 3.0, 4.0])
    angles = torch.tensor([0.0, 0.0, 0.17])
    rotation = tensor_math.quat_from_euler_xyz(angles, torch.zeros(3), torch.tensor([0.0, 0.0, 1.2]))
    _set_shelf_pose(env, rotation=rotation)
    env.target_id[:2] = 7
    env.target_width[:2] = 0.09
    before = env.scene["object_collection"].data.object_link_state_w.clone()
    before_vector = env.sweep_dir.clone()
    events.randomize_goal_object_scene(env, [2])
    state = env.scene["object_collection"].data.object_link_state_w
    assert torch.equal(state[:2], before[:2])
    assert torch.equal(env.sweep_dir[:2], before_vector[:2])
    assert env.target_id[:, 0].tolist() == [7, 7, 0]
    assert env.target_width[:, 0].tolist() == pytest.approx([0.09, 0.09, 0.04])
    shelf = env.scene["shelf"].data
    local = tensor_math.quat_apply_inverse(rotation[2:3], state[2:3, 0, :3] - shelf.root_pos_w[2:3])
    assert local[0, 2].item() == pytest.approx(1.05, abs=1e-6)
    assert torch.linalg.vector_norm(local[:, :2]).item() <= 0.020001
    local_vector = tensor_math.quat_apply_inverse(rotation[2:3], env.sweep_dir[2:3])
    assert local_vector[0, 2].item() == pytest.approx(0.0, abs=1e-6)
    object_up = tensor_math.quat_apply(state[2:3, 0, 3:7], torch.tensor([[0.0, 0.0, 1.0]]))
    shelf_up = tensor_math.quat_apply(rotation[2:3], torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(object_up, shelf_up, atol=1e-6)
    # The inherited command must preserve this goal after live object motion.
    commands = load_mdp("commands")
    env.target_id[:2] = 0
    command = commands.SweepGoalCommand(SimpleNamespace(asset_name="object_collection"), env)
    goal = command.command.clone()
    assert torch.allclose(goal[2, :3], state[2, 0, :3] + env.sweep_dir[2])
    state[2, 0, :3] += 0.07
    command._update_command()
    assert torch.equal(command.command, goal)


@pytest.mark.parametrize("params", [
    {"spawn_radius": -0.01}, {"spawn_radius": 0.15}, {"goal_distance": 0.0},
    {"goal_distance": float("nan")}, {"object_radius": 0.0}, {"edge_clearance": -0.01},
    {"shelf_half_extents": (0.05, 0.50)}, {"shelf_half_extents": (0.2,)},
])
def test_invalid_sampling_parameters_fail_before_any_reset(load_mdp, make_env, params):
    events = load_mdp("goal_events")
    env = make_env()
    before = env.scene["object_collection"].data.object_link_state_w.clone()
    with pytest.raises(ValueError):
        events.randomize_goal_object_scene(env, [0], **params)
    assert torch.equal(env.scene["object_collection"].data.object_link_state_w, before)


def test_empty_reset_and_multiple_objects(load_mdp, make_env):
    events = load_mdp("goal_events")
    env = make_env()
    before = env.scene["object_collection"].data.object_link_state_w.clone()
    events.randomize_goal_object_scene(env, [])
    assert torch.equal(env.scene["object_collection"].data.object_link_state_w, before)
    env.scene["object_collection"].num_objects = 2
    with pytest.raises(RuntimeError, match="exactly one cylinder"):
        events.randomize_goal_object_scene(env, [0])


def test_excluded_x_goal_cones_are_uniform_in_shelf_frame_after_rotation(load_mdp, make_env, tensor_math):
    import math

    events = load_mdp("goal_events")
    count = 20_000
    env = make_env(count + 1)
    rotation = tensor_math.quat_from_euler_xyz(
        torch.full((count + 1,), 0.1), torch.zeros(count + 1), torch.full((count + 1,), 1.2)
    )
    _set_shelf_pose(env, rotation=rotation)
    before = env.scene["object_collection"].data.object_link_state_w[0].clone()
    before_dir = env.sweep_dir[0].clone()
    torch.manual_seed(871)
    alpha = math.radians(15.0)
    ids = torch.arange(1, count + 1)
    events.randomize_goal_object_scene(env, ids, excluded_x_half_angle=alpha)
    assert torch.equal(env.scene["object_collection"].data.object_link_state_w[0], before)
    assert torch.equal(env.sweep_dir[0], before_dir)
    displacement = tensor_math.quat_apply_inverse(rotation[ids], env.sweep_dir[ids])
    angle = torch.atan2(displacement[:, 1], displacement[:, 0]).remainder(2.0 * math.pi)
    assert torch.all((angle >= alpha - 1e-6) & (angle <= 2 * math.pi - alpha + 1e-6))
    assert torch.all((angle - math.pi).abs() >= alpha - 1e-6)
    # Map the allowed two arcs to one contiguous interval to test uniformity.
    width = math.pi - 2.0 * alpha
    coordinate = angle - alpha - (angle > math.pi) * (2.0 * alpha)
    histogram = torch.histc(coordinate, bins=12, min=0.0, max=2.0 * width)
    assert torch.all((histogram - count / 12).abs() < 150)
    position = env.scene["object_collection"].data.object_link_state_w[ids, 0, :3]
    spawn = tensor_math.quat_apply_inverse(rotation[ids], position - env.scene["shelf"].data.root_pos_w[ids])
    assert torch.all(torch.linalg.vector_norm(spawn[:, :2], dim=-1) <= 0.020001)
    assert torch.all((spawn[:, :2] + displacement[:, :2]).abs() <= torch.tensor([0.14, 0.44]) + 1e-6)
    assert torch.allclose(displacement[:, 2], torch.zeros(count), atol=1e-6)


@pytest.mark.parametrize("angle", [-0.01, torch.pi / 2, torch.inf, torch.nan])
def test_invalid_exclusion_angle_does_not_modify_scene(load_mdp, make_env, angle):
    events = load_mdp("goal_events")
    env = make_env()
    before = env.scene["object_collection"].data.object_link_state_w.clone()
    with pytest.raises(ValueError):
        events.randomize_goal_object_scene(env, None, excluded_x_half_angle=angle)
    assert torch.equal(env.scene["object_collection"].data.object_link_state_w, before)
