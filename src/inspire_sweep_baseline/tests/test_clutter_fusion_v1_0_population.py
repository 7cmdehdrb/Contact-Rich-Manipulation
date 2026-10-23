"""Population schedules, exact geometry, masked reset snapshots and physics flags."""

import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_clutter_layout import validate_scenes


@pytest.fixture
def population_modules(load_mdp):
    from _source_units import import_unit
    return SimpleNamespace(
        population=import_unit("contact_sweep_clutter_fusion_v1_0.population"),
        layout=import_unit("contact_sweep_clutter_fusion_v1_0.layout"),
        coordinates=importlib.import_module("_baseline_cpu_pkg.assets.scenes.contact_sweep_clutter_layout"),
        events=import_unit("contact_sweep_clutter_fusion_v1_0.events"),
        samplers=load_mdp("goal_events"),
    )


@pytest.mark.parametrize("iteration,gray,white", [
    (0, 1, 1), (2999, 1, 1), (3000, 1, 2), (4499, 1, 2),
    (4500, 2, 2), (5999, 2, 2), (6000, 2, 3), (7499, 2, 3),
    (7500, 3, 3), (99999, 3, 3),
])
def test_population_boundaries_only_update_future_resets(population_modules, iteration, gray, white):
    event = SimpleNamespace(params={"gray_obstacles": 999, "white_obstacles": 999})
    env = SimpleNamespace(
        learning_iteration=iteration, cfg=SimpleNamespace(max_gray_obstacles=3, max_white_obstacles=3),
        _active_object_mask=torch.tensor([[True, True, False, False, True, False, False]]),
        event_manager=SimpleNamespace(get_term_cfg=lambda name: event, set_term_cfg=lambda name, cfg: None),
    )
    before = env._active_object_mask.clone()
    for ids in (None, [0], None):
        result = population_modules.population.obstacle_population_schedule(env, ids)
        assert result == {"gray_obstacles": gray, "white_obstacles": white}
        assert event.params == result
        assert torch.equal(env._active_object_mask, before)
    env.learning_iteration = 0
    assert population_modules.population.obstacle_population_schedule(env, None) == {
        "gray_obstacles": 1, "white_obstacles": 1,
    }


def test_population_caps_and_empty_capacity(population_modules):
    mod = population_modules.population
    event = SimpleNamespace(params={})
    env = SimpleNamespace(
        learning_iteration=9000, cfg=SimpleNamespace(max_gray_obstacles=0, max_white_obstacles=2),
        event_manager=SimpleNamespace(get_term_cfg=lambda name: event, set_term_cfg=lambda name, cfg: None),
    )
    assert mod.obstacle_population_schedule(env, None) == {"gray_obstacles": 0, "white_obstacles": 2}
    assert mod.active_population_mask(2, 0, 0, 0, 0, "cpu").tolist() == [[True], [True]]
    with pytest.raises(ValueError, match="nonnegative integer"):
        mod.validate_capacity(-1, 2)
    with pytest.raises(ValueError, match="capacity"):
        mod.active_population_mask(1, 1, 1, 2, 1, "cpu")


@pytest.mark.parametrize("gray,white", [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)])
def test_all_population_layouts_preserve_fixed_commands_and_source_safety(population_modules, validate_scenes, gray, white):
    mod = population_modules
    count = 1024
    torch.manual_seed(20261022)
    target_xy = mod.samplers._sample_uniform_disk(count, 0.02, "cpu")
    direction = mod.samplers._sample_planar_unit_vectors(count, "cpu", 0.0)
    distance = mod.samplers._goal_ray_distance(target_xy, direction, .15, (.14, .44))
    target = mod.coordinates.shelf_xy_to_generator_cm(target_xy.numpy())
    goal = mod.coordinates.shelf_xy_to_generator_cm((target_xy + direction * distance[:, None]).numpy())
    before_target, before_goal = target.copy(), goal.copy()
    shared = mod.layout.DEFAULT_LAYOUT_CONFIG
    env = SimpleNamespace(cfg=SimpleNamespace(seed=19))
    generator = mod.layout.population_layout_generator(env, gray, white)
    batch = generator.generate_batch(target, goal)
    assert batch.obstacles.shape == (count, gray + white, 2)
    assert batch.config.gray_obstacles == gray
    assert batch.config.rear_obstacles == 1
    report = validate_scenes(batch)
    assert report["valid"], report
    assert report["minima"]["surface_gap_cm"] >= 1.6 - 1e-8
    assert report["minima"]["wrist_projection_clearance_cm"] >= 2.3 - 1e-8
    np.testing.assert_array_equal(target, before_target)
    np.testing.assert_array_equal(goal, before_goal)
    assert shared.obstacles == 3 and shared.gray_obstacles == 1 and shared.rear_obstacles == 1
    assert batch.config is not shared
    assert mod.layout.population_layout_generator(env, gray, white) is generator
    # A different count obtains a different immutable config/RNG instance.
    assert mod.layout.population_layout_generator(env, gray + 1, white) is not generator


def _population_env(make_env, count=4, max_gray=3, max_white=3):
    env = make_env(count)
    env.cfg = SimpleNamespace(seed=42, max_gray_obstacles=max_gray, max_white_obstacles=max_white)
    object_count = 1 + max_gray + max_white
    env._active_object_mask = torch.zeros(count, object_count, dtype=torch.bool)
    env._active_object_mask[:, 0] = True
    env._initial_obstacle_observation = torch.zeros(count, 40)
    shelf = env.scene["shelf"].data
    shelf.root_pos_w = torch.tensor([-.7, 0., 0.]).expand(count, -1).clone()
    shelf.root_quat_w = torch.tensor([1., 0., 0., 0.]).expand(count, -1).clone()
    objects = env.scene["object_collection"]
    objects.num_objects = object_count
    objects.object_names = ["mass_cylinder", *(f"obstacle_{i}" for i in range(object_count - 1))]
    states = torch.zeros(count, object_count, 13)
    states[..., 3] = 1
    objects.data = SimpleNamespace(object_link_state_w=states, object_pos_w=states[..., :3], object_quat_w=states[..., 3:7])
    objects.write_object_link_state_to_sim = lambda values, env_ids: states.__setitem__(env_ids, values)
    objects.disabled_rows = torch.zeros(object_count * count, 1, dtype=torch.uint8)
    objects.woken_rows = []

    def set_disable_gravities(flags, rows):
        objects.disabled_rows[rows] = flags[rows]
        objects.last_set_rows = rows.clone()

    objects.root_physx_view = SimpleNamespace(
        set_disable_gravities=set_disable_gravities,
        wake_up=lambda rows: objects.woken_rows.append(rows.clone()),
    )
    return env


def test_reset_keeps_parent_target_goal_and_disables_unused_slots(population_modules, make_env):
    mod = population_modules
    env = _population_env(make_env)
    torch.manual_seed(91)
    target = mod.samplers._sample_uniform_disk(4, .02, "cpu")
    direction = mod.samplers._sample_planar_unit_vectors(4, "cpu", math.radians(30))
    length = mod.samplers._goal_ray_distance(target, direction, .15, (.14, .44))
    torch.manual_seed(91)
    mod.events.randomize_population_clutter_scene(env, None, observation_noise=0, visibility_flip_probability=0)
    objects = env.scene["object_collection"]
    state = objects.data.object_link_state_w
    torch.testing.assert_close(state[:, 0, :2], target + torch.tensor([-.7, 0.]))
    torch.testing.assert_close(env.sweep_dir[:, :2], direction * length[:, None])
    expected = torch.tensor([True, True, False, False, True, False, False]).expand(4, -1)
    assert torch.equal(env._active_object_mask, expected)
    assert torch.equal(objects.disabled_rows, (~expected).T.reshape(-1, 1).to(torch.uint8))
    assert (state[..., 2][~expected] == mod.population.PARKING_HEIGHT).all()
    slots = env._initial_obstacle_observation.reshape(4, 10, 4)
    assert slots[:, 2:].count_nonzero() == 0
    assert (slots[:, 1:, 0] <= slots[:, :-1, 0]).all()


def test_partial_reset_activates_new_counts_only_in_reset_rows(population_modules, make_env):
    env = _population_env(make_env)
    event = population_modules.events.randomize_population_clutter_scene
    event(env, None, visibility_flip_probability=0)
    states = env.scene["object_collection"].data.object_link_state_w.clone()
    before_mask, before_obs = env._active_object_mask.clone(), env._initial_obstacle_observation.clone()
    event(env, [2], gray_obstacles=3, white_obstacles=3, visibility_flip_probability=0)
    assert env._active_object_mask[2].all()
    for rows in ([0, 1, 3],):
        assert torch.equal(env._active_object_mask[rows], before_mask[rows])
        assert torch.equal(env._initial_obstacle_observation[rows], before_obs[rows])
        assert torch.equal(env.scene["object_collection"].data.object_link_state_w[rows], states[rows])
    objects = env.scene["object_collection"]
    assert objects.last_set_rows.tolist() == [2, 6, 10, 14, 18, 22, 26]
    assert objects.disabled_rows[objects.last_set_rows].count_nonzero() == 0


def test_inactive_capacity_cannot_occlude_or_become_visible_after_flip(population_modules, make_env):
    env = _population_env(make_env, count=1)
    env._active_object_mask[0, 1] = True
    positions = env.scene["object_collection"].data.object_pos_w
    positions[0] = torch.tensor([
        [-.70, .20, 1.05], [-.70, 0., 1.05], [-.50, 0., 1.05],
        [-.60, 0., 1.05], [-.55, 0., 1.05], [-.52, 0., 1.05], [-.54, 0., 1.05],
    ])
    slots = env._initial_obstacle_observation.reshape(1, 10, 4)
    slots[:, :6] = 1
    population_modules.events.apply_population_visibility(env, None, visibility_flip_probability=0)
    assert slots[0, 0].tolist() == [1, 1, 1, 1]
    assert slots[0, 1:].count_nonzero() == 0
    slots[:, :6] = 1
    population_modules.events.apply_population_visibility(env, None, visibility_flip_probability=1)
    assert slots.count_nonzero() == 0


def test_zero_capacity_reset_keeps_target_only(population_modules, make_env):
    env = _population_env(make_env, count=2, max_gray=0, max_white=0)
    population_modules.events.randomize_population_clutter_scene(env, None)
    assert env._active_object_mask.all()
    assert env._initial_obstacle_observation.count_nonzero() == 0
    assert env.scene["object_collection"].disabled_rows.count_nonzero() == 0


def _observation_slots():
    # X encodes identity, Y specifies left-to-right order: A, B, C, D.
    slots = torch.zeros(1, 10, 4)
    slots[0, 0] = torch.tensor([1., 3., -1., 30.])  # C
    slots[0, 2] = torch.tensor([1., 1., 3., 10.])   # A
    slots[0, 5] = torch.tensor([1., 4., -3., 40.])  # D
    slots[0, 7] = torch.tensor([1., 2., 1., 20.])   # B
    slots[0, 1] = torch.tensor([0., 999., 999., 999.])
    return slots


def test_observation_sort_is_left_to_right_compact_and_preserves_xyz(population_modules):
    slots = _observation_slots()
    original = slots.clone()
    result = population_modules.events.pack_obstacle_observations(slots, 0.)
    assert result[0, :, 0].tolist() == [1.] * 4 + [0.] * 6
    assert result[0, :4, 1].tolist() == [1., 2., 3., 4.]
    assert result[0, :4, 2].tolist() == [3., 1., -1., -3.]
    assert result[0, :4, 3].tolist() == [10., 20., 30., 40.]
    assert result[:, 4:].count_nonzero() == 0
    assert torch.equal(slots, original)


def test_observation_swaps_use_updated_pairs_and_never_empty_suffix(population_modules):
    result = population_modules.events.pack_obstacle_observations(_observation_slots(), 1.)
    # A B C D -> B A C D -> B C A D -> B C D A, not disjoint pairs.
    assert result[0, :4, 1].tolist() == [2., 3., 4., 1.]
    assert result[0, :, 0].tolist() == [1.] * 4 + [0.] * 6
    assert result[:, 4:].count_nonzero() == 0


def test_observation_ten_percent_trial_boundary_and_sequential_semantics(population_modules, monkeypatch):
    trials = torch.ones(1, 9)
    trials[0, :3] = torch.tensor([.099, .05, .1])
    monkeypatch.setattr(population_modules.events.torch, "rand", lambda *args, **kwargs: trials.clone())
    result = population_modules.events.pack_obstacle_observations(_observation_slots())
    # First A/B swaps, next A/C swaps, .1 itself does not pass the <.1 test.
    assert result[0, :4, 1].tolist() == [2., 3., 1., 4.]
    assert result[:, 4:].count_nonzero() == 0


def test_observation_empty_single_and_equal_coordinates(population_modules):
    slots = torch.zeros(3, 10, 4)
    slots[1, 8] = torch.tensor([1., 12., 0., 2.])
    slots[2, 1] = torch.tensor([1., 7., 0., 2.])
    slots[2, 8] = torch.tensor([1., 9., 0., 3.])
    result = population_modules.events.pack_obstacle_observations(slots, 0.)
    assert result[0].count_nonzero() == 0
    assert result[1, 0].tolist() == [1., 12., 0., 2.]
    assert result[1, 1:].count_nonzero() == 0
    assert result[2, :2, 1].tolist() == [7., 9.]  # Stable tie order.


def test_observation_swap_rate_and_seed_reproducibility(population_modules):
    slots = torch.zeros(20000, 10, 4)
    slots[:, 0] = torch.tensor([1., 1., 1., 0.])
    slots[:, 1] = torch.tensor([1., 2., -1., 0.])
    torch.manual_seed(521)
    result = population_modules.events.pack_obstacle_observations(slots)
    rate = (result[:, 0, 1] == 2.).float().mean().item()
    assert .09 < rate < .11
    assert result[:, 2:].count_nonzero() == 0
    torch.manual_seed(521)
    assert torch.equal(result, population_modules.events.pack_obstacle_observations(slots))


@pytest.mark.parametrize("probability", [-.1, 1.1, math.nan])
def test_observation_invalid_swap_probability(population_modules, probability):
    with pytest.raises(ValueError, match="adjacent_swap_probability"):
        population_modules.events.pack_obstacle_observations(_observation_slots(), probability)


def test_extended_capacity_is_opt_in(population_modules):
    mod = population_modules.population
    with pytest.raises(ValueError):
        mod.validate_capacity(7, 7)
    mask = mod.active_population_mask(4, 7, 7, 7, 7, 'cpu', observation_slots=14)
    assert mask.shape == (4, 15) and mask.all()
