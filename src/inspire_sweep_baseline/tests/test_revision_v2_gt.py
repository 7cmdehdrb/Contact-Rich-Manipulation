"""GT Actor receives live target XYZ while keeping the parent's obstacle boundary."""

import copy

import pytest
import torch

from test_revision_observations import revision_models, _observations, ROUTING


GT_ROUTING = {"actor": ["policy", "privileged", "obstacles"], "critic": ROUTING["critic"]}


def test_gt_actor_target_gradient_and_original_obstacle_boundary(revision_models):
    torch.manual_seed(42)
    obs = _observations()
    for value in obs.values():
        value.requires_grad_()
    actor = revision_models.RevisionGTModel(obs, GT_ROUTING, "actor", 8)
    parent = revision_models.RevisionModel(obs, ROUTING, "actor", 8)
    critic = revision_models.RevisionModel(obs, GT_ROUTING, "critic", 1)
    latent = actor.get_latent(obs)
    assert actor.obs_groups == ["policy", "privileged"]
    assert actor.obs_dim == 60 and latent.shape == (8, 76)
    assert parent.obs_dim == 57 and parent.mlp[0].in_features == 73
    assert critic.obs_dim == 60 and critic.mlp[0].in_features == 76
    torch.testing.assert_close(latent[:, :57], obs["policy"])
    torch.testing.assert_close(latent[:, 57:60], obs["privileged"])
    actions = actor(obs)
    grads = torch.autograd.grad(actions.sum(), tuple(obs.values()), allow_unused=True)
    assert all(grads[i] is not None and grads[i].abs().sum() > 0 for i in (0,1,2))
    assert grads[3] is None  # Actor must not see live obstacle observations
    changed = obs.clone()
    changed["privileged"] = obs["privileged"] + 1.
    assert not torch.allclose(actor(changed), actions)
    torch.testing.assert_close(parent(changed), parent(obs))
    changed = obs.clone()
    changed["critic_obstacles"] = torch.full((8,64), float('nan'))
    torch.testing.assert_close(actor(changed), actions)


def test_gt_export_and_checkpoint_preserve_124d_input(revision_models, tmp_path):
    obs = _observations(1)
    actor = revision_models.RevisionGTModel(obs, GT_ROUTING, "actor", 8)
    exported = actor.as_jit()
    assert exported.get_dummy_inputs()[0].shape == (1,124)
    inputs = torch.cat([obs[name] for name in GT_ROUTING["actor"]], dim=-1)
    scripted = torch.jit.script(exported)
    torch.testing.assert_close(scripted(inputs), actor(obs))
    checkpoint = tmp_path / 'gt.pt'
    torch.save(actor.state_dict(), checkpoint)
    restored = revision_models.RevisionGTModel(obs, GT_ROUTING, "actor", 8)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(obs), actor(obs), atol=0., rtol=0.)


@pytest.mark.parametrize('failure', ['missing_target', 'wrong_target_width', 'live_obstacles', 'old_policy_width'])
def test_gt_rejects_wrong_routing_and_shapes(revision_models, failure):
    obs, routing = _observations(), copy.deepcopy(GT_ROUTING)
    if failure == 'missing_target':
        routing['actor'] = ['policy','obstacles']
    elif failure == 'wrong_target_width':
        obs['privileged'] = torch.zeros(8,6)
    elif failure == 'live_obstacles':
        routing['actor'][-1] = 'critic_obstacles'
    else:
        obs['policy'] = torch.zeros(8,60)
    with pytest.raises(ValueError):
        revision_models.RevisionGTModel(obs, routing, 'actor', 8)
