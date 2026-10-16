"""Advance exactly one physics tick, independently of the rendering interval."""

import math


def advance_physics(sim, scene, dt, *, render=False):
    before = sim.current_time
    scene.write_data_to_sim()
    sim.step(render=False)
    elapsed = sim.current_time - before
    if not math.isclose(elapsed, dt, rel_tol=1e-5, abs_tol=1e-7):
        raise RuntimeError(f"Expected one {dt:g}s physics step, advanced {elapsed:g}s")
    scene.update(dt)
    if render:
        sim.render()
        if not math.isclose(sim.current_time - before, dt, rel_tol=1e-5, abs_tol=1e-7):
            raise RuntimeError("Rendering advanced physics unexpectedly")
    return elapsed
