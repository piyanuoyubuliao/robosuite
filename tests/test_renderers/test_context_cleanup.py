import gc

import numpy as np
import pytest

from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim


SCENE = """<mujoco><worldbody>
<light pos="0 0 3"/>
<geom type="plane" size="2 2 .1" rgba=".3 .4 .5 1"/>
<geom type="box" pos="0 0 .2" size=".2 .2 .2" rgba=".9 .2 .1 1"/>
<camera name="test" pos="0 -2 2" xyaxes="1 0 0 0 1 1"/>
</worldbody></mujoco>"""


def make_sim():
    simulation = MjSim.from_xml_string(SCENE)
    MjRenderContextOffscreen(simulation, device_id=-1)
    return simulation


def test_collecting_old_context_preserves_live_camera():
    was_enabled = gc.isenabled()
    gc.disable()
    simulation = None
    try:
        old_simulation = make_sim()
        old_simulation.render(128, 128, camera_name="test")
        simulation = make_sim()
        before = simulation.render(128, 128, camera_name="test")
        assert before.std() > 10
        del old_simulation
        gc.collect()
        for _ in range(3):
            after = simulation.render(128, 128, camera_name="test")
            np.testing.assert_allclose(after, before, rtol=0, atol=1)
    finally:
        if simulation is not None:
            simulation.free()
        if was_enabled:
            gc.enable()


def test_invalid_framebuffer_raises_instead_of_returning_pixels():
    old_simulation = make_sim()
    simulation = make_sim()
    try:
        simulation.render(128, 128, camera_name="test")
        old_simulation._render_context_offscreen.con.free()
        with pytest.raises(RuntimeError, match="Offscreen rendering failed with OpenGL error"):
            simulation.render(128, 128, camera_name="test")
    finally:
        old_simulation.free()
        simulation.free()
