import time
import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <light pos="0 0 1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    <body pos="0 0 0.1">
      <geom type="box" size="0.05 0.05 0.05" rgba="0.2 0.6 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Keep running until user closes window
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)  # be gentle on CPU
except Exception as e:
    print("Viewer crashed with exception:", repr(e))
    raise
