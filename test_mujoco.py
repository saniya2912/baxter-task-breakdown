import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 0.1">
      <geom type="box" size="0.05 0.05 0.05" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
""")

data = mujoco.MjData(model)

for _ in range(10):
    mujoco.mj_step(model, data)

print("MuJoCo step OK")
