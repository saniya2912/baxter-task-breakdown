import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <body pos="0 0 0.1">
      <geom type="box" size="0.05 0.05 0.05" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
""")

renderer = mujoco.Renderer(model, 480, 640)
data = mujoco.MjData(model)

mujoco.mj_step(model, data)
renderer.update_scene(data)
img = renderer.render()

print("Rendered image shape:", img.shape)
