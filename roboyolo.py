import os
from roboflow import Roboflow
import inference

# os.environ["ROBOFLOW_API_KEY"] = "YoMT683Ih3lWqAmTmLOv"


rf = Roboflow(api_key="YoMT683Ih3lWqAmTmLOv")
project = rf.workspace().project("uitrain")
model = project.version(398).model

# infer on a local image
print(model.predict("./image/wss.png", confidence=50, overlap=50).json())
