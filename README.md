# YOLO Object Detection

Deployed on Sieve at `developer-sievedata-com/yolo-object-detector`. You can also deploy it yourself using `sieve model` inside this repo's directory.

### Create a workflow using this model

`pip install https://storage.googleapis.com/sieve-client-package/sievedata-0.0.1.3-py3-none-any.whl`

#### Python
```Python
from sieve.api.client import SieveClient, SieveProject
from sieve.types.api import *
cli = SieveClient()

proj = SieveProject(
    name="object_detection",
    fps=5,
    store_data=True,
    workflow=SieveWorkflow([
        SieveLayer(
            iteration_type=SieveLayerIterationType.video,
            models=[
                SieveModel(
                    name="developer-sievedata-com/yolo-object-detector",
                )
            ]
        )
    ])
)
proj.create()
```

#### YAML
```
# workflow.yaml
fps: 5
store_data: true
layers:
- iteration: "video"
  models: 
  - model_name: developer-sievedata-com/yolo-object-detector
```
`sieve projects object_detection create -wf workflow.yaml`
