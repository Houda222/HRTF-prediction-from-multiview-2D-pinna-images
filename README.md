
Start by installing necessary packages in requirements.txt:

    pip install -r requirements.txt

Next, follow the instructions provided in the github repository of DepthAnything Model.
https://github.com/DepthAnything/Depth-Anything-V2.git 

To run the model, run the file inference.py

>>>>>> Notebooks:
training.ipynb: Jupyter notebook for training and evaluating the HRTF prediction model.

>>>>>> Scripts:
3DHRTF.py: model for 3D to HRTF.
HRTFNet_onefreq.py: Contains the HRTFNet class.
inference.py: Script for running inference.
metrics.py: Contains metric functions.
models.py: Contains models classes used in HRTFNet.
pointNet_utils.py: Utility functions for PointNet.
utils_d.py: Utility functions and dataset class SonicomDatabase.

├── 3DHRTF.py
├── HRTFNet.pth
├── checkpoints/
│   └── depth_anything_v2_vitl.pth
├── depth_anything_v2/
│   ├── dinov2_layers/
│   │   ├── attention.py
│   │   ├── block.py
│   │   ├── drop_path.py
│   │   ├── __init__.py
│   │   ├── layer_scale.py
│   │   ├── mlp.py
│   │   ├── patch_embed.py
│   ├── dinov2.py
│   ├── dpt.py
│   └── util/
├── HRTFNet_onefreq.py
├── inference.py
├── mean_hrtf.pt
├── metrics.py
├── models.py
├── onefreqmodel.ipynb
├── pointNet_utils.py
├── readME.md
├── requirements.txt
├── test_output.sofa
├── test.py
└── utils_d.py


