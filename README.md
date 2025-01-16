
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

|   .gitignore
|   README.md
|   requirements.txt
|   test_output.sofa
|   tree.txt
|   __init__.py
|
+---checkpoints
|       depth_anything_v2_vitl.pth
|       HRTFNet.pth
|       mean_hrtf.pt
|
+---data
|       ear_extraction.py
|       EDA.ipynb
|
+---model
|   |   3DHRTF.py
|   |   HRTFNet_onefreq.py
|   |   models.py
|   |
|   +---depth_anything_v2
|   |   |   dinov2.py
|   |   |   dpt.py
|   |   |
|   |   +---dinov2_layers
|   |   |   |   attention.py
|   |   |   |   block.py
|   |   |   |   drop_path.py
|   |   |   |   layer_scale.py
|   |   |   |   mlp.py
|   |   |   |   patch_embed.py
|   |   |   |   swiglu_ffn.py
|   |   |   |   __init__.py
|   |   |   |
|   |   |   \---__pycache__
|   |   |           attention.cpython-310.pyc
|   |   |           attention.cpython-39.pyc
|   |   |           block.cpython-310.pyc
|   |   |           block.cpython-39.pyc
|   |   |           drop_path.cpython-310.pyc
|   |   |           drop_path.cpython-39.pyc
|   |   |           layer_scale.cpython-310.pyc
|   |   |           layer_scale.cpython-39.pyc
|   |   |           mlp.cpython-310.pyc
|   |   |           mlp.cpython-39.pyc
|   |   |           patch_embed.cpython-310.pyc
|   |   |           patch_embed.cpython-39.pyc
|   |   |           swiglu_ffn.cpython-310.pyc
|   |   |           swiglu_ffn.cpython-39.pyc
|   |   |           __init__.cpython-310.pyc
|   |   |           __init__.cpython-39.pyc
|   |   |
|   |   +---util
|   |   |   |   blocks.py
|   |   |   |   transform.py
|   |   |   |
|   |   |   \---__pycache__
|   |   |           blocks.cpython-310.pyc
|   |   |           blocks.cpython-39.pyc
|   |   |           transform.cpython-310.pyc
|   |   |           transform.cpython-39.pyc
|   |   |
|   |   \---__pycache__
|   |           dinov2.cpython-310.pyc
|   |           dinov2.cpython-39.pyc
|   |           dpt.cpython-310.pyc
|   |           dpt.cpython-39.pyc
|   |
|   \---__pycache__
|           HRTFNet_onefreq.cpython-39.pyc
|           models.cpython-39.pyc
|
+---scripts
|       inference.py
|       test.py
|       training.ipynb
|
+---test_data
|       P0002_left_0.png
|       P0002_left_1.png
|       P0002_left_10.png
|       P0002_left_11.png
|       P0002_left_12.png
|       P0002_left_13.png
|       P0002_left_14.png
|       P0002_left_15.png
|       P0002_left_16.png
|       P0002_left_17.png
|       P0002_left_18.png
|       P0002_left_2.png
|       P0002_left_3.png
|       P0002_left_4.png
|       P0002_left_5.png
|       P0002_left_6.png
|       P0002_left_7.png
|       P0002_left_8.png
|       P0002_left_9.png
|       P0002_right_0.png
|       P0002_right_1.png
|       P0002_right_10.png
|       P0002_right_11.png
|       P0002_right_12.png
|       P0002_right_13.png
|       P0002_right_14.png
|       P0002_right_15.png
|       P0002_right_16.png
|       P0002_right_17.png
|       P0002_right_18.png
|       P0002_right_2.png
|       P0002_right_3.png
|       P0002_right_4.png
|       P0002_right_5.png
|       P0002_right_6.png
|       P0002_right_7.png
|       P0002_right_8.png
|       P0002_right_9.png
|
\---utils
    |   metrics.py
    |   pointNet_utils.py
    |   utils_d.py
    |
    \---__pycache__
            pointNet_utils.cpython-39.pyc


We really liked the project and we are working on another method, that looks promising, and that we are very close to finish but couldn't due too time constraints. If it works, we can submit it later.