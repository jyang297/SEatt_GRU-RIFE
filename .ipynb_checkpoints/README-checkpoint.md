# Model structure:
1. Attention module: SE
2. Optical flow estimation: RIFE
3. Context Module: GRU
4. Fusion: Unet

# Inference
Modify the Config in the start of the inference files(inferenceVideo_*.py) for the inference:
```python
frame_path = root + "/frames"
output_path = root + "/outputs/outputs_choice"
pretrained_model_path = root + '/intrain_log'
shift = int(n)
```
1. The shift is used to shift the start point of selecting the start point of interpolation. The manual shift for inferenceVideo_realVersion.py should be 16, and for inferenceVideo_replaceVersion.py should be 18.
2. The frame_path should store the frames to be interpolated.
3. The output_path should be your desired output folder.
4. The pretrained model can be downloaded here at google drive:

## 1. Video place:
Put the video frames at the root as "frames". The name of the images should follows the format: frame_%04d, %i
The shape of the images size should be divisible by 16 like $1088 \times 2048$.

## 2. Inference
### 1. InferenceVideo_realVersion
InferenceVideo_realVersion.py is used to inplement the video interpolation. The input should be 4 consecutive 
frames. 
Run with
```python
python inferenceVideo_realVersion.py
```
### 2. InferenceVideo_replaceVersion
InferenceVideo_realVersion.py is used to inplement the video interpolation comparation. It will replace the origin even frames with the prediction of the model.
Run with
```python
python inferenceVideo_replaceVersion.py
```

