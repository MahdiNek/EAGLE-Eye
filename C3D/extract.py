import skvideo.io
from c3d import C3D
from keras.models import Model
from sports1M_utils import preprocess_input, decode_predictions
import numpy as np

base_model = C3D(weights='sports1M')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc6').output)

vid_path = 'streetGray.mp4'
vid = skvideo.io.vread(vid_path)
print(np.shape(vid))
# Select 16 frames from video
vid = vid[40:56]
x = preprocess_input(vid)
features = model.predict(x)
print(np.shape(features))