from tensorflow.keras.models import load_model
from process import process_nibabel
import numpy as np


def test_pnemonia(file_path):
    model=load_model("3d_image_classification.h5")
    preprocessed_volume = process_nibabel(file_path)
    prediction = model.predict(np.expand_dims(preprocessed_volume, axis=0))[0]
    return prediction[0]

# print(test_pnemonia(test_image))