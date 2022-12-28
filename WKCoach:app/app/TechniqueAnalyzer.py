import tensorflow as tf
from tensorflow import keras 

from PIL import Image, ImageOps
import numpy as np
import json


def analyseMainFrames(imageList):
    analyzedData = {}
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    LABELS = ["stance", "execution"]
    STANCE_LABELS = ["correct", "incorrect"] 
    EXE_LABELS =["correct", "incorrect"] 
    # Load the model
    modelStance = keras.models.load_model('Models\Stance\keras_model.h5')
    #modelLegMovement = keras.models.load_model('Models\Execution_model\model.h5')
    modelExecution =keras.models.load_model('Models\Execution\keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    for label in LABELS:
        # Replace this with the path to your image
        image = Image.open(imageList[label])

        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # display the resized image
        # image.show()

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        if label == 'stance':
            print("STANCE ANALYSIS")
            prediction = modelStance.predict(data)
            print(prediction)
            idxs = prediction.argmax(axis=-1)
            for (i, j) in enumerate(idxs):
                print(STANCE_LABELS[j])
                analyzedData["stance"] = STANCE_LABELS[j]
       
        if label == 'execution':
            print("EXE ANALYSIS")
            prediction = modelExecution.predict(data)
            print(prediction)
            idxs = prediction.argmax(axis=-1)
            for (i, j) in enumerate(idxs):
                print(EXE_LABELS[j])
                analyzedData["execution"] = EXE_LABELS[j]
    json_data = json.dumps(analyzedData)
    return json_data
