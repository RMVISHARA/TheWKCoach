import cv2
import time
import numpy as np
from imutils import paths
from PIL import Image, ImageOps
import os,shutil
import json
import tensorflow as tf
from tensorflow import keras 
import cv2
from PIL import Image, ImageOps

model = keras.models.load_model('Models\converted_keras\keras_model.h5', compile=False)



def classifyFrames(image,count):

    np.set_printoptions(suppress=True)

    LABELS = ["stance", "execution"]

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    #data = np.ndarray((14739), dtype=np.float32)

    # Replace this with the path to your image
    # imagePaths = sorted(list(paths.list_images(args["dataset"])))

    if image:
        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        idxs = prediction.argmax(axis=-1)

        # print("idxs",idxs)
        for (i, j) in enumerate(idxs):
            print(LABELS[j]+"_frame"+str(count))
            return LABELS[j]+"_frame"+str(count)



def classifyMainFrames():
    args = {'dataset': 'static/stage1-output-frames'}
    if os.path.exists('static/ClassifiedFrames/execution') and os.path.exists('static/ClassifiedFrames/stance'):
       
        shutil.rmtree('static/ClassifiedFrames/stance')
        os.makedirs('static/ClassifiedFrames/stance')
        shutil.rmtree('static/ClassifiedFrames/execution')
        os.makedirs('static/ClassifiedFrames/execution')
    else:
        os.makedirs('static/ClassifiedFrames/execution')
        os.makedirs('static/ClassifiedFrames/stance')
    MODE = "COCO"

    if MODE == "COCO":
        protoFile = "pose/coco/pose_deploy_linevec.prototxt"
        weightsFile = "pose/coco/pose_iter_440000.caffemodel"
        nPoints = 18
        # POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
        #               [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]
        POSE_PAIRS = [[1, 17], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12],
                      [12, 13], [1, 16]]



    elif MODE == "MPI":
        protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
        nPoints = 15
        POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10],
                      [14, 11],
                      [11, 12], [12, 13]]
    colorz = [[0, 247, 255], [255, 225, 23], [0, 171, 255], [0, 247, 255], [0, 247, 255], [0, 171, 255],
              [239, 0, 255], [239, 0, 255], [68, 255, 0], [0, 247, 255], [0, 247, 255], [68, 255, 0],
              [239, 0, 255], [239, 0, 255], [0, 247, 255], [239, 0, 255], [0, 247, 255], [239, 0, 255]]

    keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho',
                        'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip',
                        'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

    if os.path.exists('static/keypoints-detected-frames/skeleton-with-image'):
        shutil.rmtree('static/keypoints-detected-frames/skeleton-with-image')
        os.makedirs('static/keypoints-detected-frames/skeleton-with-image')
    else:
        os.makedirs('static/keypoints-detected-frames/skeleton-with-image')

    if os.path.exists('static/keypoints-detected-frames/skeleton-only'):
        shutil.rmtree('static/keypoints-detected-frames/skeleton-only')
        os.makedirs('static/keypoints-detected-frames/skeleton-only')
    else:
        os.makedirs('static/keypoints-detected-frames/skeleton-only')

    imagePaths = sorted(list(paths.list_images(args["dataset"])))
    k = 1

    stanceCount = 1
    executionCount = 1
    stanceArray = []
    executionArray = []
    classifiedData = {}
    classifiedData[
        'baseUrl'] = "static/ClassifiedFrames"
    # --------------------------------
    for imagePath in imagePaths:
        # print("Reading Image :", imagePath)
        frame = cv2.imread("static/stage1-output-frames/frame" + str(k) + ".jpg")
        frameCopy = np.copy(frame)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        threshold = 0.1

        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

        t = time.time()
        # input image dimensions for the network
        inWidth = 368
        inHeight = 368
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inpBlob)

        output = net.forward()

        height, width = 700, 500
        keypointsOnlyFrame = np.zeros((height, width, 3), dtype="uint8")
        keypointsOnlyFrame.fill(255)

        H = output.shape[2]
        W = output.shape[3]

        # Empty list to store the detected keypoints
        points = []
        pointOnlyFrame = []
        for i in range(nPoints):

            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            if i != 0 and i != 14 and i != 15:
                # Scale the point to fit on the original image
                x = (frameWidth * point[0]) / W
                y = (frameHeight * point[1]) / H
                xk = (width * point[0]) / W
                yk = (height * point[1]) / H
                if prob > threshold:
                    cv2.circle(keypointsOnlyFrame, (int(xk), int(yk)), 5, (13, 29, 181), thickness=-1,
                               lineType=cv2.FILLED)
                    cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                lineType=cv2.LINE_AA)

                    # Add the point to the list if the probability is greater than the threshold
                    points.append((int(x), int(y)))
                    pointOnlyFrame.append((int(xk), int(yk)))
                else:
                    points.append(None)
                    pointOnlyFrame.append(None)
            else:
                points.append(None)
                pointOnlyFrame.append(None)
        # Draw Skeleton
        for pair in POSE_PAIRS:

            partA = pair[0]
            partB = pair[1]
            if points[partA] and points[partB]:
                cv2.line(keypointsOnlyFrame, pointOnlyFrame[partA], pointOnlyFrame[partB], (242, 112, 16), 2)
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
                # cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        # You may need to convert the color.
        img = cv2.cvtColor(keypointsOnlyFrame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        # For reversing the operation:
        im_np = np.asarray(im_pil)
       
        classifiedStage = classifyFrames(im_pil, k)
        imageRGB = cv2.imread("static/stage1-output-frames/" + classifiedStage.split("_")[1] + '.jpg')
        if "stance" == classifiedStage.split("_")[0]:
            cv2.imwrite('static/ClassifiedFrames/' + classifiedStage.split("_")[0] + "/" + "frame" + str(stanceCount) + '.jpg',
                        imageRGB)
            stanceArray.append("frame" + str(stanceCount))
            stanceCount += 1
        if "execution" == classifiedStage.split("_")[0]:
            cv2.imwrite(
                'static/ClassifiedFrames/' + classifiedStage.split("_")[0] + "/" + "frame" + str(executionCount) + '.jpg',
                imageRGB)
            executionArray.append("frame" + str(executionCount))
            executionCount += 1
        k += 1
    #     -----------------------------

    classifiedData["stance"] = stanceArray
    classifiedData["execution"] = executionArray
    print("ary",classifiedData)
    json_data = json.dumps(classifiedData)
    return json_data