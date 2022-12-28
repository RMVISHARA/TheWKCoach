def devideFrames(video):
    import cv2
    import os, shutil
    import argparse

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-v", "--video", required=True,
    #                 help="name of the video")
    # args = vars(ap.parse_args())
    PATH = "static/upload_folder/" + video
    # print(cv2.__version__)
    # print("Importing the video...")
    vidcap = cv2.VideoCapture(PATH);
    success, image = vidcap.read()
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    totalFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT);
    duration = float(totalFrames) / float(fps)
    # print("FPS ", fps , "duration ",duration)
    path = 'static/stage1-output-frames'
    # print("Creating the directory to save frames...")
    typeOfApplication = 'test'
    count = 0
    z = 1
    savedFrames = 0
    if 1 < duration:
        if typeOfApplication != 'train':
            if os.path.exists('static/stage1-output-frames'):
                shutil.rmtree('static/stage1-output-frames')
                os.makedirs('static/stage1-output-frames')
            else:
                os.makedirs('static/stage1-output-frames')
            # count = 0
            while success:
                if count % 5 == 0:
                    savedFrames += 1
                    cv2.imwrite(os.path.join(path, "frame%d.jpg" % savedFrames), image)  # save frame as JPEG file

                success, image = vidcap.read()
                # print('Read a new frame: ', success)
                count += 1
            # print("Frames saved Successfully ", savedFrames, "frames")

        else:
            if os.path.exists('static/train-output-stage-images'):
                shutil.rmtree('static/train-output-stage-images')
                os.makedirs('static/train-output-stage-images')
            else:
                os.makedirs('static/train-output-stage-images')
            # count = 0
            while success:
                cv2.imwrite(os.path.join('static/train-output-stage-images', "frame%d.jpg" % count),
                            image)  # save frame as JPEG file
                success, image = vidcap.read()
                # print('Read a new frame: ', success)
                count += 1


            # print("Frames saved Successfully ", count - 1, "frames")

        # print("Total Duration of the video ", duration, "s", " fps : ", fps, " no of frames captured : ", totalFrames ,
        #       "but saved frames ",savedFrames)

        # print("Please Upload a video containing stance,leg-movement,shot execution")
