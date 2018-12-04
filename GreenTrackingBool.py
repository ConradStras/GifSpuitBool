# import the neccesary packages
import cv2
import time
import numpy as np
import argparse
from collections import deque
import imutils
import pyximport

pyximport.install()
import WhiteLoopingBool as wl
import time
import cython

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
# comparison mask
greencompar = (20, 20, 20)
greencomparup = (64, 255, 255)

greenLower = (29, 84, 60)
greenUpper = (64, 255, 255)

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
    start_time = time.time()
    # grab the current frame
    (grabbed, frame) = camera.read()
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break
    # color space
    # rotating the video to see it properly
    frame = imutils.resize(frame, width=600)
    frame = imutils.resize(frame, height=450)
    (h, w) = frame.shape[:2]
    # centerRot = (w / 2, h / 2)
    # M = cv2.getRotationMatrix2D(centerRot, 180, 1.0)
    # frame = cv2.warpAffine(frame, M, (w, h))
    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # now to acquire the indexes of every green spot, read in order [y, x]
    reddeningarray = np.zeros((int(h * 8 / 10), int(w * 1 / 3), 3), dtype="uint8")
    reddeningarray[:, :] = (0, 0, 55)
    leftsection = mask[0:h, 0:int(w * 1 / 3)]
    midsection = mask[0:h, int(w * 1 / 3):int(w * 2 / 3)]
    rightsection = mask[0:h, int(w * 2 / 3):w]
    indexall = wl.whiteindex(mask)
    index1 = wl.whiteindex(leftsection)
    index2 = wl.whiteindex(midsection)
    index3 = wl.whiteindex(rightsection)
    (cluster1, bzone1) = wl.clusters(index1, leftsection)
    (cluster2, bzone2) = wl.clusters(index2, midsection)
    (cluster3, bzone3) = wl.clusters(index3, rightsection)
    (clusters, buseless) = wl.clusters(indexall, mask)
    for k in cluster1:
        if bzone1 == True:
            frame[int(1 / 10 * h):int(9 / 10 * h), 0:int(w * 1 / 3)] = cv2.add(
                frame[int(1 / 10 * h):int(9 / 10 * h), 0:int(w * 1 / 3)], reddeningarray)
            break
    for k in cluster2:
        if bzone2 == True:
            frame[int(1 / 10 * h):int(9 / 10 * h), int(w * 1 / 3):int(w * 2 / 3)] = cv2.add(
                frame[int(1 / 10 * h):int(9 / 10 * h), int(w * 1 / 3):int(w * 2 / 3)], reddeningarray)
            break
    for k in cluster3:
        if bzone3 == True:
            frame[int(1 / 10 * h):int(9 / 10 * h), int(w * 2 / 3):w] = cv2.add(
                frame[int(1 / 10 * h):int(9 / 10 * h), int(w * 2 / 3):w], reddeningarray)
            break
    for k in clusters:
        cv2.rectangle(frame, (k[0][1], k[0][0]), (k[1][1], k[1][0]), (255, 0, 0))
    cv2.line(frame, (0, int(h / 10)), (w, int(h / 10)), (0, 0, 255), 3)
    cv2.line(frame, (0, int(h * 9 / 10)), (w, int(h * 9 / 10)), (0, 0, 255), 3)
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, Non
    # e, iterations=2)
    cv2.imshow("Frame", frame)
    print("--- %s frames per seconds ---" % int(1 / (time.time() - start_time)))
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
