#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np


def largestContour(contours):
    if len(contours) == 0:
        return 0
    largest = cv2.contourArea(contours[0])
    index = 0
    for i, c in enumerate(contours):
        if cv2.contourArea(c) > largest:
            largest = cv2.contourArea(c)
            index = i
    return index


# Creating dictionary for aruco pointer
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# Properties
camRgb.setPreviewSize(640, 480)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

lrcheck = False
subpixel = False

stereo.initialConfig.setConfidenceThreshold(255)
stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

# Config
topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)
spatialLocationCalculator.setWaitForConfigInput(False)
spatialLocationCalculator.initialConfig.addROI(config)


# Linking
camRgb.preview.link(xoutRgb.input)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)
spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    color = (255, 255, 255)
    while True:
        inRgb = qRgb.get()
        rgb = inRgb.getCvFrame()

        # Process image for better edge detection
        blurRgb = cv2.GaussianBlur(rgb, (7, 7), 1)
        gray = cv2.cvtColor(blurRgb, cv2.COLOR_BGR2GRAY)
        cannyRgb = cv2.Canny(blurRgb, 60, 180)

        baseHeight = 0

        # Detect aruco marker
        corners, _, _ = cv2.aruco.detectMarkers(rgb, aruco_dict, parameters=parameters)
        if corners != ():
            int_corners = np.int0(corners)
            xCenter = (int_corners[0][0][2][0] + int_corners[0][0][3][0]) // 2
            yCenter = (int_corners[0][0][1][1] + int_corners[0][0][2][1]) // 2
            topLeft.x = xCenter - 10
            topLeft.y = yCenter - 10
            bottomRight.x = xCenter + 10
            bottomRight.y = yCenter + 10

            # Set height reading from aruco pointer location
            config.roi = dai.Rect(topLeft, bottomRight)
            config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            spatialCalcConfigInQueue.send(cfg)

            inDepth = depthQueue.get()
            depthFrame = inDepth.getFrame()
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            spatialData = spatialCalcQueue.get().getSpatialLocations()
            baseHeight = spatialData[0].spatialCoordinates.z / 10
            cv2.polylines(rgb, int_corners, True, (0, 255, 0), 5)

            # Convert pixels to cm from aruco marker 5cm x 5cm
            aruco_perimeter = cv2.arcLength(corners[0], True)
            pixel_cm_ratio = aruco_perimeter / 18.8

            # Find objects on the frame
            cont, _ = cv2.findContours(cannyRgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in cont:
                if cv2.contourArea(cnt) > 20:
                    rect = cv2.minAreaRect(cnt)
                    (x, y), (w, l), angle = rect

                    # Set depth reading to object
                    topLeft.x = int(x) - 10
                    topLeft.y = int(y) - 10
                    bottomRight.x = int(x) + 10
                    bottomRight.y = int(y) + 10

                    config.roi = dai.Rect(topLeft, bottomRight)
                    config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
                    cfg = dai.SpatialLocationCalculatorConfig()
                    cfg.addROI(config)
                    spatialCalcConfigInQueue.send(cfg)

                    spatialData = spatialCalcQueue.get().getSpatialLocations()
                    objectHeight = spatialData[0].spatialCoordinates.z / 10

                    # Convert pixels to cm
                    object_width = w / pixel_cm_ratio
                    object_length = l / pixel_cm_ratio
                    object_height = abs(baseHeight - objectHeight)

                    # Calculate volume
                    volume = object_width * object_height * object_length

                    # Draw rectangle around objects and display their width and length
                    box = np.int0(cv2.boxPoints(rect))
                    cv2.polylines(rgb, [box], True, (0, 0, 255), 2)
                    cv2.putText(rgb, "Length {} cm".format(round(object_length, 1)), (int(x - 60), int(y - 15)),
                                cv2.FONT_HERSHEY_PLAIN, 1, (100, 200, 0), 2)
                    cv2.putText(rgb, "Width {} cm".format(round(object_width, 1)), (int(x - 60), int(y)),
                                cv2.FONT_HERSHEY_PLAIN, 1, (100, 200, 0), 2)
                    cv2.putText(rgb, "Height {} cm".format(round(object_height, 1)), (int(x - 60), int(y + 15)),
                                cv2.FONT_HERSHEY_PLAIN, 1, (100, 200, 0), 2)
                    cv2.putText(rgb, "Volume {} cm3".format(round(volume, 1)), (int(x - 60), int(y + 30)),
                                cv2.FONT_HERSHEY_PLAIN, 1, (100, 200, 0), 2)

        # Display image with new data on top of it
        cv2.imshow("rgb", rgb)

        if cv2.waitKey(1) == ord('q'):
            break
