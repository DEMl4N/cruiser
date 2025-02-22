import cv2
import numpy as np


# https://github.com/murtazahassan/Learn-OpenCV-in-3-hours/blob/d4d6a14c7151aa4ebe23eb2a7cc8f94db05384c3/project2.py#L65

### ----- STEP 1 : Extract Lane Color & Remove Backgrounds ----- ###
def threshold(img):
    # Convert BGR to HSV
    imgHsv = cv2.cuda.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range of blackcolor(Lane) in HSV
    # Use ColorPicker.py for adjustment ( python ColorPicker.py )
    lowerBlack = np.array([50, 0, 25])
    upperBlack = np.array([120, 130, 100])

    # Binaryization based on lane color
    imgHsv = imgHsv.download()
    maskBlack = cv2.inRange(imgHsv, lowerBlack, upperBlack)

    return maskBlack


### ----- STEP 2 : Warping Lane Image ----- ###
def warpImg(img, points: np.float32, w: int, h: int, inverse: bool = False):
    # Fix the four vertices of the area to be transformed
    # source = np.float32(points)
    source = points
    destination = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # Get Warp Matrix
    if not inverse:
        matrix = cv2.getPerspectiveTransform(source,
                                             destination)  # Source(origin 4 pts) -> Destination(output 4 vertices)
    else:
        matrix = cv2.getPerspectiveTransform(destination,
                                             source)  # Destination(output 4 vertices) -> Source(origin 4 pts)

    # Warping with transformation matrix
    imgWarp = cv2.cuda.warpPerspective(img, matrix, (w,h))
    return imgWarp


### --- 2-1 : Test for Warping Image --- ###
def empty(a):
    pass


def initializeTrackbars(initialTrackbarVals, wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", initialTrackbarVals[0], wT // 2, empty)
    cv2.createTrackbar("Height Top", "Trackbars", initialTrackbarVals[1], hT, empty)
    cv2.createTrackbar("Width Bottom", "Trackbars", initialTrackbarVals[2], wT // 2, empty)
    cv2.createTrackbar("Height Bottom", "Trackbars", initialTrackbarVals[3], hT, empty)


def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")

    points = np.float32([(widthTop, heightTop), (wT - widthTop, heightTop), (widthBottom, heightBottom),
                         (wT - widthBottom, heightBottom)])

    return points


def drawPoints(img, points):
    for x in range(4):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return img


### ----- STEP 3 : Get Histogram ----- ###
def getHistogram(img, minPer: np.float32 = 0.5, display: bool = False, region: np.uint8 = 1):
    img = img.download()
    # ROI(Region Of Interest) = bottom of image (1/region)
    roi = int(img.shape[0] - img.shape[0] // region)
    histValues = np.sum(img[roi::], axis=0)
    # histValues = np.sum(img, axis=0) if region == 1 else np.sum(img[(img.shape[0])//region::], axis=0)

    # Use a histogram to calculate the shape of a road & basepoint(center of lane)
    maxValue = np.max(histValues)
    minValue = minPer * maxValue
    indexArray = np.where(histValues >= minValue)
    basePoint = np.int32(np.mean(indexArray))

    # Visualization for debug
    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for x, intensity in enumerate(histValues):
            cv2.line(imgHist, (x, img.shape[0]), (x, img.shape[0] - int(intensity) // 255 // region), (255, 0, 255), 1)
            cv2.circle(imgHist, (basePoint, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)
        return basePoint, imgHist

    return basePoint


### ----- STEP 4 : Smoothing Curve ----- ###
def smoothingCurve(curveList: np.ndarray, curveRaw: np.int32, maxWindow: int = 8, OUTLIER_THRESHOLD: int = 10):
    weight = np.flip(np.arange(maxWindow, dtype=int), axis=0) + 1
    avg = np.int32(np.average(curveList, weights=weight))  # Weighted average
    # Discard if curveRaw is an outlier
    if abs(curveRaw - avg) > OUTLIER_THRESHOLD:
        return avg

    # Moving Average (default window size = 5)
    curveList.append(curveRaw)
    while len(curveList) > maxWindow:
        curveList.pop(0)  # Pop old values

    avg = np.int32(np.average(curveList, weights=weight))  # Weighted average

    return avg


### ----- STEP 5 : Display ----- ###
def stackImage(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)

    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                if imgArray[x][y].shape[:2] != imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

        hor = np.empty((rows, height, width * cols, 3), np.uint8)
        # hor_con = rows * np.empty_like(imageBlank)
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)

    else:
        for x in range(rows):
            if imgArray[x].shape[:2] != imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)

        hor = np.stack(imgArray)
        ver = hor

    # print(ver.shape)
    return ver