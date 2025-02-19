import cv2
import numpy as np
import lane_utils as utils


# Global Variable (for static)
# curveList = np.zeros(5, dtype=np.int32)  # Empty integer array

# Get lane curve's trend
class LaneDetectionModule:
    def __init__(self):
        self.curveList = [np.int32(0)] * 5

    def getLaneCurve(self, img, curLane: np.uint8, targetLane: np.uint8, display: int = 0):
        imgCopy = img.copy()
        imgResult = img.copy()

        ### STEP 1 : Find Lane & Binarization
        imgThres = utils.threshold(img)

        ### STEP 2 : Warping Lane Image
        hT, wT, c = img.shape

        # Determine which region to warp according to the direction
        # Option 1 : Use 1 ROI & move to end of lane when targetLane =/= curLane
        points = np.float32(
            [[wT * 0.25, hT * 0.5], [wT * 0.75, hT * 0.5], [wT * 0.2, hT], [wT * 0.8, hT]] if targetLane == curLane \
                else [[wT * 0.4, hT * 0.5], [wT * 0.8, hT * 0.5], [wT * 0.375, hT],
                      [wT * 0.875, hT]] if targetLane > curLane \
                else [[wT * 0.2, hT * 0.5], [wT * 0.6, hT * 0.5], [wT * 0.125, hT], [wT * 0.625, hT]])

        imgWarp = utils.warpImg(imgThres, points, wT, hT)
        imgWarpPoints = utils.drawPoints(imgCopy, points)

        ### STEP 3 : Calculate Gradient of Lane(= Intensity of Curve)
        # Get histogram that accumulates pixel in a column
        if display > 1:
            middlePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.4,
                                                      region=2)  # Center position for the current lane(bottom of image)
            curveAveragePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.2,
                                                            region=1)  # Average position of nearby roads
        else:
            middlePoint = utils.getHistogram(imgWarp, minPer=0.4,
                                             region=2)  # Center position for the current lane(bottom of image)
            curveAveragePoint = utils.getHistogram(imgWarp, minPer=0.2, region=1)  # Average position of nearby roads
        curveRaw = curveAveragePoint - middlePoint if not (
                    np.isnan(middlePoint) or np.isnan(curveAveragePoint)) else 0  # Raw target curve(biased) intensity

        ### STEP 4 : Smoothing Curve Using LPF Filter
        curve = utils.smoothingCurve(self.curveList, curveRaw)

        ### STEP 5 : Display
        if display > 0:
            imgInvWarp = utils.warpImg(imgWarp, points, wT, hT, inverse=True)
            # imgInvWarp[0 : hT//3, 0 : wT] = 0   # Masking the top of inv Image
            imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)

            imgLaneColor = np.full_like(img, (0, 255, 0))
            imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)

            imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
            midY = 450

            cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0))
            cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
            cv2.line(imgResult, (wT // 2 + (curve * 3), midY - 25), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)

            for x in range(-30, 30):
                w = wT // 20
                cv2.line(imgResult, (w * x + curve // 50, midY - 10), (w * x + curve // 50, midY + 10), (0, 0, 255), 2)

            if display > 1:
                imgStacked = utils.stackImage(0.7, ([img, imgWarpPoints, imgWarp], [imgHist, imgLaneColor, imgResult]))
                cv2.imshow('ImageStack', imgStacked)
            else:
                cv2.imshow('Result', imgResult)

        ### STEP 6 : Normalization & Thresholding
        # Mapping [curveMin,curveMax] -> [quantizedMin,quantizedMax] ([-250,250] => [-40,40])
        thres, curve_thres = np.float32(250.), np.float32(40.)
        curve = -curve_thres if curve < -thres else thres if curve > thres else (curve * curve_thres / thres)

        return curve