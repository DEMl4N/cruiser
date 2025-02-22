import cv2
import numpy as np
import lane_utils as utils


# Global Variable (for static)
# curveList = np.zeros(5, dtype=np.int32)  # Empty integer array

# Get lane curve's trend
class LaneDetectionModule:
    def __init__(self):
        self.curveList = [np.int32(0)] * 8
        self.OUTLIER_THRESHOLD = 10

    def getLaneCurve(self, img, laneDiff: np.int8=0, display: int=0):
        ### STEP 1 : Find Lane & Binarization
        imgThres = utils.threshold(img)

        ### STEP 2 : Warping Lane Image
        wT, hT = img.size()

        # Determine which region to warp according to the direction
        # Option 1 : Change ROI when targetLane =/= curLane
        points = np.float32(
            [[wT * 0.3, hT * 0.3], [wT * 0.7, hT * 0.3], [wT * 0.1, hT], [wT * 0.9, hT]] if laneDiff == 0 \
                else [[wT * 0.2, hT * 0.4], [wT * 0.4, hT * 0.4], [wT * 0.3, hT], [wT * 0.6, hT]] if laneDiff > 0 \
                else [[wT * 0.6, hT * 0.4], [wT * 0.8, hT * 0.4], [wT * 0.3, hT], [wT * 0.6, hT]])
        # Option 2 : Not change ROI, but apply bias on curve
        # points = np.float32([[wT*0.3,hT*0.3],[wT*0.7,hT*0.3],[wT*0.1,hT],[wT*0.9,hT]])

        imgThresGpu = cv2.cuda_GpuMat()
        imgThresGpu.upload(imgThres)
        imgWarp = utils.warpImg(imgThresGpu, points, wT, hT)
        #imgWarpPoints = utils.drawPoints(img.download(), points)

        ### STEP 3 : Calculate Gradient of Lane(= Intensity of Curve)
        # Get histogram that accumulates pixel in a column
        if display > 1:
            """middlePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.5,
                                                      region=4)  # Center position for the current lane(bottom of image)"""
            curveAveragePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.3,
                                                            region=2)  # Average position of nearby roads
        else:
            """middlePoint = utils.getHistogram(imgWarp, minPer=0.5,
                                             region=4)  # Center position for the current lane(bottom of image)"""
            curveAveragePoint = utils.getHistogram(imgWarp, minPer=0.3, region=2)  # Average position of nearby roads

        curveRaw = curveAveragePoint - wT//2 if not np.isnan(curveAveragePoint) else 0  # Raw target curve(biased) intensity

        ### STEP 4 : Normalization & Thresholding
        # Mapping [curveMin,curveMax] -> [quantizedMin,quantizedMax] ([-160,160] => [-40,40])
        thres, curveThres = np.float32(250.), np.float32(40.)
        curve = -curveThres if curveRaw < -thres else curveThres if curveRaw > thres else (curveRaw * curveThres / thres)

        ### STEP 5 : Smoothing Curve Using LPF Filter
        curve = utils.smoothingCurve(self.curveList, curveRaw)

        ### STEP 6 : Display
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
                imgWarpPoints = utils.drawPoints(img.download(), points)
                imgStacked = utils.stackImage(0.7, ([img, imgWarpPoints, imgWarp], [imgHist, imgLaneColor, imgResult]))
                cv2.imshow('ImageStack', imgStacked)
            else:
                cv2.imshow('Result', imgResult)

        return curve