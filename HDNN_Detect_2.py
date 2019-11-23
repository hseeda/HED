# USAGE
# python detect_edges_image.py --edge-detector hed_model --image images/guitar.jpg

# import the necessary packages
import argparse
import cv2
import numpy as np
import os
import easygui

# path = easygui.fileopenbox()
path = r"/home/hassan/dev/holistically-nested-edge-detection/test/sample_1.jpg"
# os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

# print(path)
hdir = os.path.dirname(path)
# print(hdir)
hfilename = os.path.basename(path)
# print(hfilename)
hname = os.path.splitext(hfilename)[0]
# print(hname)
houtname = hname + "_out.jpg"
# print(houtname)
hout = os.path.sep.join([hdir, houtname])


# print(hout)


# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--edge-detector", type=str, required=True,
# 	help="path to OpenCV's deep learning edge detector")
# ap.add_argument("-i", "--image", type=str, required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

class CropLayer(object):
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]


# load our serialized edge detector from disk
print("[INFO] loading edge detector...")

fpath = os.path.abspath(__file__)
fdir = os.path.dirname(fpath)
print(fdir)
protoPath = os.path.sep.join([fdir, "hed_model", "deploy.prototxt"])
print(protoPath)
modelPath = os.path.sep.join([fdir, "hed_model", "hed_pretrained_bsds.caffemodel"])
print(modelPath)

net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# register our new layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)

# load the input image and grab its dimensions
source_img = cv2.imread(path)

# -====================================
# cv2.imshow("source_img", source_img)
# while cv2.waitKey(1) != 13: pass
# cv2.destroyAllWindows()
# -====================================

(H, W) = source_img.shape[:2]
# image = source_img
# image = source_img[y:y+h, x:x+w]

imgOut = np.copy(source_img[:, :, 0])

print("imgOut")
print(imgOut.size)
print(imgOut.shape)
print(imgOut.dtype)

stp = 4
ws = int(W // stp)
hs = int(H // stp)

W = ws
H = hs

print(ws, " , ", hs)

for i in range(0, stp):
    for j in range(0, stp):
        ww = ws * i
        hh = hs * j

        slice1 = source_img[hh:hh + hs, ww:ww + ws]
        image = np.copy(slice1)

        # -====================================
        # cv2.imshow("image", image)


        # while cv2.waitKey(0) != 13: pass
        # cv2.destroyAllWindows()
        # -====================================
        # print("image")
        # print(image.size)
        # print(image.shape)
        # print(image.dtype)
        r = np.average(image[:, :, 2])
        g = np.average(image[:, :, 1])
        b = np.average(image[:, :, 0])
        # print("r,g,b = ", r, " , ", g, " , ", b)

        # (H, W) = image.shape[:2]
        # print("Image Width  = ", W)
        # print("Image Height = ", H)


        # convert the image to grayscale, blur it, and perform Canny
        # edge detection
        # print("[INFO] performing Canny edge detection...")
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # canny = cv2.Canny(blurred, 30, 150)

        # construct a blob out of the input image for the Holistically-Nested
        # Edge Detector

        # cc = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        # image = image+cc

        # mean = (104.00698793, 116.66876762, 122.67891434),
        # ================================================================================
        blob1 = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                                      # mean=(230, 120, 50),
                                      # mean=(104.00698793, 116.66876762, 122.67891434),
                                      mean=(b,g,r),
                                      # mean=(100, 110, 120),
                                      swapRB=False, crop=False)
        # print( blob)
        # set the blob as the input to the network and perform a forward pass
        # to compute the edges
        # print("[INFO] performing holistically-nested edge detection...")

        # print(blob1.size)
        # print(blob1.shape)
        # print(blob1.dtype)

        net.setInput(blob1)
        hed1 = net.forward()
        hed1 = cv2.resize(hed1[0, 0], (W, H))
        hed1 = (255 * hed1).astype("uint8")

        # ================================================================================
        # blob2 = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
        #                               # mean=(230, 120, 50),
        #                               # mean=(104.00698793, 116.66876762, 122.67891434),
        #                               mean=(b, g, r),
        #                               # mean=(100, 110, 120),
        #                               swapRB=True, crop=True)
        # # set the blob as the input to the network and perform a forward pass
        # # to compute the edges
        # # print("[INFO] performing holistically-nested edge detection...")
        # net.setInput(blob2)
        # hed2 = net.forward()
        # hed2 = cv2.resize(hed2[0, 0], (W, H))
        # hed2 = (255 * hed2).astype("uint8")

        # hed = (hed1 * 0.5 + hed2 * 0.5).astype('uint8')
        hed = hed1
        # print("size = ", hed.dtype)

        # show the output edge detection results for Canny and
        # Holistically-Nested Edge Detection
        # cv2.imshow("Input", image)
        # cv2.imshow("Canny", canny)
        # --------------------------------------------------
        # cv2.imshow("HED", hed)
        # --------------------------------------------------
        # cv2.imwrite(hout, hed)
        cv2.waitKey(1)
        imgOut[hh:hh + hs, ww:ww + ws] = hed
    print("Step ", i)
cv2.imwrite(hout, imgOut)
