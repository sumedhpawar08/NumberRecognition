__author__ = 'Montya'

import cv2
import numpy as np
import os
import operator
from Tkinter import *
from tkFileDialog import *

AMOUNT_BOX_X = 0
AMOUNT_BOX_Y = 0
AMOUNT_BOX_WIDTH = 0
AMOUNT_BOX_HEIGHT = 0

MIN_CONTOUR_AREA = 30
MAX_CONTOUR_AREA = 1000
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 20

class GUI():

    window = None
    filelabel = None
    fileChooseButton = None
    label1 = None
    resultLabeel = None
    findAmoutButton = None
    variable = None
    resultVariable = None

    def initialize(self):
        self.variable = StringVar()
        self.resultVariable = StringVar()
        self.filelabel = Label(self.window,text="Choose image file:  ").grid(row=0,sticky=W)
        self.fileChooseButton = Button(self.window,text = "Browse",command = loadImage).grid(row=0,column=1,sticky=E)
        self.label1 = Label(self.window,textvariable=self.variable).grid(row=1,column=0,columnspan=2,sticky=W)
        self.findAmoutButton = Button(self.window,text = "Submit",command = findAmount).grid(row=2,column=1,sticky=E)
        self.resultLabel = Label(self.window,textvariable = self.resultVariable).grid(row=3,column=0,columnspan=2,sticky=W)

obj = GUI()

def loadImage():

    global obj
    global TEST_IMAGE_PATH
    TEST_IMAGE_PATH = askopenfilename()
    obj.variable.set(TEST_IMAGE_PATH)
    print TEST_IMAGE_PATH
    return

def findBox():

    global AMOUNT_BOX_X
    global AMOUNT_BOX_Y
    global AMOUNT_BOX_WIDTH
    global AMOUNT_BOX_HEIGHT
    global TEST_IMAGE_PATH

    img = cv2.imread(TEST_IMAGE_PATH)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur
    #imgBlurred = cv2.blur(imgGray,(5,5))
                                                # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    #cv2.imshow("th",imgThresh)
    imgThreshCopy = imgThresh.copy()

    '''
    _,imgThresh = cv2.threshold(imgBlurred,230,255,cv2.THRESH_BINARY_INV)
    imgThreshCopy = imgThresh.copy()
    '''
    #cv2.imshow("t",imgThreshCopy)

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:

        area = cv2.contourArea(npaContour)
        #print(area)
        if(area > 13000 and area < 20000):
            #print(area)
            [intX, intY, intWidth, intHeight] = cv2.boundingRect(npaContour)

                              # thickness
            AMOUNT_BOX_X = intX + 50
            AMOUNT_BOX_Y = intY
            AMOUNT_BOX_WIDTH = intWidth - 50
            AMOUNT_BOX_HEIGHT = intHeight

            cv2.rectangle(img,           # draw rectangle on original training image
                          (AMOUNT_BOX_X, AMOUNT_BOX_Y),                 # upper left corner
                          (AMOUNT_BOX_X+AMOUNT_BOX_WIDTH,AMOUNT_BOX_Y+AMOUNT_BOX_HEIGHT),        # lower right corner
                          (0, 0, 255),                  # red
                          2)
    cv2.imshow("box",img)
    return

class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        elif self.fltArea > MAX_CONTOUR_AREA: return False
        return True

###################################################################################################



def findAmount():

    global TEST_IMAGE_PATH
    findBox()
    imgSlip = cv2.imread(TEST_IMAGE_PATH)
    imgAmountBox = imgSlip[AMOUNT_BOX_Y:AMOUNT_BOX_Y+AMOUNT_BOX_HEIGHT, AMOUNT_BOX_X:AMOUNT_BOX_X+AMOUNT_BOX_WIDTH]

    #cv2.imshow("AmountBox",imgAmountBox)

    allContoursWithData = []
    validContoursWithData = []

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print "error, unable to open classifications.txt, exiting program\n"
        os.system("pause")
        return


    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print "error, unable to open flattened_images.txt, exiting program\n"
        os.system("pause")
        return

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    height,width,_ = imgAmountBox.shape

    if imgAmountBox is None:                           # if image was not read successfully
        print "error: image not read from file \n\n"        # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)
    # end if

    imgGray = cv2.cvtColor(imgAmountBox, cv2.COLOR_BGR2GRAY)       # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur

                                                        # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    imgThreshCopy = imgThresh.copy()       # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    _, npaContours, _ = cv2.findContours(imgThreshCopy,
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
    # end for

    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list
        # end if
    # end for

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    for contourWithData in validContoursWithData:            # for each contour
                                                # draw a green rect around the current char
        cv2.rectangle(imgAmountBox,                                        # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0),              # green
                      2)                       # thickness
        #cv2.imshow("imgTestingNumbers", imgAmountBox)

        if(contourWithData.intRectHeight < 10):
            strlen = len(strFinalString)
            char = strFinalString[strlen-1:strlen-1]
            if(char == "/"):
                strFinalString = strFinalString[:strlen-1]
            continue

        #elif(contourWithData.intRectWidth < 10):
         #   pass


        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 9)     # call KNN function find_nearest

        strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results



        if(strCurrentChar == "/" and contourWithData.intRectWidth < 6):
            strCurrentChar = "1"
            pass

        if(strCurrentChar == "/" or strCurrentChar == "-"):
            continue



        strFinalString = strFinalString + strCurrentChar            # append current char to full string
    # end for

    print "\n" + strFinalString + "\n"                  # show the full string
    res = "Amount = "+strFinalString
    obj.resultVariable.set(res)
    cv2.imshow("imgAmountBox", imgAmountBox)      # show input image with green boxes drawn around found digits
    return

def main():

    win = Tk()
    win.minsize(width=200,height=100)

    global obj

    obj.window = win
    obj.initialize()

    win.mainloop()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    main()
