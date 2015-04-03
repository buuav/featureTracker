__author__ = 'gauravhirlekar'

import timeit
import cv2

cam = cv2.VideoCapture(0)
num = 10
# surf = cv2.SURF(3000)
# kp1, desc1 = surf.detectAndCompute(cv2.imread('sample.png', cv2.CV_LOAD_IMAGE_GRAYSCALE), None)
img = cv2.cvtColor(cv2.resize(cam.read()[1], (640, 360)), cv2.COLOR_BGR2GRAY)
kp = cv2.SURF(3000).detect(img)
#
# flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), {})
# bf = cv2.BFMatcher(cv2.NORM_L2)
# kp2, desc2 = surf.detectAndCompute(img, None)
#
# print len(kp1), '\t', len(kp2)
# flann.knnMatch(desc1, desc2, 2)
# bf.knnMatch(desc1, desc2, 2)

# print timeit.timeit('flann.knnMatch(desc1, desc2, 2)', 'from __main__ import flann, desc1, desc2', number=1)
# print timeit.timeit('bf.knnMatch(desc1, desc2, 2)', 'from __main__ import bf, desc1, desc2', number=1)

detectors = ['SIFT', 'SURF', 'BRISK', 'ORB']
descriptors = ['SIFT', 'SURF', 'BRISK', 'FREAK', 'ORB']

for det in detectors:
    print 'Avg ms/frame for feature detection in %d frames using %s : %f' % (num, det, (1000/num)*timeit.timeit('det.detect(img)',
            'from __main__ import cv2, img\n'
            'det = cv2.FeatureDetector_create(\'%s\')' % (det, ), number=num))

for desc in descriptors:
    print 'Avg ms/frame for keypoint description in %d frames using %s : %f' % (num, desc, (1000/num)*timeit.timeit('desc.compute(img, kp)',
            'from __main__ import cv2, img\n'
            'det = cv2.FeatureDetector_create(\'SURF\')\n'
            'desc = cv2.DescriptorExtractor_create(\'%s\')\n'
            'kp = det.detect(img)' % (desc, ), number=num))

# ch = cv2.imread('chess.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
# print timeit.timeit('cv2.findChessboardCorners(img, (5,4))', 'from __main__ import cv2, img', number=10)/10
