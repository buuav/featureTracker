'''




'''

__author__ = 'gauravhirlekar'


def init_feature():
    # detector = cv2.FeatureDetector_create('SURF')
    detector = cv2.SURF(2000)
    descriptor = cv2.DescriptorExtractor_create('BRISK')
    matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')

    # detector = cv2.SURF(3000)    # Argument is the threshold Hessian value for the detector.
    # matcher = cv2.BFMatcher(cv2.NORM_L2)

    # detector = cv2.SURF(3000)
    # descriptor = cv2.BRISK()
    # detector = cv2.ORB()

    # detector = cv2.BRISK()
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    return detector, descriptor, matcher
    # return detector, matcher



def explore_match(win, img1, img2, kp_pairs, status=None, H=None):
    global start_time
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if len(kp_pairs) is 0:
        cv2.imshow(win, vis)
        return

    if H is not None and len(status) > 10:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (0, 0, 255), thickness=2)
        # print 'Frame rate : {0}\t Center point: {1}'.format(*(1/time.time()-start_time, cv2.perspectiveTransform(
        #     np.float32([w1/2, h1/2]).reshape(1, -1, 2), H).reshape(-1, 2)-np.float32([w2/2, h2/2])))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)

    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
            cv2.line(vis, (x1, y1), (x2, y2), green)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)

    cv2.imshow(win, vis)
    return vis


def filter_matches(kp1, kp2, matches, ratio=0.75):
    good_matches = [m[0] for m in matches if m[0].distance <= m[1].distance * ratio]

    # Match is good only if the closest match is much closer than the second closest match. 0.75 is an arbitrary ratio.

    kp_pairs = [(kp1[m.queryIdx], kp2[m.trainIdx]) for m in good_matches]
    p1 = np.float32([kp[0].pt for kp in kp_pairs])
    p2 = np.float32([kp[1].pt for kp in kp_pairs])
    # p1 = np.int8([[kp[0].pt for kp in kp_pairs]])
    # p2 = np.int8([[kp[1].pt for kp in kp_pairs]])
    return p1, p2, kp_pairs


if __name__ == '__main__':
    import cv2
    import numpy as np
    import time

    winName = 'Detector'
    img1 = cv2.imread('sample.png', 0)

    detector, descriptor, matcher = init_feature()
    kp1, desc1 = descriptor.compute(img1, detector.detect(img1))
    # detector, matcher = init_feature()
    # kp1, desc1 = detector.detectAndCompute(img1, None)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow(winName)

    while True:
        start_time = time.time()
        s, img2 = cap.read()
        img2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (640, 360))

        kp2, desc2 = descriptor.compute(img2, detector.detect(img2))
        # kp2, desc2 = detector.detectAndCompute(img2, None)

        if desc2 is None:
            # print "No descriptors found"
            continue
        raw_matches = matcher.knnMatch(desc1, desc2, k=2)
        # knnMatch gives k closest matched keypoints

        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches, 0.75)
        if len(p1) >= 8:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            print 'Frame rate : {0}'.format(*(1/(time.time()-start_time), ))
            # print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        else:
            H, status = None, None
            print 'Frame rate : {0} \t {1} matches found, not enough for homography estimation'.format(*(1/(time.time()-start_time), len(p1)))

        explore_match(winName, img1, img2, kp_pairs, status, H)

        if cv2.waitKey(1) & 0xFF == 27:  # Esc key ends loop
            break

    cap.release()
    cv2.destroyAllWindows()