__author__ = 'gauravhirlekar'


def init_feature():
    detector = cv2.SURF(500)    # 500 is the threshold Hessian value for the detector.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)     # Or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio=0.75):
    good_matches = [m[0] for m in matches if m[0].distance <= m[1].distance * ratio]
    # Match is good only if the closest match is much closer than the second closest match. 0.75 is arbitrary ratio.

    kp_pairs = [(kp1[m.queryIdx], kp2[m.trainIdx]) for m in good_matches]
    p1 = numpy.float32([kp[0].pt for kp in kp_pairs])
    p2 = numpy.float32([kp[1].pt for kp in kp_pairs])
    return p1, p2, kp_pairs


if __name__ == '__main__':
    import cv2
    import picamera
    import numpy
    
    img1 = cv2.imread('sample.png', 0)
<<<<<<< HEAD
    detector, descriptor, matcher = init_feature()
    kp1, desc1 = descriptor.compute(img1, detector.detect(img1))

    with picamera.PiCamera() as cap:
        cap.resolution = (640, 480)

        while True:
            with picamera.array.PiRGBArray(cap) as stream:
                cap.capture(stream, format='bgr')
                # At this point the image is available as stream.array
                img2 = stream.array

            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            kp2, desc2 = descriptor.compute(img1, detector.detect(img2))
            if desc2 is None:
                # print "No descriptors found"
                continue
            raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
            # knnMatch gives k closest matched keypoints with a L2 norm distance

            p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches, 0.75)
            if len(p1) >= 8:
                H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                print cv2.perspectiveTransform(numpy.float32([w1/2, h1/2]).reshape(1, -1, 2), H).reshape(-1, 2)-numpy.float32([w1/2, h1/2])
            else:
                H, status = None, None
                print '%d matches found, not enough for homography estimation' % len(p1)

            if cv2.waitKey(1) & 0xFF == 27:  # Esc key ends loop
                break
=======
    detector, matcher = init_feature()
    kp1, desc1 = detector.detectAndCompute(img1, None)

    cap = picamera.PiCamera()
    cap.resolution = (640, 480)

    while True:
        with picamera.array.PiRGBArray(cap) as stream:
            cap.capture(stream, format='bgr')
            # At this point the image is available as stream.array
            img2 = stream.array

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        kp2, desc2 = detector.detectAndCompute(img2, None)
        if desc2 is None:
            # print "No descriptors found"
            continue
        raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
        # knnMatch gives k closest matched keypoints with a L2 norm distance

        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches, 0.7)
        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            print cv2.perspectiveTransform(np.float32([w1/2, h1/2]).reshape(1, -1, 2), H).reshape(-1, 2)-np.float32([w1/2, h1/2])
        else:
            H, status = None, None
            print '%d matches found, not enough for homography estimation' % len(p1)

        if cv2.waitKey(1) & 0xFF == 27:  # Esc key ends loop
            break

    cap.close()
    cv2.destroyAllWindows()
>>>>>>> parent of c052758... Big Changes
