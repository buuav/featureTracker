__author__ = 'gauravhirlekar'
import multiprocessing
import time
import cv2
import numpy as np

# cam = cv2.VideoCapture(0)
# cv2.waitKey(1000)
# img = cv2.cvtColor(cam.read()[1], cv2.COLOR_BGR2GRAY)


class Consumer(multiprocessing.Process):
    def __init__(self, in_queue, out_queue):
        multiprocessing.Process.__init__(self)
        self._detector = cv2.FeatureDetector_create('SURF')
        self._descriptor = cv2.DescriptorExtractor_create('BRISK')
        self._in_queue = in_queue
        self._out_queue = out_queue

    def run(self):
        print 'Launched thread ID: %s' % (self.name, )
        while True:
            msg = self._in_queue.get()

            if msg is None:
                continue
            elif isinstance(msg, str) and msg == 'quit':
                break
            else:
                # self._out_queue.put(self._detector.detect(msg))
                self._descriptor.compute(msg, self._detector.detect(msg))
                self._in_queue.task_done()
        print 'Killing thread ID: %s' % (self.name, )


def producer():
    detector = cv2.FeatureDetector_create('SURF')
    descriptor = cv2.DescriptorExtractor_create('BRISK')

    # img = [cv2.imread(fname, 0) for fname in ['left%02d.jpg' % (i, ) for i in range(1, 15)]]
    img = cv2.imread('img.png', 0)
    subImages = []
    for splImg in np.array_split(img, 2, 0):
        subImages.extend(np.array_split(splImg, 2, 1))
    img_queue = multiprocessing.JoinableQueue()
    result_queue = multiprocessing.Queue()

    worker_threads = build_worker_pool(img_queue, result_queue, size=4)
    start_time = time.time()

    # descriptor.compute(img, detector.detect(img))
    for i in subImages:
        img_queue.put(i)

    for _ in worker_threads:
        img_queue.put('quit')
    for worker in worker_threads:
        worker.join()

    print 'Task done! Time taken : {}'.format(time.time()-start_time)
    print result_queue.get()
    return


def build_worker_pool(in_queue, out_queue, size=None):
    if size is None:
        size = multiprocessing.cpu_count()
    worker_threads = []
    for _ in range(size):
        worker = Consumer(in_queue, out_queue)
        worker.start()
        worker_threads.append(worker)
    return worker_threads

if __name__ == '__main__':
    producer()


