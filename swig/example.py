import trace_skeleton
import cv2
import random

im = cv2.imread("../output/skeleton.png", 0)

_, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)

polys = trace_skeleton.from_numpy(im)

for l in polys:
    print(l)
    c = (200 * random.random(), 200 * random.random(), 200 * random.random())
    for i in range(0, len(l) - 1):

        cv2.line(im, (l[i][0], l[i][1]), (l[i + 1][0], l[i + 1][1]), c, thickness=2)

cv2.imwrite("output.png", im)
