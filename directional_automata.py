import numpy as np
import cv2
from mapUtils import visualize
import interfaceUtils


# for testing
w = 96
h = 96
interfaceUtils.runCommand(f"execute at @p run setbuildarea ~-{int(w/2)} 0 ~-{int(h/2)} ~{int(w/2)} 256 ~{int(h/2)}")

# 2D build area
buildArea = interfaceUtils.requestBuildArea()
if buildArea != -1:
    x1 = buildArea["xFrom"]
    z1 = buildArea["zFrom"]
    x2 = buildArea["xTo"]
    z2 = buildArea["zTo"]
    # print(buildArea)
    area = (x1, z1, x2 - x1, z2 - z1)

TAU = 6.283
PI = TAU/2
rng = np.random.default_rng()

# space = (rng.random((16,16)) * 3).astype(np.uint8)
# space = cv2.resize((rng.random((16,16))), (area[2], area[3]), interpolation=cv2.INTER_NEAREST) * TAU
space = rng.random((area[2], area[3])) * TAU

cv2.namedWindow("output", 0)

strctElmt = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

t = 0
y = 64

while True:
    vector = np.transpose(np.array((np.cos(space), np.sin(space))), [1,2,0])
    # vectorBlurred = cv2.blur(vector, (7,7))
    vectorBlurred = cv2.GaussianBlur(vector, (13,13), 2)
    # vectorBlurred = cv2.MedianB(vector, (5,5), 1)
    # vectorBlurred = cv2.dilate(vector, (5,5))

    perpendicular = np.dot(vector, np.array([[0, -1], [1, 0]]))
    dotProduct = np.einsum("ijl,ijl->ij", vectorBlurred, perpendicular)
    # tesP = np.dot(vector[0,0,:], np.transpose(perpendicular[0,0,:]))
    # dP = dotProduct[0,0]
    # space += np.sign(dotProduct) * 0.01
    space += (dotProduct > 0) * 0.1
    space = space % TAU

    h = space / TAU * 360
    s = np.ones(space.shape) * .5
    v = np.ones(space.shape) * 1
    hsv = cv2.merge((h,s,v)).astype(np.float32)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("output", rgb)
    cv2.waitKey(0 if t == 0 else 1)
    # cv2.waitKey(0)
    t += 1

    # precompute a bit
    # continue
    if t < 100 or t % 4 != 0:
    # if t < 100:
        continue

    # place blocks
    # scaledSpace = cv2.resize(space, (area[2], area[3]), interpolation=cv2.INTER_CUBIC)
    # blockPositions = np.where(scaledSpace == 0)
    blockPositions = np.where(space < (TAU/6))
    for b in zip(*blockPositions):
        x = area[0] + b[0]
        z = area[1] + b[1]
        interfaceUtils.placeBlockBatched(x, y, z, "white_wool", 1000)

    interfaceUtils.sendBlocks()
    y += 1