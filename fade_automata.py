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
smol = np.floor(rng.random((5,5)) * 3) / 2
space = cv2.resize(smol, (area[2], area[3]), interpolation=cv2.INTER_NEAREST)
# space = rng.random((area[2], area[3])) 
# space = np.round(space)
# space = np.clip(space, 0, 1)
# space = (rng.random((area[2], area[3])) > 0.999).astype(np.float64)
# space = np.zeros((area[2], area[3]))
# space[45,45] = 1
# space[44,35:55] = .9

cv2.namedWindow("output", 0)

strctElmt = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

t = 0
y = 64

while True:
    isOne = (space == 1).astype(np.uint8)
    isZero = (space == 0).astype(np.uint8)
    newShit = isZero & cv2.dilate(isOne, strctElmt)
    # newShit = isZero & cv2.dilate(isOne, strctElmt) & (rng.random(space.shape) < .8)
    space -= rng.random(space.shape) * 0.1
    space += newShit * 4
    # visualize(space, newShit, isOne, isZero)

    # space += (rng.random((area[2], area[3])) > 0.999).astype(np.float64)
    space = np.clip(space, 0, 1)
    

    v = space
    s = np.ones(space.shape) * 0
    h = np.ones(space.shape) * 0
    hsv = cv2.merge((h,s,v)).astype(np.float32)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("output", rgb)
    cv2.waitKey(0 if t == 0 else 1)
    # cv2.waitKey(0)
    t += 1

    # precompute a bit
    # continue
    # if t < 100 or t % 4 != 0:
    if t < 100:
        continue

    # place blocks
    # scaledSpace = cv2.resize(space, (area[2], area[3]), interpolation=cv2.INTER_CUBIC)
    # blockPositions = np.where(scaledSpace == 0)
    blockPositions = np.where(space == 1)
    for b in zip(*blockPositions):
        x = area[0] + b[0]
        z = area[1] + b[1]
        interfaceUtils.placeBlockBatched(x, y, z, "white_wool", 1000)

    interfaceUtils.sendBlocks()
    y += 1