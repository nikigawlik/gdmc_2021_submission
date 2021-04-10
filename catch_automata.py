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


rng = np.random.default_rng()

# space = (rng.random((16,16)) * 3).astype(np.uint8)
# space = (rng.random((area[2], area[3])) * 3).astype(np.uint8)
space = cv2.resize((rng.random((16,16)) * 3).astype(np.uint8), (area[2], area[3]), interpolation=cv2.INTER_NEAREST)

cv2.namedWindow("output", 0)

strctElmt = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

t = 0
y = 64

while True:
    space0 = (space == 0).astype(np.uint8)
    space1 = (space == 1).astype(np.uint8)
    space2 = (space == 2).astype(np.uint8)
    noise = rng.random((space.shape)) < .99
    newSpace0 = (space2 & cv2.dilate(space0, strctElmt) & noise)
    newSpace1 = (space0 & cv2.dilate(space1, strctElmt) & noise)
    newSpace2 = (space1 & cv2.dilate(space2, strctElmt) & noise)
    space0 = newSpace0 | (space0 & ~newSpace1)
    space1 = newSpace1 | (space1 & ~newSpace2)
    space2 = newSpace2 | (space2 & ~newSpace0)

    # visualize(space0, space1, space2, newSpace0, newSpace1, newSpace2)

    space = space0 * 0 + space1 * 1 + space2 * 2

    img = space * int(255/2)
    cv2.imshow("output", img)
    cv2.waitKey(0 if t == 0 else 1)
    t += 1

    # precompute a bit
    continue
    if t < 100 or t % 10 != 0:
        continue

    # place blocks
    # scaledSpace = cv2.resize(space, (area[2], area[3]), interpolation=cv2.INTER_CUBIC)
    # blockPositions = np.where(scaledSpace == 0)
    blockPositions = np.where(space == 0)
    for b in zip(*blockPositions):
        x = area[0] + b[0]
        z = area[1] + b[1]
        interfaceUtils.placeBlockBatched(x, y, z, "white_wool", 1000)

    interfaceUtils.sendBlocks()
    y += 1