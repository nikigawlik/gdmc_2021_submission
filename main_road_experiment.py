from mapUtils import calcGoodHeightmap, cv2SizedWindow, visualize
from worldLoader import WorldSlice
import interfaceUtils
import numpy as np
import cv2

# for testing
w = 128
h = 128
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


ws = WorldSlice(area, ["MOTION_BLOCKING_NO_LEAVES", "OCEAN_FLOOR", "WORLD_SURFACE"])

heightmap = calcGoodHeightmap(ws).astype(np.uint8)

hmTrees = ws.heightmaps["WORLD_SURFACE"]
heightmapOcean = ws.heightmaps["OCEAN_FLOOR"].astype(np.uint8)

borderMap = np.ones(heightmap.shape, dtype=np.uint8)
borderMap[1:-1,1:-1] = 0

landmapBorder = ((heightmapOcean > 62) & (borderMap == 0)).astype(np.uint8)
landmap = ((heightmapOcean > 62)).astype(np.uint8)

# visualize(landmapBorder)

dst, labels = cv2.distanceTransformWithLabels(landmapBorder, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
dst2, labels2 = cv2.distanceTransformWithLabels(landmap, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)

idToPos = np.zeros((labels.max()+1, 2), np.int)
xy = np.where(landmapBorder==0)

for p in zip(xy[0], xy[1]):
    idToPos[labels[p],:] = p

ps = idToPos[labels]
baseHeight = heightmap[ps[:,:,0], ps[:,:,1]]
baseHeight = np.maximum(baseHeight, 63)

adjDist = (dst / 3).astype(np.uint8) 
adjDist2 = (dst2 / 3).astype(np.uint8) 
desiredHeight = adjDist2 + 63

newHeightmap = np.clip(desiredHeight, baseHeight - adjDist, baseHeight + adjDist)

difference = (newHeightmap.astype(np.int) - heightmap)

cutTrees = np.where(hmTrees > heightmap)

# cv2.imshow("nhm", newHeightmap)
# cv2.waitKey(0)

# cut trees
for p in zip(cutTrees[0], cutTrees[1]):
    for y in range(hmTrees[p] - 1, heightmap[p] - 1, -1):
        interfaceUtils.placeBlockBatched(p[0] + area[0], y, p[1] + area[1], "air")

fill = np.where((difference > 0))
cut = np.where(difference < 0)

for p in zip(fill[0], fill[1]):
    block1 = ws.getBlockAt((area[0] +p[0], heightmap[p]-2, area[1] +p[1]))
    block2 = ws.getBlockAt((area[0] +p[0], heightmap[p]-1, area[1] +p[1]))
    for y in range(heightmap[p]-1, newHeightmap[p]-1):
        interfaceUtils.placeBlockBatched(area[0] + p[0], y, area[1] + p[1], block1)
    interfaceUtils.placeBlockBatched(area[0] + p[0], newHeightmap[p]-1, area[1] + p[1], block2)

for p in zip(cut[0], cut[1]):
    for y in range(heightmap[p]-1, newHeightmap[p]-1, -1):
        interfaceUtils.placeBlockBatched(area[0] + p[0], y, area[1] + p[1], "air")


# visualize(distFromWater)
# gradient = np.stack([cv2.Scharr(distFromWater, cv2.CV_16S, 1, 0), cv2.Scharr(distFromWater, cv2.CV_16S, 0, 1)])
# gradient = np.linalg.norm(gradient, axis=0)
# gradient = np.abs(cv2.Scharr(distFromWater, cv2.CV_16S, 1, 0)) + np.abs(cv2.Scharr(distFromWater, cv2.CV_16S, 0, 1))

# cv2SizedWindow("img", heightmap.shape)

# for i in range(gradient.max() + 1):
#     pixelmap = (gradient <= i) & (landmap > 0)
#     cv2.imshow("img", pixelmap.astype(np.uint8) * 255)
#     cv2.waitKey(0)

# cv2.destroyWindow("img")

# cv2.watershed()

cv2SizedWindow("img1", heightmap.shape)
cv2SizedWindow("img2", heightmap.shape)

cv2.imshow("img1", landmapBorder * 255)
# cv2.imshow("img2", distFromWater)
cv2.waitKey(0)


# space = np.zeros((area[2]/16, area[3]/16))
# for i in range(100):
#     bgr = space.astype(np.uint8) * 255
#     cv2.imshow("img", bgr)
#     cv2.waitKey(1)
