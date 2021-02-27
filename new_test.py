import time
import numpy as np
from numpy.matrixlib.defmatrix import bmat
import wfc_implementation as wfci
import mapUtils
import interfaceUtils
import cv2
import random

w = 10
h = 10
cSize = 16
area = (700, -750, w * cSize, h * cSize)
yBase = 64
buildArea = interfaceUtils.requestBuildArea()
if buildArea != -1:
    x1 = buildArea["xFrom"]
    z1 = buildArea["zFrom"]
    x2 = buildArea["xTo"]
    z2 = buildArea["zTo"]
    yBase = buildArea["yFrom"]
    print("provided build area is at position %s, %s with size %s, %s, base y = %s" % (x1, z1, x2-x1, z2-z1, yBase))
    w = int((x2-x1) / cSize)
    h = int((z2-z1) / cSize)
    area = (x1, z1, w * cSize, h * cSize)
print("(adjusted) build area is at position %s, %s with size %s, %s" % area)
print("w = %s, h = %s" % (w, h))


tileW = tileH = 2
# tiles = np.array([    [[0,0],     [0,0]],    [[0,1],     [1,1]],    [[0,0],     [1,1]],    [[0,0],     [1,0]]])
tiles = np.array([[0,0,0,0],[0,1,0,0],[1,1,0,1],[1,1,0,0],[1,1,1,1]])
# tiles = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,0,0,1,1,1,0,1,1,1,0,0,0,0],[0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0]])
# tiles = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,0,0,1,1,1,0,1,1,1,0,0,0,0],[0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0],[0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,0],[0,1,1,0,1,2,2,1,1,2,2,1,0,1,1,0]])
# tiles = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,0,0,1,1,1,0,1,1,1,0,0,0,0],[0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0],[0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,0],[1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0],[1,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0],[1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0]])
tiles = tiles.reshape((tiles.shape[0], tileW, tileH))
tiles = wfci.expandRotations(tiles)

layers = 12
heightmap = np.zeros((w * tileW, h * tileH), np.int)
wfcLayers = np.zeros((layers, w * tileW, h * tileH), np.int)

cv2.namedWindow("layers", 0)
cv2.resizeWindow("layers", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)

for i in range(layers):
    r = wfci.runWFC(tiles, w, h) * (i + 1)
    r = np.transpose(r)
    heightmap = np.where(heightmap == 0, r, heightmap)
    wfcLayers[i] = r
    cv2.imshow("layers", (mapUtils.normalize(r) * 255).astype(np.uint8))
    cv2.waitKey(100)

cv2.destroyWindow("layers")

# while cv2.getWindowProperty("layers", 0) >= 0:
#     time.sleep(0.1)

heightmap = np.fmax(heightmap - 1, 0)

print("base heightmap is size %s, %s" % heightmap.shape)
# mapUtils.visualize(heightmap)
# mapUtils.visualize(heightmap, heightmap1)
# mapUtils.visualize(*wfcLayers, heightmap)

# ddc = mapUtils.normalize(mapUtils.distanceToCenter(heightmap.shape))
# ddc -= 0.71
# frc = mapUtils.normalize(mapUtils.fractalnoise(heightmap.shape, maxFreq=2))
# frc -= np.median(frc)
# heightmap *= (ddc < 0) & (frc < 0)
# heightmap *= (ddc < 0)

def revtup(x):
    return tuple(x)[::-1] # convert to tuple and reverse

heightmap = heightmap.astype(np.uint8)
heightmap = cv2.resize(heightmap, revtup(x*2 for x in heightmap.shape), interpolation=cv2.INTER_NEAREST)

print("heightmap resised to %s, %s" % heightmap.shape)
mapUtils.visualize(heightmap)

strctElmt3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
strctElmt3x3C = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

rng = np.random.default_rng()
halfSize = (int(heightmap.shape[0] / 2), int(heightmap.shape[1] / 2))
fullSize = heightmap.shape
nse = rng.random(halfSize[0] * halfSize[1], dtype = np.float64).reshape(halfSize)
nse = cv2.resize(nse, heightmap.shape[::-1], interpolation=cv2.INTER_NEAREST)

# mapUtils.visualize(nse)

dilatedhm = cv2.dilate(heightmap, strctElmt3x3C)
heightmap = np.where(nse > 0.25, heightmap, dilatedhm)

heightmap = cv2.resize(heightmap, (area[3], area[2]), interpolation=cv2.INTER_NEAREST)

nse2 = rng.random(fullSize[0] * fullSize[1], dtype = np.float64).reshape(fullSize)
nse2 = cv2.resize(nse, heightmap.shape[::-1], interpolation=cv2.INTER_NEAREST)

dilatedhm = cv2.dilate(heightmap, strctElmt3x3C)
doubledilatedhm = cv2.dilate(dilatedhm, strctElmt3x3C)
heightmap = np.where(nse2 > 0.66, heightmap, np.where(nse2 > .33, dilatedhm, doubledilatedhm))

print("heightmap resised to %s, %s" % heightmap.shape)
mapUtils.visualize(heightmap, nse2)


heightmap = mapUtils.normalize(heightmap) * 120
heightmap = heightmap.astype(np.int)
yMax = int(heightmap.max())
yHeight = yMax + 1
# buffer = np.zeros((yHeight, *heightmap))

# mapUtils.visualize(heightmap)

# wallNoise = mapUtils.fractalnoise(heightmap.shape, minFreq=3, maxFreq=3)
wallNoise = mapUtils.noise(heightmap.shape, tuple(int(x/5) for x in heightmap.shape))
wallNoiseMedian = np.median(wallNoise)
wallNoiseMask = wallNoise > wallNoiseMedian
mapUtils.visualize(wallNoiseMask)

# gradientBlocks = ["black", "gray", "light_gray", "white"]
gradientBlocks = mapUtils.minecraft_colors
def softPickBlockColor(value):
    value = value * (len(gradientBlocks) - 1.001)
    i = int(value)
    bias = value - i
    j = 0 if random.random() > bias else 1
    return gradientBlocks[i+j]


cv2.namedWindow("layers", 0)
cv2.resizeWindow("layers", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)

for dy in range(yHeight):
    walls1 = (heightmap <= dy).astype(np.uint8)
    walls1 = walls1 - cv2.erode(walls1, strctElmt3x3)

    walls2 = (heightmap <= dy+1).astype(np.uint8)
    walls2 = walls2 - cv2.erode(walls2, strctElmt3x3)

    walls = np.fmax(walls1, walls2)

    roof = (heightmap - 1 == dy).astype(np.uint8)

    floor = ((heightmap > dy) & ((heightmap-dy-1) % 4 == 0)).astype(np.uint8)
    floor = cv2.dilate(floor, strctElmt3x3)

    cv2.imshow("layers", (mapUtils.normalize(walls) * 255).astype(np.uint8))
    cv2.waitKey(1)

    for dx in range(area[2]):
        for dz in range(area[3]):
            x = area[0] + dx
            z = area[1] + dz
            y = yBase + dy

            # color = softPickBlockColor(dy / yHeight)
            # color = "light_gray"

            # block = ("%s_concrete" % color) if wallNoiseMask[dx, dz] else "light_gray_stained_glass"
            block = "smooth_stone" if wallNoiseMask[dx, dz] else "light_gray_stained_glass"

            if floor[dx, dz] > 0 and not roof[dx, dz] > 0:
                interfaceUtils.placeBlockBatched(x, y, z, "light_gray_concrete")
            elif walls[dx, dz] > 0:
                interfaceUtils.placeBlockBatched(x, y, z, block)
            elif roof[dx, dz] > 0:
                interfaceUtils.placeBlockBatched(x, y, z, "dark_prismarine_slab")



cv2.destroyWindow("layers")

interfaceUtils.sendBlocks()

