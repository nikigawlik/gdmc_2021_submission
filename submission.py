from functools import reduce
from math import cos, sin
from random import randint, random, randrange
import cv2
import numpy as np
from numpy.core.shape_base import block
from mapUtils import calcGoodHeightmap, cv2SizedWindow, visualize, listWhere
import interfaceUtils
from worldLoader import WorldSlice
from mapUtils import minecraft_woods, minecraft_colors
import pptk

# wow lets go

# step 1 - discovery

# TODO travelling agents / building spot


# for testing
w = 96
h = 96
interfaceUtils.runCommand(f"execute at @p run setbuildarea ~-{int(w/2)} 0 ~-{int(h/2)} ~{int(w/2)} 256 ~{int(h/2)}")
interfaceUtils.setBuffering(True)

# 2D build area
x1, y1, z1, x2, y2, z2 = interfaceUtils.requestBuildArea()
area = (x1, z1, x2 - x1, z2 - z1)

# step 2 - analysis

def recalcSlice():
    global worldSlice, hmTrees, hmOceanFloor, heightmap

    worldSlice = WorldSlice(area, ["MOTION_BLOCKING_NO_LEAVES", "WORLD_SURFACE", "OCEAN_FLOOR"])

    hmTrees = worldSlice.heightmaps["WORLD_SURFACE"]
    hmOceanFloor = worldSlice.heightmaps["OCEAN_FLOOR"]

    heightmap = calcGoodHeightmap(worldSlice)
    heightmap = heightmap.astype(np.uint8)

recalcSlice()

# step 3 - construction
rng = np.random.default_rng()


# ground floor test

cv2SizedWindow("hmTest", heightmap.shape)

for i in range(512):
    # nbMin = 

    cv2.imshow("hmTest", heightmap)
    cv2.waitKey(0)


# prepare ground floor

forbiddenMap =  np.zeros(heightmap.shape, dtype=np.uint8)
elevatorPos = (32,32)
cv2.rectangle(forbiddenMap, (30,30), (34,34), (1), -1)

flattenedHM = cv2.medianBlur(heightmap, 13)
difference = (flattenedHM.astype(np.int) - heightmap)

fill = np.where((difference > 0) & (difference < 6))
bridge = np.where((difference > 0) & (difference >= 6))
cut = np.where(difference < 0)
pave = np.where(forbiddenMap > 0)

cutTrees = np.where(hmTrees > heightmap)

# cut trees
for p in zip(cutTrees[0], cutTrees[1]):
    for y in range(hmTrees[p] - 1, heightmap[p] - 1, -1):
        interfaceUtils.placeBlockBatched(p[0] + area[0], y, p[1] + area[1], "air")
# fill
for p in zip(fill[0], fill[1]):
    for y in range(heightmap[p], flattenedHM[p]):
        interfaceUtils.placeBlockBatched(p[0] + area[0], y, p[1] + area[1], "dirt")
# bridge
for p in zip(bridge[0], bridge[1]):
    interfaceUtils.placeBlockBatched(p[0] + area[0], flattenedHM[p] - 1, p[1] + area[1], "light_gray_wool")
# cut
for p in zip(cut[0], cut[1]):
    for y in range(heightmap[p] - 1, flattenedHM[p] - 1, -1):
        interfaceUtils.placeBlockBatched(p[0] + area[0], y, p[1] + area[1], "air")
# pave
for p in zip(pave[0], pave[1]):
    print(f"wool at{p[0]} {p[1]}")
    interfaceUtils.placeBlockBatched(p[0] + area[0], flattenedHM[p], p[1] + area[1], "white_wool")

interfaceUtils.sendBlocks()

cmd = f"tp @p {elevatorPos[0] + area[0]} {flattenedHM[elevatorPos]+1} {elevatorPos[1] + area[1]}"
print(f"command: {cmd}")
print(interfaceUtils.runCommand(cmd))

recalcSlice()

originalHeightmap = np.array(heightmap)

heightmap += 4

# drop box algorithm
boxes = []

buildingsOnlyHeightmap = np.zeros(heightmap.shape, dtype=np.uint8)

borderMap = np.zeros(heightmap.shape, dtype=np.uint8)
cv2.rectangle(borderMap, (0, 0), (area[3]-1, area[2]-1), (1), 3)

# visualize(heightmap)

maxHeight = 150
minHeight = heightmap.min()

# block cache
# binary like array 
# 1 = blocked
shape3d = (area[2], maxHeight-minHeight, area[3])
blockCache = np.zeros(shape3d, dtype=np.uint8)

# box ids map 
# 3d map describing ground floors of boxes
# value is id of box
boxIDMap = np.zeros(shape3d, dtype=np.int)

maxBoxWidth = (area[2] - 3, area[3] - 3)

BUILD = True
COLOR = True

boxID = 0
# place n boxes
for i in range(15):
    # determine box size first
    sx = randrange(7, min(21, maxBoxWidth[0]))
    sz = randrange(7, min(21, maxBoxWidth[1]))
    sy = randint(1, 3) * 6 # 1, 2 or 3 floors high 
    
    # round heightmap down up to 2 blocks to ensure we only build in 3-high 'floors'
    heightmap = heightmap - heightmap%3

    # center offset of box
    offset = (int(sx/2), int(sz/2))
    
    # gap between neighboring buildings
    gap = randrange(1,3)

    # use the footprint of the box as a dilation kernel to find valid places to build
    strctFootprint = cv2.getStructuringElement(cv2.MORPH_RECT, (sz + 2*gap, sx + 2*gap))
    anchor = (offset[1] + gap, offset[0] + gap)

    dilatedBuildingsHeightmap = cv2.dilate(buildingsOnlyHeightmap, strctFootprint, anchor=anchor)
    dilatedBorderMap = cv2.dilate(borderMap, strctFootprint, anchor=anchor)
    dilatedHeightmap = cv2.dilate(heightmap, strctFootprint, anchor=anchor)
    dilatedForbiddenMap = cv2.dilate(forbiddenMap, strctFootprint, anchor=anchor)

    ## rank building positions by their y value
    desiredBuildPosMap = (dilatedBuildingsHeightmap == buildingsOnlyHeightmap) * (dilatedForbiddenMap == 0) * (dilatedBorderMap == 0) * (255-dilatedHeightmap)
    maxi = desiredBuildPosMap.max()
    if maxi == 0:
        raise Exception("lol no space") # TODO obv
    buildPositions = listWhere(desiredBuildPosMap == maxi)

    # get valid building positions as list
    # buildPositions = listWhere((dilatedBuildingsHeightmap == buildingsOnlyHeightmap) & (forbiddenMap == 0) & (dilatedBorderMap == 0))

    ## helper functions to build close to last placed box
    lastBoxPos = (boxes[-1][0], boxes[-1][1], boxes[-1][2]) if len(boxes) > 0 else (int(area[2]/2), 0, int(area[3]/2))
    distSquared = lambda p1: (p1[0] - lastBoxPos[0])**2 + (dilatedHeightmap[p1] - lastBoxPos[2])**2 * 100 + (p1[1] - lastBoxPos[2])**2

    if len(buildPositions) > 0:
        p = reduce(lambda a,b : a if distSquared(a) < distSquared(b) else b, buildPositions)
        # p = buildPositions[randrange(len(buildPositions))]
    else:
        # fallback for when no valid positions are found
        print("WARNING: Falling back to center position, because no valid positions were found!!!")
        # TODO maybe we should just exit the program??
        p = (int(area[2]/2), int(area[3]/2))
    
    # dx, dz is the position of the box
    dx = p[0] #- anchor[0]
    dz = p[1] #- anchor[1]

    # cx, cz is the position of the corner of the box (lowest x, lowest z)
    cx = p[0] - offset[0]
    cz = p[1] - offset[1]
    
    # y position is sampled at dx, dz
    # since we did the dilation with the right kernel, it is guranteed that we 
    # can build here without obstructions (unless we didn't find any)
    y = dilatedHeightmap[dx, dz]

    # randomly turn it into a platform (only in the lower half of the settlement)
    # if random() < .2 and y - minHeight < maxHeight/2 :
    #     y = y + sy - 1
    #     sy = 1

    # check if we ran out of vertical space
    if y + sy >= maxHeight:
        continue

    # remember box for later
    boxes.append([dx, y, dz, sx, sy, sz]) # local space! center pos

    # x,y,z are corner pos
    x = area[0] + cx
    z = area[1] + cz

    print(f"build cube at {(x, y, z)}")

    # build pillars to support the box
    for rx in range(2):
        for rz in range(2):
            xx = cx + (sx - 1) * rx
            zz = cz + (sz - 1) * rz
            yFloor = hmOceanFloor[xx, zz]
            for yy in range(yFloor, y):
                if BUILD:
                    interfaceUtils.setBlock(area[0] + xx, yy, area[1] + zz, "gray_wool")

            # update block cache
            blockCache[xx, (yFloor-minHeight):(y-minHeight), zz] = 1

    # build the box
    if BUILD:
        # interfaceUtils.buildHollowCube(x, y, z, sx, sy, sz, None if COLOR else "gray_wool")
        interfaceUtils.buildWireCube(x, y, z, sx, sy, sz, None if COLOR else "gray_wool")

    # update heightmaps
    bheight = y + sy
    heightmap[cx:cx+sx, cz:cz+sz]              = bheight
    hmOceanFloor[cx:cx+sx, cz:cz+sz]              = bheight
    buildingsOnlyHeightmap[cx:cx+sx, cz:cz+sz] = bheight

    # update block cache and id map
    dy = y - minHeight
    # mark as obstructed
    blockCache[cx:cx+sx, dy:dy+sy, cz:cz+sz] = 1
    # remember box with id
    boxIDMap[cx:cx+sx, dy:dy+sy, cz:cz+sz] = boxID    # only floor -> [cx:cx+sx, y+sy-minHeight-1, cz:cz+sz]

    boxID += 1

    # visualization

    # visualize(mhm, heightmap, buildingsOnlyHeightmap)
    # visualize(buildingsOnlyHeightmap)
    
    img = buildingsOnlyHeightmap + dilatedBuildingsHeightmap
    img[cx, cz] = 230
    img[cx+sx, cz] = 230
    img[cx, cz+sz] = 230
    img[cx+sx, cz+sz] = 230
    img[dx, dz] = 230
    img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("img", img)
    # cv2.waitKey(0 if i == 0 else 1)
    cv2.waitKey(1)

interfaceUtils.sendBlocks()

# 2nd pass -> traversability

# debug / visualization windows
cv2.namedWindow("slices", 0)
cv2.resizeWindow("slices", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)
# cv2.namedWindow("lastLayer", 0)
# cv2.resizeWindow("lastLayer", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)
cv2.namedWindow("bCache", 0)
cv2.resizeWindow("bCache", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)
cv2.namedWindow("hMap", 0)
cv2.resizeWindow("hMap", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)
# cv2.namedWindow("combined", 0)
# cv2.resizeWindow("combined", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)


dilSize = 10 # dilation size used to connect platforms
walkwayWidth = 4 # minimum width of the walkways

# create some structuring elements (kernels)
kSize = dilSize * 2 + 1
strctBigCirc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kSize,kSize))
strctCross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
# strctMedCirc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

traversable = (blockCache == 0)

# BUILD = False

pass1Heightmap = heightmap # remember old heightmap
heightmap = np.array(originalHeightmap) # and start with the original

lastLayer = np.zeros((3, area[2], area[3]), np.uint8)
lastLayerY = -1

for i in range(maxHeight - minHeight):
    # pathway maps
    if i > 0 and i < maxHeight - minHeight - 1:
        free = traversable[:,i,:] & traversable[:,i+1,:]
        blocked = ~free
        ground = ~traversable[:,i-1,:] & free
        y = minHeight + i


        # platforms
        platform0 = ground.astype(np.uint8)

        padWidth = dilSize + 1
        platform1 = np.pad(platform0, (padWidth, padWidth), mode='constant') 
        platform1 = cv2.dilate(platform1, strctBigCirc)
        platform1 = cv2.erode(platform1, strctCross, iterations=dilSize)
        platform1 = platform1[padWidth:-padWidth, padWidth:-padWidth]

        platform2 = np.where(pass1Heightmap != y, platform0, 0) 
        platform2 = cv2.dilate(platform2, strctCross, iterations=walkwayWidth)

        platform = cv2.bitwise_or(platform1, platform2)

        # retval, labels, stats, centroids = cv2.connectedComponentsWithStats(platform)
        # print(f"retval={retval}, labels={labels}, stats={stats}, centroids={centroids}")
        
        # actualHeight = y + platform
        diffToHM = np.maximum(0, y - heightmap - 1) * platform
        diffToHM = cv2.bitwise_and(diffToHM, 0b01111111) # fix underflows

        platform = (free & ~ground) * platform

        blockCache[:,i-1,:] |= platform > 0

        heightmap = np.where(blockCache[:,i-1,:] > 0, y-1, heightmap)
        heightmap = np.where(blockCache[:,i,:] > 0, y, heightmap)

        cv2.imshow("hMap", heightmap) 

        # connection
        llBlocked = lastLayer[0,:,:]
        llGround = lastLayer[1,:,:]
        llPlatform = lastLayer[2,:,:]
        combinedBlocked = cv2.bitwise_or(llBlocked, blocked.astype(np.uint8))
        combinedGround = cv2.bitwise_or(llGround, ground.astype(np.uint8))
        combinedPlatform = cv2.bitwise_or(llPlatform, platform)

        bgr = cv2.merge((combinedPlatform, combinedGround, combinedBlocked)) * 255
        # cv2.imshow("combined", bgr)

        # visualize
        r = blocked.astype(np.uint8) * 128 + blockCache[:,i,:] * 127
        g = ground.astype(np.uint8) * 255 
        b = platform.astype(np.uint8) * int(255 / walkwayWidth)
        # b = (originalHeightmap > i + minHeight).astype(np.uint8) * 255

        bgr = cv2.merge((b, g, r))

        # show boxes
        for b in filter(lambda box: box[1] == y, boxes):
            bgr[b[0], b[2], :] = (130, 250, 250)


        cv2.imshow("slices", bgr)
        llimg = cv2.cvtColor(np.transpose(lastLayer, [1,2,0]) * 255, cv2.COLOR_RGB2BGR)
        # cv2.imshow("lastLayer", llimg)
        cv2.waitKey(0 if i==1 else 1)
        # cv2.waitKey(1)

        # build layer if necessary
        if platform.max() > 0:
            blockPositions = np.where(platform > 0)
            foundation = np.where(diffToHM > 5, 0, diffToHM)
            y = minHeight + i
            for p in zip(*blockPositions):
                # y += -walkwayWidth + platform[p]
                x = area[0] + p[0]
                z = area[1] + p[1]
                for yy in range(y-foundation[p]-1, y):
                    if BUILD:
                        interfaceUtils.setBlock(x, yy, z, "gray_wool")

            lastLayerY = y # TODO kindaweird
            lastLayer[0,:,:] = blocked.astype(np.uint8)
            lastLayer[1,:,:] = ground.astype(np.uint8)
            lastLayer[2,:,:] = platform
        interfaceUtils.sendBlocks()



cv2.destroyAllWindows()
interfaceUtils.sendBlocks()

# step 4 - traversability

# step 5 - population

cv2.namedWindow("test", 0)
cv2.resizeWindow("test", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)

for i in range(maxHeight - minHeight):
    slice = blockCache[:,i,:]
    slice = slice - cv2.erode(slice, strctCross)

    cv2.imshow("test", slice * 255) 
    cv2.waitKey(0 if i==1 else 1)
    

# step 6  - decoration


interfaceUtils.sendBlocks()