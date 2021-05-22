from blockRegistry import sendPattern
import blockRegistry
from functools import reduce
from math import cos, sin
from random import randint, random, randrange
import cv2
import numpy as np
from numpy.core.shape_base import block
from mapUtils import calcGoodHeightmap, cv2SizedWindow, fractalnoise, noise, normalize, visualize, listWhere
import interfaceUtils
from worldLoader import WorldSlice
from mapUtils import minecraft_woods, minecraft_colors
import pptk

# wow lets go

# step 1 - discovery

# TODO travelling agents / building spot


# for testing
w = 128
h = 128
interfaceUtils.runCommand(f"execute at @p run setbuildarea ~-{int(w/2)} 0 ~-{int(h/2)} ~{int(w/2)} 256 ~{int(h/2)}")
interfaceUtils.setBuffering(True)

# 2D build area
x1, y1, z1, x2, y2, z2 = interfaceUtils.requestBuildArea()
area = (x1, z1, x2 - x1, z2 - z1)

# step 2 - analysis

def recalcSlice():
    global worldSlice, hmTrees, hmOceanFloor, heightmap

    worldSlice = WorldSlice(x1, z1, x2, z2, ["MOTION_BLOCKING_NO_LEAVES", "WORLD_SURFACE", "OCEAN_FLOOR"])

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
    cv2.waitKey(1)


## Ground Prep ##

forbiddenMap =  np.zeros(heightmap.shape, dtype=np.uint8)
elevatorPos = (32,32)
cv2.rectangle(forbiddenMap, (30,30), (34,34), (1), -1)

flattenedHM = cv2.medianBlur(heightmap, 13)
difference = (flattenedHM.astype(np.int) - heightmap)

fill = np.where((difference > 0) & (difference < 6))
bridge = np.where((difference > 0) & (difference >= 6))
cut = np.where(difference < 0)
pave = np.where(forbiddenMap > 0)

TERRAFORM = False

if TERRAFORM:
    cutTrees = np.where(hmTrees > heightmap)

    # cut trees
    for p in zip(cutTrees[0], cutTrees[1]):
        for y in range(hmTrees[p] - 1, heightmap[p] - 1, -1):
            interfaceUtils.setBlock(p[0] + area[0], y, p[1] + area[1], "air")
    # fill
    for p in zip(fill[0], fill[1]):
        for y in range(heightmap[p], flattenedHM[p]):
            interfaceUtils.setBlock(p[0] + area[0], y, p[1] + area[1], "dirt")
    # bridge
    for p in zip(bridge[0], bridge[1]):
        interfaceUtils.setBlock(p[0] + area[0], flattenedHM[p] - 1, p[1] + area[1], "light_gray_wool")
    # cut
    for p in zip(cut[0], cut[1]):
        for y in range(heightmap[p] - 1, flattenedHM[p] - 1, -1):
            interfaceUtils.setBlock(p[0] + area[0], y, p[1] + area[1], "air")
    # pave
    for p in zip(pave[0], pave[1]):
        print(f"wool at{p[0]} {p[1]}")
        interfaceUtils.setBlock(p[0] + area[0], flattenedHM[p], p[1] + area[1], "white_wool")

    interfaceUtils.sendBlocks()

    ## player teleport
    # cmd = f"tp @p {elevatorPos[0] + area[0]} {flattenedHM[elevatorPos]+1} {elevatorPos[1] + area[1]}"
    # print(f"command: {cmd}")
    # print(interfaceUtils.runCommand(cmd))

recalcSlice()

originalHeightmap = np.array(heightmap)

# heightmap += 4

## Drop Boxes Algorithm ##
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
shape2d = (area[2], area[3])
shape3d = (area[2], maxHeight-minHeight, area[3])
blockCache = np.zeros(shape3d, dtype=np.uint8)

# box ids map 
# 3d map describing ground floors of boxes
# value is id of box
boxIDMap = np.zeros(shape3d, dtype=np.int)

maxBoxWidth = (area[2] - 3, area[3] - 3)

# boxPlacedMap = np.zeros(heightmap.shape)

BUILD = True
COLOR = False

boxID = 0
# place n boxes
for i in range(200):
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

    # rank building positions by their y value
    desiredBuildPosMap = (dilatedBuildingsHeightmap == buildingsOnlyHeightmap) * (dilatedForbiddenMap == 0) * (dilatedBorderMap == 0) * (255-dilatedHeightmap)
    maxi = desiredBuildPosMap.max()
    if maxi == 0:
        # raise Exception("lol no space") # TODO obv
        print("lol no space")
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

    cv2.rectangle(forbiddenMap, (cx, cz), (cx + sx, cz + sz), 1)
    
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
    boxes.append([dx, y, dz, sx, sy, sz, cx, cz]) # local space! center pos

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
        col = None if COLOR else "gray_wool"
        # interfaceUtils.buildHollowCube(x, y, z, sx, sy, sz, None if COLOR else "gray_wool")
        interfaceUtils.buildWireCube(x, y, z, sx, sy, sz, col)
        interfaceUtils.buildHollowCube(x, y+sy-1, z, sx, 1, sz, col)
        interfaceUtils.buildHollowCube(x, y, z, sx, 1, sz, col)

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

# interfaceUtils.setBuffering(False)

## Post-Process Boxes ##
for box in boxes:
    dx, y, dz, sx, sy, sz, cx, cz = tuple(box)

    if random() > 0.5:
        doorX = randrange(1, sx-1)
        doorZ = randrange(0, 2) * (sz - 1)
    else:
        doorX = randrange(0, 2) * (sx - 1)
        doorZ = randrange(1, sz-1)

    if BUILD:
        interfaceUtils.setBlock(area[0] + cx + doorX, y+1, area[1] + cz + doorZ, "acacia_door[half=lower]")
        interfaceUtils.setBlock(area[0] + cx + doorX, y+2, area[1] + cz + doorZ, "acacia_door[half=upper]")



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

## Slices Bottom -> Top ##

dilSize = 10 # dilation size used to connect platforms
walkwayWidth = 4 # minimum width of the walkways

# create some structuring elements (kernels)
kSize = dilSize * 2 + 1
strctBigCirc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kSize,kSize))
strctCross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
strctRect = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# strctMedCirc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

traversable = (blockCache == 0)

# BUILD = False

pass1Heightmap = heightmap # remember old heightmap
heightmap = np.array(originalHeightmap) # and start with the original

lastLayer = np.zeros((3, area[2], area[3]), np.uint8)
lastLayerY = -1

elevators = np.zeros(originalHeightmap.shape)
platformCache = np.zeros(shape3d, dtype=np.uint8)

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

        floor = cv2.bitwise_or(platform1, platform2)

        # TODO
        # get rid of 1-wide artifacts
        # platform = cv2.filter2D(...)

        # TODO visualize

        platformCache[:,i,:] = floor[:,:]

        # calc space under platform
        diffToHM = np.maximum(0, y - heightmap - 1) * floor
        diffToHM = cv2.bitwise_and(diffToHM, 0b01111111) # fix underflows

        # prep platform for build and update blockCache
        floor = (free & ~ground) * floor
        blockCache[:,i-1,:] |= floor > 0

        # update hm according to last 2 layers of heightmap
        heightmap = np.where(blockCache[:,i-1,:] > 0, y-1, heightmap)
        heightmap = np.where(blockCache[:,i,:] > 0, y, heightmap)

        cv2.imshow("hMap", heightmap) 

        # # connection
        # llBlocked = lastLayer[0,:,:]
        # llGround = lastLayer[1,:,:]
        # llPlatform = lastLayer[2,:,:]
        # combinedBlocked = cv2.bitwise_or(llBlocked, blocked.astype(np.uint8))
        # combinedGround = cv2.bitwise_or(llGround, ground.astype(np.uint8))
        # combinedPlatform = cv2.bitwise_or(llPlatform, platform)

        # bgr = cv2.merge((combinedPlatform, combinedGround, combinedBlocked)) * 255
        # # cv2.imshow("combined", bgr)

        # visualize
        r = blocked.astype(np.uint8) * 128 + blockCache[:,i,:] * 127
        g = ground.astype(np.uint8) * 255 
        b = floor.astype(np.uint8) * int(255 / walkwayWidth)
        # b = (originalHeightmap > i + minHeight).astype(np.uint8) * 255

        bgr = cv2.merge((b, g, r))

        # show boxes
        for b in filter(lambda box: box[1] == y, boxes):
            bgr[b[0], b[2], :] = (130, 250, 250)


        cv2.imshow("slices", bgr)
        llimg = cv2.cvtColor(np.transpose(lastLayer, [1,2,0]) * 255, cv2.COLOR_RGB2BGR)
        # cv2.imshow("lastLayer", llimg)
        # cv2.waitKey(0 if i==1 else 1)
        # cv2.waitKey(0)
        cv2.waitKey(1)

        # build layer if necessary
        if floor.max() > 0:
            wallPositions = np.where(floor > 0)
            foundation = np.where(diffToHM > 5, 0, diffToHM)
            y = minHeight + i
            for p in zip(*wallPositions):
                # y += -walkwayWidth + platform[p]
                x = area[0] + p[0]
                z = area[1] + p[1]
                for yy in range(y-foundation[p]-1, y):
                    if BUILD:
                        interfaceUtils.setBlock(x, yy, z, "gray_wool")

            lastLayerY = y # TODO kindaweird
            lastLayer[0,:,:] = blocked.astype(np.uint8)
            lastLayer[1,:,:] = ground.astype(np.uint8)
            lastLayer[2,:,:] = floor
        interfaceUtils.sendBlocks()

cv2.destroyAllWindows()

cv2.namedWindow("elevators", 0)
cv2.resizeWindow("elevators", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)
cv2.namedWindow("outline", 0)
cv2.resizeWindow("outline", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)

## Slices Top -> Bottom ##

for i in range(maxHeight - minHeight - 1, -1, -1):
    y = minHeight + i
    floor = platformCache[:,i,:]
    platformCount, labels, stats, centroids = cv2.connectedComponentsWithStats(floor, connectivity=4)
    # print(f"platformCount={platformCount}, labels={labels}, stats={stats}, centroids={centroids}")
    
    elevatorShape = (5,5) # if random() > .5 else (3,5)
    strctElmt = cv2.getStructuringElement(cv2.MORPH_RECT, elevatorShape)

    overlaps = np.where(elevators > 0, labels, 0)
    labels = labels.astype(np.uint8)
    labels = cv2.dilate(labels, strctElmt)
    outline = cv2.dilate(labels, strctCross) - labels
    
    cv2.imshow("elevators", (elevators * 255).astype(np.uint8))
    cv2.imshow("outline", (normalize(outline) * 255).astype(np.uint8))
    cv2.waitKey(1)

    for j in range(platformCount):
        if not j in overlaps:
            buildPositions = listWhere(outline == j) ## TODO might be empty (otherwhere too)
            if len(buildPositions) == 0:
                buildPositions = listWhere(labels == j)
            pos = buildPositions[randrange(len(buildPositions))]
            pos = (pos[1], pos[0])

            print(f"start elevator at {pos[0] + area[0]}, {pos[1] + area[1]}")
            elevators = cv2.rectangle(elevators, tuple(pos), tuple(pos), 1, -1)

    elevatorsBuildShape = elevators * (originalHeightmap < y-1)
    elevatorsBuildShapeFloor = elevators * (originalHeightmap == y-1)
    elevatorsBuildShapeDilated = cv2.dilate(elevatorsBuildShape, strctElmt)
    elevatorsBuildOutline = elevatorsBuildShapeDilated - cv2.erode(elevatorsBuildShapeDilated, strctCross)
    platformDil = cv2.dilate(floor, strctCross)
    elevatorsBuildOutline = np.where(platformDil > 0, 0, elevatorsBuildOutline)

    blockCache[:,i,:] = np.where(elevatorsBuildShapeDilated > 0, 2, blockCache[:,i,:])
            
    # build elevators
    # wallPositions = np.where(elevatorsBuildOutline > 0)
    # for p in zip(*wallPositions):
    #     x = area[0] + p[0]
    #     z = area[1] + p[1]
    #     if BUILD:
    #         interfaceUtils.setBlock(x, y, z, "gray_wool")
    # interfaceUtils.sendBlocks()

    centerPositions = np.where(elevatorsBuildShape > 0)
    for p in zip(*centerPositions):
        # if y % 5 == 0:
        #     x = area[0] + p[0] - 2
        #     z = area[1] + p[1] - 2
        #     if BUILD:
        #         sendPattern(blockRegistry.patternElevatorLight, x, y, z)
        # else:
        x = area[0] + p[0] - 1
        z = area[1] + p[1] - 1
        if BUILD:
            sendPattern(blockRegistry.patternElevator, x, y, z)
    
    centerPositions = np.where(elevatorsBuildShapeFloor > 0)
    for p in zip(*centerPositions):
        x = area[0] + p[0] - 1
        z = area[1] + p[1] - 1
        if BUILD:
            sendPattern(blockRegistry.patternElevatorFloor, x, y, z)

    cv2.waitKey(1)
        

cv2.destroyAllWindows()
interfaceUtils.sendBlocks()


# cache test
# print("Doing cache test, will nuke everything")
# for i in range(shape3d[1]):
#     for j in range(shape3d[0]):
#         for k in range(shape3d[2]):
#             numID = blockCache[(j,i,k)]
#             x = area[0] + j
#             z = area[1] + k
#             y = minHeight + i
#             if numID == 0:
#                 interfaceUtils.setBlock(x, y, z, "air")
#             elif numID == 1:
#                 interfaceUtils.setBlock(x, y, z, "white_wool")
#             elif numID == 2: 
#                 interfaceUtils.setBlock(x, y, z, "light_gray_wool")
#             else:
#                 interfaceUtils.setBlock(x, y, z, "red_wool")


interfaceUtils.sendBlocks()


cv2.namedWindow("railings", 0)
cv2.resizeWindow("railings", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)
cv2.namedWindow("bCache", 0)
cv2.resizeWindow("bCache", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)

## Pseudo-CA Bottom -> Top
# labels:
lGround = 1
lTopsoil = 2
# lPaved = 3
lPlatform = 4
lIndoorFeet = 5
lOutdoorFeet = 6
lIndoorHead = 7
lOutdoorHead = 8
lIndoorAir = 9
lIndoorFloor = 10
lRailingFloor = 11
lRailingFeet = 12
lRailingHead = 13
lWall = 14

def buildByCondition(condition, y, blockID):
    buildPositions = np.where(condition)
    for p in zip(*buildPositions):
        x = area[0] + p[0]
        z = area[1] + p[1]
        if BUILD:
            interfaceUtils.setBlock(x, y, z, blockID)
    interfaceUtils.sendBlocks()

layer = np.ones(shape2d, dtype=np.uint8) * 0

for i in range(0, maxHeight - minHeight - 1):
    y = minHeight + i

    prevLayer = layer
    bCache = blockCache[:,i,:]
    bCachePlus1 = blockCache[:,i+1,:]

    bCachePlus4 = blockCache[:,i+4,:] if i+4 < maxHeight - minHeight else np.ones(shape2d, dtype=np.uint8) * 0

    layer = np.zeros(shape2d)

    layer = np.where((bCache == 1) & (bCachePlus1 == 0), lPlatform, layer)
    
    layer = np.where(y < originalHeightmap - 1, lGround, layer)
    layer = np.where(y == originalHeightmap - 1, lTopsoil, layer)

    layer = np.where(prevLayer == lPlatform, lOutdoorFeet, layer)
    layer = np.where(prevLayer == lOutdoorFeet, lOutdoorHead, layer)

    # layer = np.where(np.isin(prevLayer, [0, lIndoorAir]) & (bCache == 1) & (bCachePlus1 == 1) & (bCachePlus4 == 1), lIndoorFloor, layer)
    # layer = np.where(prevLayer == lIndoorFloor, lIndoorFeet, layer)
    # layer = np.where(prevLayer == lIndoorFeet, lIndoorHead, layer)
    # layer = np.where(prevLayer == lIndoorHead, lIndoorAir, layer)

    # floor = np.isin(layer, [lIndoorFloor, lPlatform]).astype(np.uint8)
    floor = (bCache == 1).astype(np.uint8)
    railing = floor - cv2.erode(floor, strctRect)
    railing = np.where(layer == lPlatform, railing, 0)

    building = (bCache == 1).astype(np.uint8)
    walls = building - cv2.erode(building, strctRect)

    layer = np.where(walls > 0, lWall, layer)

    layer = np.where(railing > 0, lRailingFloor, layer)
    layer = np.where(prevLayer == lRailingFloor, lRailingFeet, layer)
    layer = np.where(prevLayer == lRailingFeet, lRailingHead, layer)

    # TODO modify block cache to fit labels

    
    cv2.imshow("railings", railing * 255) 
    cv2.imshow("bCache", (normalize(bCache) * 255).astype(np.uint8)) 
    cv2.waitKey(0 if i==1 else 1)

    # build
    if BUILD:
        rndNoise = noise(shape2d, shape2d)
        buildByCondition(layer == lTopsoil, y, "blackstone")
        buildByCondition(layer == lPlatform, y, "blue_terracotta")
        buildByCondition((layer == lIndoorFeet) & (rndNoise < 0.25) , y, "white_carpet")
        buildByCondition((layer == lOutdoorFeet) & (rndNoise < 0.25) , y, "gray_carpet")
        # buildByCondition(np.isin(layer, [lIndoorHead, lOutdoorHead, lIndoorAir, lRailingHead]), y, "air")
        # buildByCondition(layer == lOutdoorHead, y, "air")
        # buildByCondition(layer == lIndoorAir, y, "air")
        # buildByCondition(layer == lRailingHead, y, "air")
        buildByCondition(layer == lIndoorFloor, y, "white_concrete")
        buildByCondition(layer == lRailingFloor, y, "light_gray_concrete")
        buildByCondition(layer == lRailingFeet, y, "white_stained_glass_pane")
        buildByCondition(layer == lWall, y, "gray_concrete")



# step 4 - traversability

# step 5 - population

# cv2.namedWindow("test", 0)
# cv2.resizeWindow("test", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)

# for i in range(maxHeight - minHeight):
#     slice = blockCache[:,i,:]
#     slice = slice - cv2.erode(slice, strctCross)

#     cv2.imshow("test", slice * 255) 
#     cv2.waitKey(0 if i==1 else 1)
    

# step 6  - decoration


interfaceUtils.sendBlocks()
