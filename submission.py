from functools import reduce
from math import cos, sin
from random import randint, random, randrange
import cv2
import numpy as np
from numpy.core.shape_base import block
from mapUtils import calcGoodHeightmap, visualize, listWhere
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

# 2D build area
buildArea = interfaceUtils.requestBuildArea()
if buildArea != -1:
    x1 = buildArea["xFrom"]
    z1 = buildArea["zFrom"]
    x2 = buildArea["xTo"]
    z2 = buildArea["zTo"]
    # print(buildArea)
    area = (x1, z1, x2 - x1, z2 - z1)


# step 2 - analysis

worldSlice = WorldSlice(area, ["MOTION_BLOCKING_NO_LEAVES"])
heightmap = calcGoodHeightmap(worldSlice)
heightmap = heightmap.astype(np.uint8)
baseheight = cv2.medianBlur(heightmap, 13)

originalHeightmap = np.array(heightmap)

# big cube
DO_BLOCK_ANA = False
if DO_BLOCK_ANA:
    logIDs = [f"{wood}_log" for wood in minecraft_woods]
    leavesIDs = [f"{wood}_leaves" for wood in minecraft_woods]

    blockGroups = {
        0: ["air", "cave_air", "void_air"],
        1: ["stone", "sandstone", "diorite", "granite", "andesite"],
        2: ["sand", "grass_block", "mycelium", "podzol", "dirt"],
        3: ["water", "flowing_water", "lava", "flowing_lava"],
        4: logIDs + leavesIDs,
    }

    paletteLookup = {}
    for i in blockGroups.keys():
        for blockID in blockGroups[i]:
            paletteLookup["minecraft:" + blockID] = i

    unknownID = max(blockGroups.keys()) + 1

    unknownBlocks = set() # for diagnostic purposes 

    blockIDMap = np.ones((128, 128, 128), dtype=np.uint8) * unknownID

    for dy in range(128):
        y = 64 + dy
        print(f"processing layer {dy}/128")
        for dx in range(128):
            x = area[0] + dx
            for dz in range(128):
                z = area[1] + dz
                
                blockID = worldSlice.getBlockAt((x, y, z))
                numID = paletteLookup.get(blockID, 255)
                if numID == 255:
                    unknownBlocks.add(blockID)
                else:
                    blockIDMap[dx, dy, dz] = numID 

    print(f"unknown blocks: {unknownBlocks}")

    def flattenCube():
        result = np.zeros((128 * 16, 128 * 8))
        for n in range(8):
            for m in range(16):
                i = 16 * n + m
                print(f"drawing layer {i}/128")
                dx1 = m*128
                dx2 = (m+1)*128
                dz1 = n*128
                dz2 = (n+1)*128
                result[dx1:dx2, dz1:dz2] = blockIDMap[:,i,:]
        return result

    flattened = flattenCube()
    flattened = np.transpose(flattened)
    visualize(flattened)

# step 3 - construction
rng = np.random.default_rng()

# drop box algorithm
boxes = []

buildingsOnlyHeightmap = np.zeros(heightmap.shape, dtype=np.uint8)
forbiddenMap =  np.zeros(heightmap.shape, dtype=np.uint8)
# forbiddenMap isn't used right now, but will be at some point

borderMap = np.zeros(heightmap.shape, dtype=np.uint8)
cv2.rectangle(borderMap, (0, 0), (area[3]-1, area[2]-1), (1), 3)

# visualize(heightmap)
# TODO refactor to use more vecors, less tuples

maxHeight = 150
minHeight = heightmap.min()

# block cache
# 0 = completely free
# 255 = blocked
blockCache = np.zeros((area[2], maxHeight-minHeight, area[3]), dtype=np.uint8)

# traffic 
# 0 = indifferent
# 255 = go here it's the place to be
traffic = np.zeros((area[2], maxHeight-minHeight, area[3]), dtype=np.uint8)

maxBoxWidth = (area[2] - 3, area[3] - 3)

BUILD = True    

for i in range(100):
    if i == -1:
        # elevator shaft box
        o = randrange(2)
        sx = [7,10][o]
        sz = [10,7][o]
        sy = 100
    else:
        sx = randrange(7, min(21, maxBoxWidth[0]))
        sz = randrange(7, min(21, maxBoxWidth[1]))
        sy = randint(1, 3) * 6 # 1, 2 or 3 floors high

    offset = (int(sx/2), int(sz/2))
    
    gap = randrange(0,5)
    strctCross = cv2.getStructuringElement(cv2.MORPH_RECT, (sz + 2*gap, sx + 2*gap))
    anchor = (offset[1] + gap, offset[0] + gap)
    mhm = cv2.dilate(buildingsOnlyHeightmap, strctCross, anchor=anchor)

    dilatedBorderMap = cv2.dilate(borderMap, strctCross, anchor=anchor)
    dilatedHeightmap = cv2.dilate(heightmap, strctCross, anchor=anchor)

    # desiredBuildPosMap = (mhm == buildingsOnlyHeightmap) * (forbiddenMap == 0) * (dilatedBorderMap == 0) * (255-dilatedHeightmap)
    # maxi = desiredBuildPosMap.max()
    # if maxi == 0:
    #     raise Exception("lol no space") # TODO obv

    # buildPositions = listWhere(desiredBuildPosMap == maxi)
    buildPositions = listWhere((mhm == buildingsOnlyHeightmap) & (forbiddenMap == 0) & (dilatedBorderMap == 0))

    lastBoxPos = (boxes[-1][0], boxes[-1][1], boxes[-1][2]) if len(boxes) > 0 else (int(area[2]/2), 0, int(area[3]/2))
    distSquared = lambda p1: (p1[0] - lastBoxPos[0])**2 + (dilatedHeightmap[p1] - lastBoxPos[2])**2 * 100 + (p1[1] - lastBoxPos[2])**2

    if len(buildPositions) > 0:
        p = reduce(lambda a,b : a if distSquared(a) < distSquared(b) else b, buildPositions)
        # p = buildPositions[randrange(len(buildPositions))]
    else:
        p = (int(area[2]/2), int(area[3]/2))
    
    dx = p[0] #- anchor[0]
    dz = p[1] #- anchor[1]

    # corner x,z
    cx = p[0] - offset[0]
    cz = p[1] - offset[1]
    
    y = dilatedHeightmap[dx, dz]
    y = y - y%3

    # randomly turn it into a platform
    if random() < .2:
        y = y + sy - 1
        sy = 1

    if y + sy >= maxHeight:
        continue

    boxes.append([dx, y, dz, sx, sy, sz]) # local space! center pos

    # x,y,z are corner pos
    x = area[0] + cx
    z = area[1] + cz

    print(f"build cube at {(x, y, z)}")

    # pillars
    for rx in range(2):
        for rz in range(2):
            xx = cx + (sx - 1) * rx
            zz = cz + (sz - 1) * rz
            yFloor = heightmap[xx, zz]
            for yy in range(yFloor, y):
                if BUILD:
                    interfaceUtils.placeBlockBatched(area[0] + xx, yy, area[1] + zz, "gray_wool")

            blockCache[xx, (yFloor-minHeight):(y-minHeight), zz] = 1

    if BUILD:
        # if i == 0:
        interfaceUtils.buildWireCube(x, y, z, sx, sy, sz)
        # else:
        #     interfaceUtils.buildHollowCube(x, y, z, sx, sy, sz)

    # color = minecraft_colors[(y)%16]
    # interfaceUtils.buildSolidCube(x, y, z, sx, 1, sz, f"{color}_wool") # floor
    
    # interfaceUtils.buildSolidCube(x-1, y, z-1, sx+2, 1, sz+2, f"{color}_wool") # floor
    # color = minecraft_colors[((y+sy-1))%16]
    # interfaceUtils.buildSolidCube(x-1, y+sy-1, z-1, sx+2, 1, sz+2, f"{color}_concrete") # roof


    bheight = y + sy
    heightmap[cx:cx+sx, cz:cz+sz]              = bheight
    buildingsOnlyHeightmap[cx:cx+sx, cz:cz+sz] = bheight


    dy = y - minHeight
    # block out walls
    blockCache[cx:cx+sx, (dy):(dy+sy), cz:cz+sz] = 1
    # encourage roof
    traffic[cx:cx+sx, y+sy-minHeight-1, cz:cz+sz] = 128

    # visualize(mhm, heightmap, buildingsOnlyHeightmap)
    # visualize(buildingsOnlyHeightmap)
    
    img = buildingsOnlyHeightmap + mhm
    img[cx, cz] = 230
    img[cx+sx, cz] = 230
    img[cx, cz+sz] = 230
    img[cx+sx, cz+sz] = 230
    img[dx, dz] = 230
    img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("img", img)
    cv2.waitKey(0 if i == 0 else 1)
    # cv2.waitKey(1)


# test

cv2.namedWindow("slices", 0)
cv2.resizeWindow("slices", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)
cv2.namedWindow("lastLayer", 0)
cv2.resizeWindow("lastLayer", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)
cv2.namedWindow("bCache", 0)
cv2.resizeWindow("bCache", int(heightmap.shape[1] / heightmap.shape[0] * 512), 512)


dilSize = 10
walkwayWidth = 4
kSize = dilSize * 2 + 1
strctBigCirc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kSize,kSize))
strctCross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
# strctMedCirc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
traversible = blockCache == 0


# BUILD = False

heightmap = np.array(originalHeightmap)

lastLayer = np.zeros((3, area[2], area[3]), np.uint8)
lastLayerY = -1

for i in range(maxHeight - minHeight):
    # pathway maps
    # TODO think about lowest and heighest layer instead of just excluding them?
    if i > 0 and i < maxHeight - minHeight - 1:
        free = traversible[:,i,:] & traversible[:,i+1,:]
        blocked = ~free
        ground = ~traversible[:,i-1,:] & free

        # platforms
        platform = ground.astype(np.uint8)
        padWidth = dilSize + 1
        platform = np.pad(platform, (padWidth, padWidth), mode='constant') 
        platform = cv2.dilate(platform, strctBigCirc)
        platform = cv2.erode(platform, strctCross, iterations=(dilSize - walkwayWidth - 1))

        # # 'distance erode'
        # for j in range(walkwayWidth - 1):
        #     platform = np.maximum(cv2.erode(platform+1, strctCross), platform)-1

        # platform /= walkwayWidth

        platform = platform[padWidth:-padWidth, padWidth:-padWidth]
        
        # actualHeight = y + platform
        y = minHeight + i - 1 
        diffToHM = np.maximum(0, y - heightmap) * platform
        diffToHM = cv2.bitwise_and(diffToHM, 0b01111111) # fix underflows

        # platform = np.where(diffToHM > 0, platform, walkwayWidth * np.sign(platform))

        platform = (free & ~ground) * platform

        blockCache[:,i-1,:] |= platform > 0

        heightmap = np.where(blockCache[:,i-1,:] > 0, y-1, heightmap)
        cv2.imshow("bCache", heightmap)

        # visualize
        r = blocked.astype(np.uint8) * 255
        g = ground.astype(np.uint8) * 255 
        b = platform.astype(np.uint8) * int(255 / walkwayWidth)
        # b = (originalHeightmap > i + minHeight).astype(np.uint8) * 255

        bgr = cv2.merge((b, g, r))

        cv2.imshow("slices", bgr)
        cv2.imshow("lastLayer", np.transpose(lastLayer, [1,2,0]) * 255)
        cv2.waitKey(0 if i==1 else 1)

        # build layer if necessary
        if platform.max() > 0:
            blockPositions = np.where(platform > 0)
            foundation = np.where(diffToHM > 4, 0, diffToHM)
            y = minHeight + i - 1
            for p in zip(*blockPositions):
                # y += -walkwayWidth + platform[p]
                x = area[0] + p[0]
                z = area[1] + p[1]
                for yy in range(y-foundation[p], y+1):
                    if BUILD:
                        interfaceUtils.placeBlockBatched(x, yy, z, "gray_wool", 400)

            lastLayerY = y
            lastLayer[0,:,:] = blocked.astype(np.uint8)
            lastLayer[1,:,:] = ground.astype(np.uint8)
            lastLayer[2,:,:] = platform
        interfaceUtils.sendBlocks()



cv2.destroyAllWindows()
interfaceUtils.sendBlocks()

# step 4 - traversability
POINTCLOUD = False

if POINTCLOUD:
    agentN = 100
    # agentPos = (rng.random((agentN, 3)) * np.array(traversability.shape)).astype(np.int)
    
    agentPos = np.transpose(np.array(np.where(traffic > 0)))
    pointcloud = agentPos[:,[0,2,1]]

    sortedBoxes = sorted(boxes, key = lambda box: box[1])
    delta = int(len(boxes) / 7)
    layers = [box[1] for box in sortedBoxes[::delta]]
    print(layers)

    pointcolors = np.ones((pointcloud.shape[0], 3)) * np.array([1,1,0])
    v = pptk.viewer(pointcloud, pointcolors)
    v.wait()
    v.close()

# step 5 - population


# step 6  - decoration


interfaceUtils.sendBlocks()