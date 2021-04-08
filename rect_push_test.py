from worldLoader import WorldSlice
import numpy as np
import cv2
import mapUtils
import interfaceUtils
import pptk


area = (700, -750, 256, 256)
halfDX = int(area[2] / 2)
halfDZ = int(area[3] / 2)
print(interfaceUtils.runCommand(f"execute at @p run setbuildarea ~-{halfDX} {64} ~-{halfDZ} ~{area[2] - halfDX} 256 ~{area[3] - halfDZ}"))

yBase = 64
buildArea = interfaceUtils.requestBuildArea()
if buildArea != -1:
    x1 = buildArea["xFrom"]
    z1 = buildArea["zFrom"]
    x2 = buildArea["xTo"]
    z2 = buildArea["zTo"]
    yBase = buildArea["yFrom"]
    print("provided build area is at position %s, %s with size %s, %s, base y = %s" % (x1, z1, x2-x1, z2-z1, yBase))
    area = (x1, z1, 256, 256)
print("(adjusted) build area is at position %s, %s with size %s, %s" % area)

img = cv2.imread("images/cityRects1.png")

cv2.imshow("img", img)
cv2.waitKey(1)

worldSlice = WorldSlice(area)

# calculate a heightmap suitable for building:
heightmap = mapUtils.calcGoodHeightmap(worldSlice)
heightmap = heightmap.astype(np.uint8)

hmo = heightmap
heightmap = cv2.medianBlur(heightmap, 13)

buildingHeightmap = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

paths = (buildingHeightmap == 255) * 1
void = (buildingHeightmap == 0) * 1

buildingHeightmap = buildingHeightmap - paths * 255
cv2.imshow("heightmap", (mapUtils.normalize(buildingHeightmap) * 255).astype(np.uint8))
cv2.waitKey(1)

yHeight = 60
buildingHeightmap = mapUtils.normalize(buildingHeightmap)
buildingHeightmap = (buildingHeightmap * yHeight).astype(np.uint8)

# buildingHeightmap += heightmap - heightmap.min

mapUtils.visualize(buildingHeightmap)

def placeBlock(x, y, z, block):
    interfaceUtils.placeBlockBatched(x, y, z, block, limit = 200)

pointcloud = []
pointcolors = []

for dy in range(yHeight):
    print("layer %i" % dy)
    for dx in range(area[2]):
        for dz in range(area[3]):
            x = area[0] + dx
            z = area[1] + dz
            y = dy + heightmap[dx,dz]

            if dy < buildingHeightmap[dx,dz]:
                placeBlock(x, y, z, "light_gray_concrete")
                pointcloud.append([x, z, y])
                pointcolors.append(buildingHeightmap[dx,dz])

interfaceUtils.sendBlocks()

pointcloud = np.array(pointcloud)
pointcolors = np.array(pointcolors)
v = pptk.viewer(pointcloud, pointcolors)
v.wait()