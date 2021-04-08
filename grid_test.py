import interfaceUtils
# x position, z position, x size, z size
area = (0, 0, 128, 128)  # default build area

# see if a build area has been specified
# you can set a build area in minecraft using the /setbuildarea command

interfaceUtils.runCommand("execute at @p run setbuildarea ~-64 64 ~-64 ~64 192 ~64")

buildArea = interfaceUtils.requestBuildArea()
if buildArea != -1:
    x1 = buildArea["xFrom"] >> 4 << 4
    y1 = buildArea["yFrom"] >> 4 << 4
    z1 = buildArea["zFrom"] >> 4 << 4
    x2 = (buildArea["xTo"] >> 4 << 4) + 1
    y2 = (buildArea["yTo"] >> 4 << 4) + 1
    z2 = (buildArea["zTo"] >> 4 << 4) + 1
    # print(buildArea)
    area = (x1, z1, x2 - x1, z2 - z1)

for y in range(y1, y2):
    for x in range(x1, x2):
        for z in range(z1, z2):
            mx = x % 16 == 0
            my = y % 16 == 0
            mz = z % 16 == 0
            if (mx & my) | (my & mz) | (mz & mx):
                interfaceUtils.placeBlockBatched(x, y, z, "white_wool", 200)

interfaceUtils.sendBlocks() 
