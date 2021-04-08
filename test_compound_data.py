from worldLoader import WorldSlice
import interfaceUtils

# for testing
interfaceUtils.runCommand("execute at @p run setbuildarea ~ ~-1 ~ ~ ~ ~")

# 2D build area
buildArea = interfaceUtils.requestBuildArea()
if buildArea != -1:
    x1 = buildArea["xFrom"]
    y1 = buildArea["yFrom"]
    z1 = buildArea["zFrom"]
    x2 = buildArea["xTo"]
    y2 = buildArea["yTo"]
    z2 = buildArea["zTo"]
    # print(buildArea)
    area = (x1, z1, x2 - x1, z2 - z1)


ws = WorldSlice(area, [])

block = ws.getBlockCompoundAt((x1, y1, z1))

print(block)
print(block["Properties"])
print(block["Properties"]["half"])