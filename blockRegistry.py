

from interfaceUtils import sendDirect


def createPattern2D(array, palette):
    lines = []
    for i in range(len(array)):
        row = array[i]
        for j in range(len(row)):
            blockID = palette[row[j]]
            if blockID is not None:
                lines.append(f"~{i} ~ ~{j} {blockID}")

    return str.join("\n", lines)

def sendPattern(pattern, x, y, z):
    sendDirect(pattern, x, y, z)


patternElevator = createPattern2D(\
   [[0,1,0],
    [1,2,1],
    [0,1,0]], ["air", "black_stained_glass_pane", "ladder"]
)
patternElevatorFloor = createPattern2D(\
   [[0,1,0],
    [1,1,1],
    [0,1,0]], ["slime_block", "gray_concrete", "soul_sand"]
)
patternElevatorLight = createPattern2D(\
   [[0,0,4,0,0],
    [0,1,2,1,0],
    [4,2,3,2,4],
    [0,1,2,1,0],
    [0,0,4,0,0]], [None, "air", "black_stained_glass_pane", "ladder", "sea_lantern"]
)