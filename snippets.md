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
        ym1 = 64 + dy
        print(f"processing layer {dy}/128")
        for dx in range(128):
            x = area[0] + dx
            for dz in range(128):
                z = area[1] + dz
                
                blockID = worldSlice.getBlockAt((x, ym1, z))
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








        # # 'distance erode'
        # for j in range(walkwayWidth - 1):
        #     platform = np.maximum(cv2.erode(platform+1, strctCross), platform)-1

        # platform /= walkwayWidth



# agent pointcloud
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