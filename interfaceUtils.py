# ! /usr/bin/python3
"""### Provide tools for placing and getting blocks and more.

This module contains functions to:
* Request the build area as defined in-world
* Run Minecraft commands
* Get the name of a block at a particular coordinate
* Place blocks in the world
"""
__all__ = ['requestBuildArea', 'runCommand',
           'setBlock', 'getBlock',
           'placeBlockBatched', 'sendBlocks']
# __version__

from random import randrange
from numpy.lib.function_base import place
import requests
from mapUtils import minecraft_colors


def requestBuildArea():
    """**Requests a build area and returns it as an dictionary containing 
    the keys xFrom, yFrom, zFrom, xTo, yTo and zTo**"""
    response = requests.get('http://localhost:9000/buildarea')
    if response.ok:
        return response.json()
    else:
        print(response.text)
        return -1


def runCommand(command):
    """**Executes one or multiple minecraft commands (separated by newlines).**"""
    # print("running cmd " + command)
    url = 'http://localhost:9000/command'
    try:
        response = requests.post(url, bytes(command, "utf-8"))
    except ConnectionError:
        return "connection error"
    return response.text

# --------------------------------------------------------- get/set block


def getBlock(x, y, z):
    """**Returns the namespaced id of a block in the world.**"""
    url = f'http://localhost:9000/blocks?x={x}&y={y}&z={z}'
    # print(url)
    try:
        response = requests.get(url)
    except ConnectionError:
        return "minecraft:void_air"
    return response.text
    # print("{}, {}, {}: {} - {}".format(x, y, z, response.status_code, response.text))


def setBlock(x, y, z, str):
    """**Places a block in the world.**"""
    url = f'http://localhost:9000/blocks?x={x}&y={y}&z={z}'
    # print('setting block {} at {} {} {}'.format(str, x, y, z))
    try:
        response = requests.put(url, str)
    except ConnectionError:
        return "0"
    return response.text
    # print("{}, {}, {}: {} - {}".format(x, y, z, response.status_code, response.text))


# --------------------------------------------------------- block buffers

blockBuffer = []


def placeBlockBatched(x, y, z, str, limit=50):
    """**Place a block in the buffer and send if the limit is exceeded.**"""
    registerSetBlock(x, y, z, str)
    if len(blockBuffer) >= limit:
        return sendBlocks(0, 0, 0)
    else:
        return None


def sendBlocks(x=0, y=0, z=0, retries=5):
    """**Sends the buffer to the server and clears it.**"""
    global blockBuffer
    body = str.join("\n", ['~{} ~{} ~{} {}'.format(*bp) for bp in blockBuffer])
    url = f'http://localhost:9000/blocks?x={x}&y={y}&z={z}'
    try:
        response = requests.put(url, body)
        clearBlockBuffer()
        return response.text
    except (ConnectionError, requests.ConnectionError) as e:
        print(f"Request failed: {e} Retrying ({retries} left)")
        if retries > 0:
            return sendBlocks(x, y, z, retries - 1)
        else:
            raise e


def registerSetBlock(x, y, z, str):
    """**Places a block in the buffer.**"""
    global blockBuffer
    # blockBuffer += () '~{} ~{} ~{} {}'.format(x, y, z, str)
    blockBuffer.append((x, y, z, str))


def clearBlockBuffer():
    """**Clears the block buffer.**"""
    global blockBuffer
    blockBuffer = []

def randomWool():
    return f"{minecraft_colors[randrange(len(minecraft_colors))]}_wool"

def buildWireCube(x, y, z, sx, sy, sz, blockID = None):
    blockID = randomWool() if blockID is None else blockID

    for i in range(sx):
        placeBlockBatched(x + i, y, z, blockID)
    for i in range(sy):
        placeBlockBatched(x, y + i, z, blockID)
    for i in range(sz):
        placeBlockBatched(x, y, z + i, blockID)
    for i in range(sy):
        placeBlockBatched(x + sx - 1, y + i, z, blockID)
    for i in range(sz):
        placeBlockBatched(x + sx - 1, y, z + i, blockID)
    for i in range(sx):
        placeBlockBatched(x + i, y + sy - 1, z, blockID)
    for i in range(sz):
        placeBlockBatched(x, y + sy - 1, z + i, blockID)
    for i in range(sx):
        placeBlockBatched(x + i, y, z + sz - 1, blockID)
    for i in range(sy):
        placeBlockBatched(x, y + i, z + sz - 1, blockID)
    for i in range(sx):
        placeBlockBatched(x + i, y + sy - 1, z + sz - 1, blockID)
    for i in range(sy):
        placeBlockBatched(x + sx - 1, y + i, z + sz - 1, blockID)
    for i in range(sz):
        placeBlockBatched(x + sx - 1, y + sy - 1, z + i, blockID)


def buildSolidCube(x, y, z, sx, sy, sz, blockID = None):
    blockID = randomWool() if blockID is None else blockID
    # blockID = "light_gray_concrete"


    for yy in range(y, y+sy):
        for xx in range(x, x+sx):
            for zz in range(z, z+sz):
                placeBlockBatched(xx, yy, zz, blockID)

def buildHollowCube(x, y, z, sx, sy, sz, blockID = None):
    blockID = randomWool() if blockID is None else blockID
    # blockID = "light_gray_concrete"

    for yy in range(y, y+sy):
        for xx in range(x, x+sx):
            for zz in range(z, z+sz):
                if xx == x or yy == y or zz == z or xx == x+sx-1 or yy == y+sy-1 or zz == z+sz-1:
                    placeBlockBatched(xx, yy, zz, blockID)