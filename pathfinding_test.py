from math import cos, sin
from os import write
from numpy.core.fromnumeric import swapaxes
from numpy.core.numeric import Infinity
from mapUtils import noise, normalize, visualize
import cv2
import numpy as np

TAU = 6.283
PI = TAU/2
rng = np.random.default_rng()

space = np.zeros((512, 512))
blurredSpace = np.array(space)

center = np.array(space.shape) * 0.5

# obstacle = np.zeros(space.shape, dtype=np.uint8)
obstacle = noise(space.shape, (12,12))
# cv2.circle(obstacle, tuple(center.astype(np.int)), 6, (48), -1)

nx = int(space.shape[0] / 100)
ny = int(space.shape[1] / 100)
r = 40
nTargets = nx * ny
# targetPos = rng.random((nTargets, 2)) * np.array(space.shape)
targetPos = (np.array([[(x+1)/(nx+1), (y+1)/(ny+1)] for x in range(nx) for y in range(ny)])) * space.shape
d = rng.random((nTargets,)) * TAU
targetPos += np.transpose(np.array([np.cos(d), np.sin(d)])) * r

targetRadius = 4

# up to n exits
exitsPerTarget = 64
targetExitDir = rng.random((nTargets,exitsPerTarget)) * TAU

nAgents = 60000
# agentPos = rng.random((nAgents, 2)) * np.array(space.shape)
# agentPos = np.ones((nAgents, 2)) * np.array(space.shape) * 0.5
# agentDir = rng.random((nAgents,)) * .4
agentPos = targetPos[(rng.random(nAgents) * nTargets).astype(np.int),:]
agentDir = rng.random((nAgents,)) * TAU
agentTarget = targetPos[(rng.random(nAgents) * nTargets).astype(np.int),:]

# spawnTargetID = (rng.random(nAgents) * nTargets).astype(np.int)
# agentPos = targetPos[spawnTargetID,:]
# agentDir = targetExitDir[spawnTargetID, (rng.random((nAgents,)) * exitsPerTarget).astype(np.int)]

img = np.zeros(space.shape, dtype=np.uint8)

border = 4.0

feelerDist = 16
feelerSize = 8
theta = TAU/8
turnSpeed = .2
moveSpeed = .5


def readAtP(positions, arr):
    positions = positions.astype(np.int)
    positions = np.clip(positions, [0,0], np.array(arr.shape)[:2] - 1)
    return arr[positions[:,0],positions[:,1]]


def readAtP2(positions, arr):
    positions = positions.astype(np.int)
    positions = np.clip(positions, [0,0], np.array(arr.shape)[:2] - 1)
    return arr[positions[:,:,0],positions[:,:,1]]

def writeAtP(positions, arr, val):
    positions = positions.astype(np.int)
    positions = np.clip(positions, [0,0], np.array(arr.shape)[:2] - 1)
    arr[positions[:,0],positions[:,1]] = val

def addAtP(positions, arr, val):
    positions = positions.astype(np.int)
    positions = np.clip(positions, [0,0], np.array(arr.shape)[:2] - 1)
    arr[positions[:,0],positions[:,1]] += val

rotLeft = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
rotRight = np.array([[cos(-theta), -sin(-theta)], [sin(-theta), cos(-theta)]])

initialFrame = True
t = 0

cv2.namedWindow("agents", 0)
cv2.namedWindow("result", 0)
cv2.resizeWindow("agents", int(space.shape[1] / space.shape[0] * 512), 512)
cv2.resizeWindow("result", int(space.shape[1] / space.shape[0] * 512), 512)

while True:
    t += 0.01
    # selective destruction
    borderFactor = np.abs(agentPos - center) - center + border
    borderFactor = np.maximum(borderFactor[:,0], borderFactor[:,1])
    borderFactor = np.maximum(borderFactor, 0)
    atEdge = (borderFactor > .5)

    inTarget = np.linalg.norm(agentPos - agentTarget) < targetRadius

    inObstacle = readAtP(agentPos, obstacle) > 0.5

    randomlyChosen = rng.random(nAgents) > (0.95 + 0.05 *readAtP(agentPos, space))

    respawn = atEdge | randomlyChosen | inObstacle

    # addAtP(agentPos, space, -1 * respawn)

    # agentDir += (borderFactor > .5) * (TAU/2)
    # agentPos = agentPos % np.array(space.shape)
    # agentPos = np.where(atEdge.reshape((nAgents,1)), np.ones((nAgents,2)) * center, agentPos)
    # agentPos = np.where(respawn.reshape((nAgents,1)), rng.random((nAgents,2)) * np.array(space.shape), agentPos)
    agentDir = np.where(respawn, rng.random((nAgents,)) * TAU, agentDir)
    agentPos = np.where(respawn.reshape((nAgents,1)), targetPos[(rng.random(nAgents) * nTargets).astype(np.int),:], agentPos)
    # spawnTargetID = (rng.random(nAgents) * nTargets).astype(np.int)
    # agentPos = np.where(respawn.reshape((nAgents,1)), targetPos[spawnTargetID,:], agentPos)
    # agentDir = np.where(respawn, targetExitDir[spawnTargetID, (rng.random((nAgents,)) * exitsPerTarget).astype(np.int)], agentDir)

    # adjust target exits
    relativePos = np.transpose(np.array([np.cos(targetExitDir), np.sin(targetExitDir)]), axes=[1,2,0])
    leftPos = targetPos.reshape((nTargets, 1, 2)) + np.matmul(relativePos, rotLeft) * 16
    rightPos = targetPos.reshape((nTargets, 1, 2)) + np.matmul(relativePos, rotRight) * 16
    targetExitDir += (-readAtP2(leftPos, blurredSpace) + readAtP2(rightPos, blurredSpace)) * (1-readAtP2(relativePos, blurredSpace)) * 0.1

    # rotate agents
    relativePos = np.transpose(np.array([np.cos(agentDir), np.sin(agentDir)]))
    # forwPos = agentPos + relativePos * 6
    leftPos = agentPos + np.matmul(relativePos, rotLeft) * feelerDist
    rightPos = agentPos + np.matmul(relativePos, rotRight) * feelerDist

    # adjustedTurnSpeed = turnSpeed * min(t*0.2, 10)
    agentDir += (-readAtP(leftPos, blurredSpace) + readAtP(rightPos, blurredSpace)) * (1-readAtP(relativePos, blurredSpace)) * turnSpeed


    # move agents
    delta = np.transpose(np.array([np.cos(agentDir), np.sin(agentDir)]))
    agentPos = agentPos + delta * moveSpeed
    # draw agents
    writeAtP(agentPos, space, 1)
    # diffuse etc.
    space = space * 0.99

    space = cv2.GaussianBlur(space, (3,3), .4)
    blurredSpace = cv2.blur(space,(feelerSize,feelerSize))

    r = np.zeros(space.shape, dtype=np.uint8)
    for i in range(nTargets):
        cv2.circle(r, (int(targetPos[i][1]), int(targetPos[i][0])), targetRadius, (255, 128, 0), -1)
    g = (normalize(np.maximum(space, 0)) * 255).astype(np.uint8)
    # b = (normalize(-np.minimum(space, 0)) * 255).astype(np.uint8)
    b = (obstacle * 2).astype(np.uint8) * 255
    img = cv2.merge((b, g, r))
    cv2.imshow("agents", img)

    img2 = ((space > 0.2) * 255).astype(np.uint8)
    strctElmt = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img2 = cv2.dilate(cv2.erode(img2, strctElmt), strctElmt)
    cv2.imshow("result", img2)

    cv2.waitKey(0 if initialFrame else 1)
    initialFrame = False