from mapUtils import fractalnoise, noise, normalize
import numpy as np
import cv2

# constants
TAU = 6.283
PI = TAU/2
rng = np.random.default_rng()

# simulation space
space = np.zeros((128, 128, 1))
size = np.array(space.shape)[:2]

# aversion
aversion = normalize(np.expand_dims(fractalnoise(space.shape), 2))

# spawnpoints
interval = 32
radius = interval * 0.4
nx = int(size[0] / interval)
ny = int(size[1] / interval)
nSpawnpoints = nx * ny
spawnpointPos = (np.array([[(x+1)/(nx+1), (y+1)/(ny+1)] for x in range(nx) for y in range(ny)])) * size
d = rng.random((nSpawnpoints,)) * TAU
spawnpointPos += np.transpose(np.array([np.cos(d), np.sin(d)])) * radius


# agents
nAgents = 3000
agentSpeed = .5
turnStrength = 0.07

def spawnAgents():
    # pos = rng.random((nAgents, 2)) * size
    pos = spawnpointPos[(rng.random(nAgents) * nSpawnpoints).astype(np.int)]
    d = rng.random(nAgents) * TAU
    vel = np.transpose(np.array([np.cos(d), np.sin(d)]))
    return pos, vel

agentPos, agentVel = spawnAgents()

def writeAtP(positions, arr, val):
    positions = positions.astype(np.int)
    positions = np.clip(positions, [0,0], np.array(arr.shape)[:2] - 1)
    arr[positions[:,0],positions[:,1]] = val

def readAtP(positions, arr):
    positions = positions.astype(np.int)
    positions = np.clip(positions, [0,0], np.array(arr.shape)[:2] - 1)
    return arr[positions[:,0],positions[:,1]]
    
def readAtP2(positions, arr):
    positions = positions.astype(np.int)
    positions = np.clip(positions, [0,0], np.array(arr.shape)[:2] - 1)
    return arr[positions[:,:,0],positions[:,:,1]]

cv2.namedWindow("agents2", 0)
cv2.resizeWindow("agents2", int(space.shape[1] / space.shape[0] * 512), 512)
cv2.namedWindow("gradient", 0)
cv2.resizeWindow("gradient", int(space.shape[1] / space.shape[0] * 512), 512)
cv2.namedWindow("search", 0)
cv2.resizeWindow("search", int(space.shape[1] / space.shape[0] * 512), 512)

t = 0
while True:
    # respawn
    timeout = 512
    rollingRespawn = np.zeros(nAgents)
    rollingRespawn[t%timeout::timeout] = 1

    respawn = rollingRespawn
    respawn =  np.expand_dims(respawn, 1)

    newPos, newVel = spawnAgents()

    agentPos = np.where(respawn, newPos, agentPos)
    agentVel = np.where(respawn, newVel, agentVel)

    # move
    searchSpace = space * (1-aversion)
    cv2.imshow("search", searchSpace)
    gradientX = cv2.Scharr(searchSpace, cv2.CV_64F, 1, 0)
    gradientY = cv2.Scharr(searchSpace, cv2.CV_64F, 0, 1)
    gradient = np.transpose(np.array([gradientY, gradientX]), [1,2,0])

    agentVel += readAtP(agentPos + agentVel, gradient) * turnStrength
    agentVel = agentVel / np.expand_dims(np.linalg.norm(agentVel, axis=1), 1)

    agentPos += agentVel * agentSpeed
    writeAtP(agentPos, space, 1)

    # blur and dissolve
    # space = cv2.GaussianBlur(space, (3,3), .4)
    space *= 0.96

    # draw
    g = (space * 255).astype(np.uint8)
    b = g
    r = np.zeros(space.shape, dtype=np.uint8)
    for i in range(nSpawnpoints):
        cv2.circle(b, (int(spawnpointPos[i][1]), int(spawnpointPos[i][0])), 4, (255), -1)
    
    img = cv2.merge((b, g, r))
    
    cv2.imshow("agents2", img)

    r = gradientX
    g = gradientY
    b = np.zeros(space.shape)
    img = cv2.merge((b, g, r))
    cv2.imshow("gradient", img)
    cv2.waitKey(0 if t == 0 else 1)
    t += 1