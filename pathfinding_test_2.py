import numpy as np
import cv2


TAU = 6.283
PI = TAU/2
rng = np.random.default_rng()

space = np.zeros((256, 256, 1))
size = np.array(space.shape)[:2]

nAgents = 10000
agentPos = rng.random((nAgents, 2)) * size
d = rng.random(nAgents) * TAU
spd = 5
agentVel = np.transpose(np.array([np.cos(d), np.sin(d)]))


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

t = 0
while True:
    # move
    gradientX = cv2.Scharr(space, cv2.CV_64F, 1, 0)
    gradientY = cv2.Scharr(space, cv2.CV_64F, 0, 1)
    gradient = np.transpose(np.array([gradientY, gradientX]), [1,2,0])

    agentVel += readAtP(agentPos + agentVel, gradient)
    agentVel = agentVel / np.expand_dims(np.linalg.norm(agentVel, axis=1), 1)

    agentPos += agentVel * spd
    writeAtP(agentPos, space, 1)

    # blur and dissolve
    space = cv2.GaussianBlur(space, (3,3), .4)
    space *= 0.95

    # draw
    img = (space * 255).astype(np.uint8)
    cv2.imshow("agents2", img)

    r = gradientX
    g = gradientY
    b = gradientY + gradientX
    img = cv2.merge((b, g, r))
    cv2.imshow("gradient", img)
    cv2.waitKey(0 if t == 0 else 1)
    t += 1