import interfaceUtils
import time

i = 0
while True: 
    time.sleep(0.1)
    i += 1
    rotation = i % 16
    pingpong = abs(i%6-2) + 1

    cmd = "setblock -126 74 175 minecraft:oak_sign[rotation=%i]{Text%i:'{\"text\":\"G D M C\"}'}" % (rotation, pingpong)
    interfaceUtils.runCommand(cmd)