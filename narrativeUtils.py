import random

vowels = ["a", "e", "i", "o", "u", "y", "oo", "uh", "ei", "au", "'"]
consonants = ["qu", "w", "r", "t", "y", "i", "p", "s", "d", "f", "g", "h", "j", "k", "l", "z", "x", "c", "v", "b", "n", "m"]

def choose(x):
    return x[random.randrange(0, len(x))]

def chance(x):
    return random.random() < x

def genSil():
    txt = choose(consonants)
    while chance(0.2):
        txt += choose(consonants)
    txt += choose(vowels)
    return txt

def genName():
    txt = ""
    for i in range(choose([1, 1, 1, 2, 2, 3])):
        txt += genSil()
    if chance(0.5):
        txt += choose(consonants)
    return txt.capitalize()

def genSurname():
    txt = ""
    txt += choose(consonants)
    txt += choose(vowels)
    cns = choose(consonants)
    for i in range(choose([1, 2])):
        txt += cns
    if chance(0.5):
        txt += choose(vowels)
        txt += choose(consonants)
    return txt.capitalize()

for i in range(100):
    print("%i\t%s %s" % (i, genName(), genSurname()))