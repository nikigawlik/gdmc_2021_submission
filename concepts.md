things to try: 
push rects randomly in +- x/z, this will join some rects and combine some rects

concepts:
id map: values are 0,1,2,3,... etc. which map to rooms objects in an array

skyscraper layers are similar to a classic rougelike

=> we can create a maximally connected graph, by growing rooms from a seed point upwards

=> we can then run a cyclic generator on the graph, or something simpler, like overlapping spanning trees

```
#########         
#       #    ######  ###########
#       #    #    #  #         #
#       ++++++    #  #         #
#       #    #    ++++         #############
#####+###    ##+###  #         #           #
     +         +     #         +           #
    #+#########+#    #         #           #
    #           #    #         #           #
    #           #    #################+#####
    #           #    #                     #
    #           ++++++                     #
    #############    #######################
```

------

wallspace

<!-- outlmap -> general graph -> cycles -->

outlmap -> walker 

big loop algorithm:

(0 -> void, 1 -> unprocessed outline, 2 -> current loop)
- for pixel in allpixels:
  - is pixel == 1?
    - create empty stack for positions
    - mark this pixel as 2
    - repeat
      - if neighboring pixel exists that is 1 or 2
        - push position to stack
        - walk to that pixel
        - set current position to 0
        - if neighboring pixel is 2
          - save stack as a loop object
          - break repeat
      - else
        - this is a dead end
        - pop element from stack and set it as current position



death of a dream of what could have been, when did it die i dont know


n bisschen mehl, gries, 



n x n pathfinding algorithm!

- repeat 10000 times
  <!-- - choose random start and destination -->
  <!-- - solve with A* -> heuristic is manhattan distance + cost map + roofs -->
  - let an agent start at a house and randomly walk towards another house
    - agent either
      - walks forward
      - turns left / right
      - walks forward + up/down
    - agent checks areas forward, left and right to make a decision, but prefers forward
  - mark path on cost map -> path as good, walk spaces as bad
  - diffuse cost map


Current state of platforms :O)



Problem:

I have a bunch of random boxes which might touch each other and might not, might open up to a platform or not. I want to build pathways which connect all boxes _to each other_ (so I can walk from any box to any box) in a way that is natural, uses existing platforms and also connects to the ground.

B RA I N SO TO RM

oky

so i can just have a travesibility map 2d
  - and only add boxes with connections to tvsbility map 
  - and roofs also connect to map
  - and I can't override the map or else it wont be travesible
I flow down in a weird way like a bunch of sirup
I run a cellular automaton which is 2d but the axis is time and it's magical or some shit
I can pathfind between random points and just see were it goes
I can have a grid where spaces are either platform, stairs, or air:  
I can take a staircase and grow from it actually
```
....____.....
___/....\____
```
-------------
I can generate a box pile, define connections between boxes (hypothetical) and use that as a graph to do transformations on
I just run a shit ton of agents which just walk or shit
I run the cellular automaton, but it's actually agents with rotation etc. instead of fixed cells
  - also agents can turn around
I run an A* from every box to the center elevator shaft
  - it prioritizes existing pathways
  - I could re-run multiple times 
poisson disc sampling through additive blurred disc overlaps and max()
-------------
Just run down ladders to closest platform
Platforms _are_ areas of interest
Every box just has it's own staircase
When placing boxes we are only allowed to place them next to staircases (which are a special kind of box)
Buildings can be any shape, so we can incorporate a staircase into every building and just leave it out if we don't need it
2D city blocks with each a staircase -> staircase kinda defines a 'block'
fire escapes could be a fun second staircase
Additional connections are allowed
-------------
mixed development means thinking of a 'block' as a unit -> bottom is stores, top is housing
```  
                       #-----#
              roof     | s  /|
          #------------#-----#
        , | home       | t  /|
        ###------------#-----#
        , | home       | a  /|
        ###------------#-----#
        , | home       | i  /|
        ###------------#-----#
        , | home       | r  /|
        ###-------#----#-----#
          | store |    | s  /|
--street--#-------#----#-----#
```
IF we want scifi aesthetic, we want to have a second street somehow above the bottom one

Create 2D connected road grid -> blocks with staircases
-> second layer is on top

```
1. poisson placed stairwells
2. voronoi between stairwells (watershed)
3. Identify corners & edges 
4. build graph


Solution:

We keep the city map to 2D
And base it on staircases and Voronoi/Watershed
We define blocks, block == all spaces reachable through a specific staircase
A block is a linear progression from street -> staircase -> roof, with spokes ('flats') reachable from the staircase
We can recycle existing code probably and generate the blocks in parallel bottom to top

Risks:
Watershed does not result in nice building shapes
- city map prototype -> check out shapes
Connection from road to staircase doesn't work out well
- city map prototype -> look at how we can do the connection
Connection from staircase to flats does not work out well
- city map prototype -> can prove this for bottom layer at least
- packing prototype -> can check for higher levels
'blocks' aren't adjacent in good ways (need to be packed close to each other)
- packing prototype -> see if it works
Road network is shit
- city map prototype + existing code -> should give an idea for the shape
threedimensionality is lost
- should see either in city map proto + existing code /or in the packing experiment
Doesn't look as interesting because of added structure
- should see either in city map proto + existing code /or in the packing experiment

PT_1:
- City map: Use traversability map & random points (poisson) & watershed to create sections
  - PT_1.2 poisson disc sampling through best random candidate & traversability preserving
- find verteces and edges through edge detection and create a road graph
- intersect random spanning trees for a road map
- place staircases first
- then place other buildings (ground level)
PT_1 Qs:
- Do blocks have a nice shape?
- Do roads have a nice shape?
- Does it work well in hilly terrain and water?
- can we connect the staircase to a road?
- is the whole 'gestalt' interesing?

PT_2: 
- Introduce box stacking to PT_1
PT_2 Qs:
- Are boxes packing tightly and efficiently?
- Is the 3D shape interesting?
- Can we connect the flats to the staircases?
- Can we use this as a starting point for other layers on top of it?