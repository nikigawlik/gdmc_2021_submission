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
   - 