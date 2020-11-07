#https://www.codingame.com/ide/puzzle/the-descent
#Szymon Maj s14965
import sys
import math

# The while loop represents the game.
# Each iteration represents a turn of the game
# where you are given inputs (the heights of the mountains)
# and where you have to print an output (the index of the mountain to fire on)
# The inputs you are given are automatically updated according to your last actions.


# game loop
while True:
    target = 0
    target_height = 0
    for mountain_index in range(8):
        mountain_height = int(input())  # represents the height of one mountain.
        if mountain_height > target_height:
            target_height = mountain_height
            target = mountain_index

        
    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

    # The index of the mountain to fire on.
    print(target)
