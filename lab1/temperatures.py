#https://www.codingame.com/ide/puzzle/temperature-code-golf
import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.
def distance(x):
    return x if x > 0 else -x


n = int(input())  # the number of temperatures to analyse
result = 5526 if n > 0 else 0
for i in input().split():
    #a temperature expressed as an integer ranging from -273 to 5526
    if (distance(int(i)) < distance(result)) or (distance(int(i)) == distance(result) and int(i) > result):
        result = int(i) 

# Write an answer using print
# To debug: print("Debug messages...", file=sys.stderr, flush=True)

print(result)
