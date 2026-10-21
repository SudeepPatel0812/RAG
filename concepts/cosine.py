import math 

def cosine(a, b):
    total = 0
    sum_a = 0
    sum_b = 0
    for x in a:
        sum_a += x * x

    for y in b:
        sum_b += y * y 

    for x, y in zip (a,b):
        total = total + x*y

    dot = total
    len_a = math.sqrt(sum_a)
    len_b = math.sqrt(sum_b)
    return dot / (len_a * len_b)