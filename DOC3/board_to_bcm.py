BOARD2BCM=[
   -1, -1,  2, -1,  3, -1,  4, 14,
   -1, 15, 17, 18, 27, -1, 22, 23,
   -1, 24, 10, -1,  9, 25, 11,  8,
   -1,  7,  0,  1,  5, -1,  6, 12,
   13, -1, 19, 16, 26, 20, -1, 21]

def Board_to_BCM(pin):
   if pin < 1 or pin > 40:
      return -1
   else:
      return BOARD2BCM[pin-1]
