# This file is used to convert an integer division by a contstant into a multiplication.
# For details on the algorithm, look here: http://ridiculousfish.com/files/faster_unsigned_division_by_constants.pdf
# The conclusion section definse what is implemented here in simplest terms.

from argparse import ArgumentParser
from math import log2, floor

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Calculates an N bit constant to multiply an unsigned integer by to compute division')
    parser.add_argument('-b', '--bits', type=int, required=True,
                        help='The number of bits the division is being performed on')
    parser.add_argument('-d', '--divisor', type=int, required=True,
                        help='The unsigned int to divide by')

    args = parser.parse_args()

    bits = args.bits
    divisor = args.divisor

    for p in range(floor(log2(divisor))+1):
        pow_2 = 1 << (bits + p)

        if pow_2 % divisor <= 1 << p:
            print('Found a solution')
            m = pow_2//divisor
            print('add 1 to your number')
            print(f'multiply by {m}')
            print(f'in hex: {hex(m)}')
            print(f'then right shift {bits+p} times')
            print()
            print('Note: for the addition, if it could overflow, use saturating addition')
            print('If it came from a signed type initially, this is not possible')
            exit()

    print('Found no solution, maybe try a different algorithm')
