# This file is used to convert an integer division by a contstant into a multiplication.
# For details on the algorithm, look here: https://gmplib.org/~tege/divcnst-pldi94.pdf
# Note, the version implemented here is for unsigned division, though there are other algs for signed.
# Also, if this alg fails, there is another that should work here: http://ridiculousfish.com/files/faster_unsigned_division_by_constants.pdf

from argparse import ArgumentParser
from math import ceil

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

    for l in range(bits):
        pow_2 = 1 << (bits + l)

        # We use int division because we need precision.
        m = pow_2//divisor

        # We were supposed to round up, not trunc(round down).
        # So correct that now.
        if pow_2 % divisor > 0:
            m += 1

        if pow_2 <= m*divisor <= pow_2+(1 << l):
            print('Found a solution')
            print(f'multiply by {m}')
            print(f'in hex: {hex(m)}')
            print(f'then right shift {bits+l} times')
            exit()

    print('Found no solution, maybe try a different algorithm')
