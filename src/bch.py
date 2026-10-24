import functools

import galois
import numpy as np


class BinaryField:
    def __init__(self, degree):
        self.degree = degree
        self.order = (1 << degree) - 1
        self.modulus = int(galois.primitive_poly(2, degree))
        self.exp = [0] * (2 * self.order)
        self.log = [0] * (self.order + 1)
        value = 1
        for i in range(self.order):
            self.exp[i] = value
            self.log[value] = i
            value <<= 1
            if value & (1 << degree):
                value ^= self.modulus
        self.exp[self.order :] = self.exp[: self.order]

    def mul(self, a, b):
        return 0 if a == 0 or b == 0 else self.exp[self.log[a] + self.log[b]]

    def div(self, a, b):
        if b == 0:
            raise ZeroDivisionError
        return 0 if a == 0 else self.exp[(self.log[a] - self.log[b]) % self.order]

    def power(self, a, exponent):
        return 0 if a == 0 else self.exp[(self.log[a] * exponent) % self.order]

    def trim(self, polynomial):
        polynomial = list(polynomial)
        while len(polynomial) > 1 and polynomial[-1] == 0:
            polynomial.pop()
        return polynomial

    def add(self, a, b):
        out = [0] * max(len(a), len(b))
        for i, value in enumerate(a):
            out[i] ^= value
        for i, value in enumerate(b):
            out[i] ^= value
        return self.trim(out)

    def multiply(self, a, b):
        out = [0] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                out[i + j] ^= self.mul(x, y)
        return self.trim(out)

    def divide(self, a, b):
        remainder = self.trim(a)
        b = self.trim(b)
        if b == [0]:
            raise ZeroDivisionError
        quotient = [0] * max(1, len(a) - len(b) + 1)
        while remainder != [0] and len(remainder) >= len(b):
            offset = len(remainder) - len(b)
            coefficient = self.div(remainder[-1], b[-1])
            quotient[offset] ^= coefficient
            for j, value in enumerate(b):
                remainder[offset + j] ^= self.mul(coefficient, value)
            remainder = self.trim(remainder)
        return self.trim(quotient), remainder

    def gcd(self, a, b):
        while b != [0]:
            a, b = b, self.divide(a, b)[1]
        return [self.div(value, a[-1]) for value in a]

    def roots(self, polynomial):
        polynomial = self.trim(polynomial)
        if len(polynomial) == 1:
            return []
        if len(polynomial) == 2:
            return [self.div(polynomial[0], polynomial[1])]
        for j in range(self.degree):
            term = [0, 1 << j]
            trace = [0]
            for _ in range(self.degree):
                trace = self.add(trace, term)
                term = self.divide(self.multiply(term, term), polynomial)[1]
            factor = self.gcd(polynomial, trace)
            if 1 < len(factor) < len(polynomial):
                quotient, remainder = self.divide(polynomial, factor)
                if remainder != [0]:
                    raise ValueError("Invalid trace factor")
                return self.roots(factor) + self.roots(quotient)
        raise ValueError("Locator does not split into distinct field roots")

    def berlekamp_massey(self, sequence):
        c = [1] + [0] * len(sequence)
        b = c.copy()
        length = 0
        shift = 1
        discrepancy_old = 1
        for j in range(len(sequence)):
            discrepancy = sequence[j]
            for i in range(1, length + 1):
                discrepancy ^= self.mul(c[i], sequence[j - i])
            if discrepancy == 0:
                shift += 1
                continue
            old = c.copy()
            factor = self.div(discrepancy, discrepancy_old)
            for i in range(len(c) - shift):
                c[i + shift] ^= self.mul(factor, b[i])
            if 2 * length <= j:
                length = j + 1 - length
                b = old
                discrepancy_old = discrepancy
                shift = 1
            else:
                shift += 1
        return self.trim(c[: length + 1])


class BCHMap:
    def __init__(self, dimension, degree):
        self.dimension = dimension
        self.degree = degree
        self.field_bits = dimension.bit_length()
        self.identity = degree * self.field_bits >= dimension
        self.rank = dimension if self.identity else degree * self.field_bits
        if self.rank > 63:
            raise ValueError("BCH syndrome dimension exceeds 63 bits")
        if self.identity:
            self.matrix = np.eye(dimension, dtype=np.uint8)
            self.field = None
        else:
            self.field = BinaryField(self.field_bits)
            self.matrix = np.zeros((self.rank, dimension), dtype=np.uint8)
            for coordinate in range(dimension):
                for j in range(degree):
                    value = self.field.power(coordinate + 1, 2 * j + 1)
                    for bit in range(self.field_bits):
                        self.matrix[j * self.field_bits + bit, coordinate] = (
                            value >> bit
                        ) & 1

        self.quadratic_roots = None
        if not self.identity and self.degree == 2:
            self.quadratic_roots = [-1] * (self.field.order + 1)
            for value in range(self.field.order + 1):
                self.quadratic_roots[self.field.mul(value, value) ^ value] = value

    def encode(self, locations):
        if self.identity:
            return sum(1 << j for j in locations)
        result = 0
        for j in range(self.degree):
            value = 0
            for location in locations:
                value ^= self.field.power(location + 1, 2 * j + 1)
            result |= value << (j * self.field_bits)
        return result

    def decode(self, syndrome):
        syndrome = int(syndrome)
        if self.identity:
            result = tuple(j for j in range(self.dimension) if (syndrome >> j) & 1)
        elif self.degree == 2:
            mask = (1 << self.field_bits) - 1
            first = syndrome & mask
            third = (syndrome >> self.field_bits) & mask
            square = self.field.mul(first, first)
            cube = self.field.mul(square, first)
            if first == 0:
                result = ()
            elif third == cube:
                result = (first - 1,)
            else:
                product = self.field.div(third ^ cube, first)
                root = self.quadratic_roots[self.field.div(product, square)]
                if root < 0:
                    raise ValueError("Quadratic syndrome has no field roots")
                left = self.field.mul(first, root)
                right = left ^ first
                result = tuple(sorted((left - 1, right - 1)))
        else:
            powers = [0] * (2 * self.degree + 1)
            for j in range(1, len(powers)):
                if j % 2:
                    powers[j] = (syndrome >> ((j // 2) * self.field_bits)) & (
                        (1 << self.field_bits) - 1
                    )
                else:
                    powers[j] = self.field.mul(powers[j // 2], powers[j // 2])
            locator = self.field.berlekamp_massey(powers[1:])
            roots = self.field.roots(locator)
            result = tuple(sorted(self.field.div(1, root) - 1 for root in roots))
        if (
            len(result) > self.degree
            or any(j < 0 or j >= self.dimension for j in result)
            or self.encode(result) != syndrome
        ):
            raise ValueError("Syndrome is outside the promised degree ball")
        return result


@functools.lru_cache(maxsize=64)
def bch_map(dimension, degree):
    return BCHMap(dimension, degree)
