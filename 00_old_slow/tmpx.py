class A:
    def __init__(self, x):
        self.x = x

    def __mul__(self, other):
        return self.x * other

    def __rmul__(self, other):
        return other * self.x


print(3 * A(5))
