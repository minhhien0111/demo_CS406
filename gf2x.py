


class GF2x:
  def precalculate(self):
    self.exp = [1]
    self.log = [0] * self.modulus
    for i in range(1, self.modulus-1):
      v = self.exp[-1] * 2
      if v > self.modulus-1:
        v ^= self.polyMod
      self.exp.append(v)
      self.log[v] = i

  def __init__(self, power, polyMod):
    self.power = power
    self.modulus = 1 << power
    self.polyMod = polyMod
    self.precalculate()

  def mul(self, a, b):
    if a == 0 or b == 0:
      return self(0)
    return self(self.exp[(self.log[a] + self.log[b]) % (self.modulus - 1)])

  def div(self, a, b):
    if a == 0:
      return self(0)
    return self(self.exp[(self.log[a] - self.log[b]) % (self.modulus - 1)])

  def fromInt(self, val):
    while True:
      t = len(bin(val)[2:])
      if t < self.power+1:
        break
      val ^= self.polyMod << (t - (self.power+1))
    return val

  def __call__(self, v):
    if isinstance(v, GF2Inst):
      return v
    return GF2Inst(v, self)


class GF2Inst:
  def __init__(self, value, gf):
    self.gf = gf
    self.value = gf.fromInt(value)

  def _samefield(calc):
    def wrapper(self, other):
      if self.gf != other.gf:
        raise "Both side must be in the same field"
        return None
      else:
        return calc(self, other)
    return wrapper

  @_samefield
  def __add__(self, other):
    return GF2Inst(self.value ^ other.value, self.gf)

  @_samefield
  def __sub__(self, other):
    return GF2Inst(self.value ^ other.value, self.gf)

  @_samefield
  def __mul__(self, other):
    return self.gf.mul(self.value, other.value)

  @_samefield
  def __truediv__(self, other):
    return self.gf.div(self.value, other.value)

  def __int__(self):
    return self.value

  def __repr__(self):
    return str(self.value)

  def __pow__(self, v):
    if self.value <= 1 or v == 1:
      return self
    if v == 0:
      return self.gf(1)
    return self.gf(self.gf.exp[(self.gf.log[self.value] * v) % (self.gf.modulus - 1)])

  def __eq__(self, v):
    return self.value == self.gf(v).value


class Poly:
  def __init__(self, field):
    self.gf = field

  def _intoField(calc):
    def wrapper(self, poly1, poly2):
      p1 = [self.gf(v) for v in poly1]
      p2 = [self.gf(v) for v in poly2]
      return calc(self, p1, p2)
    return wrapper

  @_intoField
  def add(self, p1, p2):
    if len(p1) < len(p2):
      p1, p2 = p2, p1
    for i in range(len(p2)):
      p1[-i-1] += p2[-i-1]
    return p1

  @_intoField
  def sub(self, p1, p2):
    if len(p1) < len(p2):
      p1 = [self.gf(0) for _ in range(len(p2) - len(p1))] + p1
    for i in range(len(p2)):
      p1[-i-1] -= p2[-i-1]
    return p1

  @_intoField
  def mul(self, poly1, poly2):
    coef = [self.gf(0) for i in range(len(poly1) + len(poly2) - 1)]
    deg1 = len(poly1)
    deg2 = len(poly2)
    for d1, c1 in enumerate(poly1):
      d1 = deg1 - d1 - 1
      for d2, c2 in enumerate(poly2):
        d2 = deg2 - d2 - 1
        coef[d1+d2] += c1*c2
    return coef[::-1]

  @_intoField
  def fullDiv(self, poly1, poly2):
    deg1 = len(poly1)
    deg2 = len(poly2)
    if deg1 < deg2:
      return ([0], poly1)

    quot = []
    for i in range(0, deg1-deg2+1):
      qc = poly1[i] / poly2[0]
      quot.append(qc)
      for j in range(deg2):
        poly1[i+j] += poly2[j] * qc
    return quot, poly1[-deg2+1:]

  @_intoField
  def mod(self, poly1, poly2):
    deg1 = len(poly1)
    deg2 = len(poly2)
    if deg1 < deg2:
      return poly1

    for i in range(0, deg1-deg2+1):
      qc = poly1[i] / poly2[0]
      for j in range(deg2):
        poly1[i+j] += poly2[j] * qc
    return poly1[-deg2+1:]

  def generator(self, size):
    out = [1, 1]
    for i in range(1, size):
      out = self.mul(out, [1, 1 << i])
    return out

  def evaluate(self, poly, x):
    p = [self.gf(v) for v in poly]
    x = self.gf(x)
    v = p[0]
    for c in p[1:]:
      v = v * x + c
    return v


if __name__ == '__main__':
  F = GF2x(4, 19)
  Px = Poly(F)
