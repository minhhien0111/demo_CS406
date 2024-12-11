from gf2x import *
from hashlib import sha256

def BCHDecoder(F, receivedPoly, syndromePoints):
  n = len(receivedPoly)
  unreadable = []
  for i, v in enumerate(receivedPoly):
    if v == -1:
      unreadable.append(n-i-1)
      receivedPoly[i] = 0
      
  Px = Poly(F)
  s = []
  two = F(2)
  allzero = True
  for idx in syndromePoints:
    s.append(Px.evaluate(receivedPoly, two ** idx))
    allzero = allzero and (s[-1] == 0)

  if allzero:
    return receivedPoly

  s = s[::-1]
  gamma = [1]
  for v in unreadable:
    gamma = Px.mul(gamma, [two ** v, 1])
  s2 = Px.mul(s, gamma)
  xd1 = [1] + [0] * len(syndromePoints)

  qs = []
  a = s2
  b = xd1
  lim = (len(syndromePoints) + len(unreadable) - 2) // 2
  while len(b) > lim + 1:
    (q, r) = Px.fullDiv(a, b)
    a = b
    b = r
    while b[0] == 0:
      b = b[1:]
    qs.append(q)

  coef = ([1], Px.sub([0], qs[-1]))
  for q in qs[::-1][1:]:
    t = coef[0]
    c1 = coef[1]
    c2 = Px.sub(t, Px.mul(coef[1], q))
    coef = (c1, c2)

  locator = coef[0]
  fixed = len(locator) == 1 and locator[0] == 1
  for i in range(len(receivedPoly)):
    rem = Px.mod(locator, [two ** i, 1])
    if rem[0] == 0:
      receivedPoly[-i-1] ^= 1
      fixed = True

  if not fixed:
    return []

  xi = Px.mul(gamma, locator)
  omega = Px.mod(Px.mul(s, xi), xd1)
  xi.pop()
  for i in range(2, len(xi)+1, 2):
    xi[-i] = 0
  for v in unreadable:
    v2 = len(receivedPoly) - v
    t = int(Px.evaluate(omega, two ** v2) / Px.evaluate(xi, two ** v2))
    receivedPoly[v2-1] = t
    if t > 1:
      return []
  return receivedPoly

def Decoder(F, receivedPoly, syndromePoints):
  n = len(receivedPoly)
  unreadable = []
  for i, v in enumerate(receivedPoly):
    if v == -1:
      unreadable.append(n-i-1)
      receivedPoly[i] = 0
      
  Px = Poly(F)
  s = [0] * (max(syndromePoints)+1)
  two = F(2)
  allzero = True
  for idx in syndromePoints:
    s[idx] = Px.evaluate(receivedPoly, two ** idx)
    allzero = allzero and (s[idx] == 0)

  if allzero:
    return receivedPoly

  xd1 = [1] + [0] * (len(syndromePoints) + 1)
  s = s[::-1]
  gamma = [1]
  for v in unreadable:
    gamma = Px.mul(gamma, [two ** v, 1])
  s2 = Px.mod(Px.mul(s, gamma), xd1)

  qs = []
  a = s2
  b = xd1
  lim = (len(syndromePoints) + len(unreadable)) // 2
  while len(b) > lim + 1:
    (q, r) = Px.fullDiv(a, b)
    a = b
    b = r
    while b[0] == 0:
      b = b[1:]
    qs.append(q)

  if len(qs) == 0:
    return []

  coef = ([F(1)], Px.sub([0], qs[-1]))
  for q in qs[::-1][1:]:
    t = coef[0]
    c1 = coef[1]
    c2 = Px.sub(t, Px.mul(coef[1], q))
    coef = (c1, c2)

  error_location = unreadable
  locator = coef[0]
  fixed = len(locator) == 1 and locator[0] == 1
  for i in range(len(receivedPoly)):
    rem = Px.mod(locator, [two ** i, 1])
    if rem[0] == 0:
      error_location.append(i)

  if (len(error_location) == 0):
    return []

  for i in range(len(locator)):
    locator[i] /= locator[-1]

  omega = Px.mod(Px.mul(s2, locator), xd1)
  xi = Px.mul(locator, gamma)
  xi.pop()
  for i in range(2, len(xi)+1, 2):
    xi[-i] = 0

  for i in error_location:
    receivedPoly[n-i-1] ^= int(two**i * Px.evaluate(omega, two**(-i)) / Px.evaluate(xi, two**(-i)))

  return receivedPoly



if __name__ == '__main__':
  F = GF2x(8, 285)
  sp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  cw = [-1, -1, -1, -1, 55, 66, 6, 18, -1, -1, -1, -1, -1, -1, -1, -1, 6, 54, -1, -1, -1, -1, -1, -1, -1, 82, 6, 70, 246, 226, 119, 66, 6, 230, -1, -1, -1, 7, -1, -1, 7, 54, 54, 22, 226, 6, 151, 66, 194, 6, 134, 246, -1, -1, -1, 224, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17, 40, 209, 127, 134, 151, 16, 159, 107, 29, 100, 104, 220, 233, 201, 21, 121, 225, 218, 190, 112]

  # print(sha256(bytes(cw)).hexdigest()[:10])
  # print(sha256(bytes(d1)).hexdigest()[:10])
  decoded = Decoder(F, cw[:], sp)
  # print(sha256(bytes(decoded)).hexdigest()[:10])
  print(decoded)
