import numpy as np
import json

def lengthSize(mode, version):
  l = [
    [10, 12, 14],
    [9, 11, 13],
    [8, 16, 16],
    [8, 10, 12]
  ]
  t = 0
  if version > 9:
    t = 1
  if version > 26:
    t = 2
  return l[mode.bit_length()-1][t]

with open('ectable.json', 'r') as f:
  ectable = json.load(f)

def getECEntry(version, error_level):
  idx = (version - 1) * 4 + error_level
  return ectable[idx]

maskFunc = [
  lambda i, j: (i+j) % 2 == 0,
  lambda i, j: i % 2 == 0,
  lambda i, j: j % 3 == 0,
  lambda i, j: (i+j) % 3 == 0,
  lambda i, j: (i // 2 + j // 3) % 2 == 0,
  lambda i, j: (i*j) % 2 + (i*j) % 3 == 0,
  lambda i, j: ((i*j)%3 + i*j) % 2 == 0,
  lambda i, j: ((i*j)%3 + i+j) % 2 == 0
]

def getSize(version):
  return 4*version + 17

def alignmentPosition(version):
  size = getSize(version)
  if version == 1:
    return []
  l = 6
  r = size-7
  d = r-l
  if version < 7:
    return [l, r]
  if version < 14:
    return [l, (l+r)//2, r]
  k = 3
  if version > 20:
    k = 4
  if version > 27:
    k = 5
  if version > 34:
    k = 6
  d2 = d + (-d) % (2*k)
  v = d2 // k
  if version == 32:
    v = 26
  out = [l]
  for i in range(k-1, -1, -1):
    out.append(r - i*v)
  return out

def getNextCell(size, r, c):
  c2 = c-1 if c > 6 else c
  backOnly = c2 % 2 == 1
  if backOnly:
    return (r, c-1)
  if c2 % 4 == 2:
    if r == 0:
      if c2 == 6:
        return (r, 5)
      return (r, c-1)
    return (r-1, c+1)
  if r == size-1:
    return (r, c-1)
  return (r+1, c+1)

def generateMask(size, maskId):
  t = np.zeros((size, size), dtype=np.int8)
  for i in range(size):
    for j in range(size):
      t[i][j] = int(maskFunc[maskId](i, j))
  return t