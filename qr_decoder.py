import cv2
import numpy as np
from qr_utils import *
import matplotlib.pyplot as plt
from decoder import Decoder
from gf2x import *

F1 = GF2x(4, 19)
F2 = GF2x(8, 285)
versionBitStrs = []
F_version = GF2x(12, 7973)
for i in range(7, 41):
  versionBitStrs.append((i << 12) + int(F_version(i << 12)))

def decodeVersionString(vBitS):
  vInt = 0
  unknown = 0
  for v in vBitS:
    vInt <<= 1
    unknown <<= 1
    if v == -1:
      unknown += 1
    else:
      vInt += v

  d = unknown.bit_count()
  closest = None
  best_dist = None
  for i, t in enumerate(versionBitStrs):
    k = (t ^ vInt) & (~unknown)
    dist = d + k.bit_count()*2
    if dist > 7:
      continue
    if best_dist is None or dist < best_dist:
      closest = i + 7
      best_dist = dist

  return closest
  

def dataMask(version):
  size = getSize(version)
  mask = np.ones((size, size), dtype=np.uint8)
  mask[:9, :9] = mask[:9, -8:] = mask[-8:, :9] = mask[6, :] = mask[:, 6] = 0
  if version >= 7:
    mask[:6, -11:-8] = mask[-11:-8, :6] = 0

  al = alignmentPosition(version)
  for r in al:
    for c in al:
      if (r == al[0] and (c == al[0] or c == al[-1])) or (c == al[0] and r == al[-1]):
        continue
      mask[r-2:r+3, c-2:c+3] = 0

  return mask

def readingOrderArray(version):
  format1 = np.array([8]*8 + [7] + list(range(5, -1, -1))), np.array(list(range(6)) + [7] + [8]*8)
  format2 = np.array(list(range(-1, -8, -1)) + [8]*8), np.array([8]*8 + list(range(-7, 0)))
  version1 = [], []
  version2 = [], []
  if version >= 7:
    for i in range(6):
      for j in range(-11, -8):
        version1[0].append(i)
        version1[1].append(j)
        version2[1].append(i)
        version2[0].append(j)
    version1 = version1[0][::-1], version1[1][::-1]
    version2 = version2[0][::-1], version2[1][::-1]

  data = [], []
  mask = dataMask(version)
  size = getSize(version)
  r, c = size-1, size-1
  while c >= 0:
    if mask[r][c] == 1:
      data[0].append(r)
      data[1].append(c)
    r, c = getNextCell(size, r, c)

  return format1, format2, version1, version2, data, mask

def readFormat(data, t1, t2):
  mask = list(map(int, "101010000010010"))
  format1 = [-1 if v == -1 else m ^ v for m, v in zip(mask, data[t1])]
  format2 = [-1 if v == -1 else m ^ v for m, v in zip(mask, data[t2])]
  syndromes = list(range(1, 7))
  Decoder(F1, format1, syndromes)
  Decoder(F1, format2, syndromes)
  out = []
  if len(format1) > 0:
    out = format1[:5]
  elif len(format2) > 0:
    out = format2[:5]

  if len(out) == 0:
    return None, None

  error_level = out[0]*2 + (1 - out[1])
  maskIdx = out[2]*4 + out[3]*2 + out[4]
  return error_level, maskIdx


def decodeBinary(length, bitStr):
  out = []
  for i in range(length):
    out.append(int(bitStr[i*8:i*8+8], 2))
  return bytes(out)

def decodeNumeric(length, bitStr):
  offset = 0
  out = ""
  for i in range(0, length-2, 3):
    out += str(int(bitStr[offset:offset+10], 2)).zfill(3)
    offset += 10
    
  if length % 3 == 1:
    out += str(int(bitStr[offset:offset+4], 2))
  elif length % 3 == 2:
    out += str(int(bitStr[offset:offset+7], 2)).zfill(2)
  return out

def readVersion(data, v1, v2):
  if len(v1[0]) == 0:
    return 0

  vers1 = list(data[v1])
  vers2 = list(data[v2])
  v1 = decodeVersionString(vers1)
  v2 = decodeVersionString(vers2)
  if v1 is not None and v2 is not None:
    if v1 == v2:
      return v1
    else:
      return None
  if v1 is None:
    return v2
  if v2 is None:
    return v1
  return None

def decodeCodewords(codewords, version, error_level):
  CwD, ECz, nB1, Cw1, nB2, Cw2 = getECEntry(version, error_level)
  group_desc = [(nB1, Cw1)]
  if nB2 > 0:
    group_desc.append((nB2, Cw2))

  nBlocks = sum([t[0] for t in group_desc])
  ecPart = codewords[-nBlocks * ECz:]
  dataPart = codewords[:-nBlocks * ECz]

  dataBlocks = [[] for _ in range(nBlocks)]
  ecBlocks = [[] for _ in range(nBlocks)]

  assert (len(dataPart) - nB2) % nBlocks == 0

  for i in range(len(dataPart) - nB2):
    dataBlocks[i % nBlocks].append(dataPart[i])
  for i in range(nB2):
    dataBlocks[-i-1].append(dataPart[-i-1])

  for i in range(len(ecPart)):
    ecBlocks[i % nBlocks].append(ecPart[i])

  syndromes = list(range(ECz))
  dataOut = []
  for i in range(nBlocks):
    res = Decoder(F2, dataBlocks[i] + ecBlocks[i], syndromes)
    if len(res) == 0:
      return None
    dataOut += res[:-ECz]

  bitStr = ''.join([bin(t)[2:].zfill(8) for t in dataOut])
  mode = int(bitStr[:4], 2)
  lz = lengthSize(mode, version)
  length = int(bitStr[4:4+lz], 2)
  bitStr = bitStr[4+lz:]

  if mode == 4:
    return decodeBinary(length, bitStr)
  if mode == 1:
    return decodeNumeric(length, bitStr)

  return mode, length, dataOut


def rawQRParse(data):
  version = (data.shape[0] - 17) // 4
  size = data.shape[0]
  t1, t2, v1, v2, d, m = readingOrderArray(version)
  el, maskId = readFormat(data, t1, t2)
  if el is None:
    return None
  vers = readVersion(data, v1, v2)
  if vers is None or (vers != 0 and vers != version):
    return None
  
  flipMask = generateMask(size, maskId)
  for i in range(size):
    for j in range(size):
      if m[i][j] == 0:
        continue
      if flipMask[i][j] == 1 and data[i][j] != -1:
        data[i][j] ^= 1

  bitStr = data[d]
  codewords = []
  for i in range(0, len(bitStr)-7, 8):
    segment = bitStr[i:i+8]
    if -1 not in segment:
      codewords.append(int(''.join(map(str, segment)), 2))
    else:
      codewords.append(-1)

  # print(codewords, version, el)
  return decodeCodewords(codewords, version, el)



# import matplotlib.pyplot as plt
# def printMask(img, invert=False):
#   plt.imshow(img * (-1 if invert else 1), cmap='gray')
#   plt.show()


# img = cv2.imread('test_qr.png')
# unit = 10
# h, w, _ = img.shape
# raw_data = [[0] * (w // unit) for _ in range(h//unit)]
# for i in range(h // unit):
#   for j in range(w // unit):
#     c = img[i*unit+4][j*unit+4][0]
#     if c == 255:
#       raw_data[i][j] = 0
#     elif c == 0:
#       raw_data[i][j] = 1
#     else:
#       raw_data[i][j] = -1


# raw_data = np.array(raw_data, np.int8)
# printMask(raw_data, True)
# print(rawQRParse(raw_data))
