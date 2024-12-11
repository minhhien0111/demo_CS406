import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import choice
import quads
from tqdm import tqdm
from scipy.spatial import ConvexHull
import os

colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

def debugLines(im, lines, filebasename, suffix):
  imt = im*1
  for line in lines:
    x1,y1,x2,y2 = line
    cv2.line(imt,(x1,y1),(x2,y2),choice(colors),2)
  cv2.imwrite(f'logs/{filebasename}_{suffix}.png', imt)
  del imt

def debugPoly(im, polylines, filebasename, suffix):
  imt = im*1
  for i, polyline in enumerate(polylines):
    cv2.polylines(imt,[np.array(polyline, dtype=np.int32).reshape(-1, 1, 2)], True, colors[i % len(colors)],2)
  cv2.imwrite(f'logs/{filebasename}_{suffix}.png', imt)
  del imt

def debugFinder(im, polylines, finderIdx, filebasename, suffix):
  imt = im*1
  for i, pattern in enumerate(finderIdx):
    for idx in pattern:
      cv2.polylines(imt,[polylines[idx].reshape(-1, 1, 2).astype(int)], True, colors[i % len(colors)],2)
  cv2.imwrite(f'logs/{filebasename}_{suffix}.png', imt)
  del imt

def combine_lines(lines, container_size, radius=10, verbose=0):
  angles = []
  h, w = container_size
  qt = quads.QuadTree((w/2, h/2), w, h)
  available = set(range(len(lines)))
  mapper = {}

  it = enumerate(lines)
  if verbose >= 1:
    it = tqdm(it, total=len(lines))

  for i, line in it:
    x1, y1 = line[:2]
    x2, y2 = line[-2:]
    if (x1, y1) not in mapper:
      mapper[(x1, y1)] = []
    if (x2, y2) not in mapper:
      mapper[(x2, y2)] = []
    mapper[(x1, y1)].append(i)
    mapper[(x2, y2)].append(i)
    if (x1, y1) not in qt:
      qt.insert((x1, y1))
    if (x2, y2) not in qt:
      qt.insert((x2, y2))

  output = []
  for i, line in enumerate(lines):
    if i not in available:
      continue
    available.remove(i)
    x, y = line[-2:]
    m = 0
    while True:
      npx = qt.nearest_neighbors((x, y), count=10)
      done = True
      for p in npx:
        nx = p.x
        ny = p.y
        d = np.linalg.norm([nx-x, ny-y])
        if d > radius:
          break
        indices = mapper[(nx, ny)]
        for idx in indices:
          if idx in available:
            done = False
            break
        else:
          continue
        break

      if done:
        break

      lineTo = lines[idx]
      n1 = np.linalg.norm([x - lineTo[0], y - lineTo[1]])
      n2 = np.linalg.norm([x - lineTo[-2], y - lineTo[-1]])
      if n2 < n1:
        lineTo = (np.array(lineTo).reshape(-1, 2)[::-1]).flatten().tolist()
      line.extend(lineTo)
      available.remove(idx)
      x, y = line[-2:]
      m += 1
      if m > 100000:
        print("Something probably went wrong")
        break

    output.append(line)

  return output

def areLine(points):
  norm = np.linalg.norm
  d = sum(norm(points[1:] - points[:-1], axis=1)) - norm(points[0] - points[-1])
  return abs(d) < 0.01

def combine_points(points):
  mapper = {}
  available = set(range(len(points)))
  for i, point in enumerate(points):
    tp = tuple(point)
    if tp not in mapper:
      mapper[tp] = i
    else:
      available.remove(i)

  pl = []
  output = []
  for i in range(len(points)):
    point_chains = []
    if i not in available:
      continue

    stack = [i]
    while len(stack) > 0:
      idx = stack.pop()
      if idx not in available:
        continue
      available.remove(idx)
      point_chains.append(idx)
      x, y = points[idx]
      for i in range(-1, 2):
        for j in range(-1, 2):
          if i == 0 and j == 0:
            continue
          npt = (x+i, y+j)
          if npt not in mapper:
            continue
          if mapper[npt] not in available:
            continue
          stack.append(mapper[npt])
    points_list = points[point_chains].tolist()
    
    pl = np.array(sorted(points_list))
    st = [pl]
    while len(st) > 0:
      p = st.pop()
      if len(p) == 2 or areLine(p):
        output.append(np.concatenate((p[0], p[-1])))
      else:
        l = len(p) // 2
        st.append(p[:l+1])
        st.append(p[l:])

  return output

def lineEq(p1, p2):
  ''' return ax + by + c = 0 '''
  x1, y1 = p1
  x2, y2 = p2
  if x1 == np.inf:
    return [y1[0], y1[1], -y1[0]*x2-y1[1]*y2]
  if x2 == np.inf:
    return [y2[0], y2[1], -y2[0]*x1-y2[1]*y1]
  return [y2-y1, x1-x2, -x1*(y2-y1) + y1*(x2-x1)]

def parallelLine(eq, point):
  a, b, _ = eq
  x, y = point
  return [a, b, -a*x-b*y]

def intersection(eq1, eq2):
  if eq1[0] * eq2[1] == eq1[1] * eq2[0]:
    return (np.inf, eq1[:2])
  A = np.array([[eq1[0], eq1[1]], [eq2[0], eq2[1]]])
  b = np.array([[-eq1[2]], [-eq2[2]]])
  return np.array((np.linalg.inv(A)@b).squeeze(axis=1))

def polyNormalize(poly):
  try:
    hull = ConvexHull(poly)
  except:
    return [], np.inf

  hm = poly[hull.vertices]
  keep = []
  l = len(hm)
  for i, p in enumerate(hm):
    prv = hm[(i-1) % l]
    nxt = hm[(i+1) % l]
    a = np.linalg.norm(p - prv)
    b = np.linalg.norm(p - nxt)
    c = np.linalg.norm(nxt - prv)
    theta = np.arccos((a*a + b*b - c*c)/(2*a*b))
    dt = abs(theta - np.pi)
    if dt > 0.1:
      keep.append(p)

  if len(keep) < 4:
    return [], np.inf

  keep = np.array(keep)
  lines = []
  l = []
  n = len(keep)
  for i in range(n):
    prv = keep[i]
    nxt = keep[(i+1)%n]
    l.append(np.linalg.norm(prv-nxt))

  idx = sorted(np.argsort(l)[-4:])
  for i in idx:
    prv = keep[i]
    nxt = keep[(i+1)%n]
    lines.append(lineEq(prv, nxt))

  final = []
  for i in range(4):
    l1 = lines[i]
    l2 = lines[(i+1)%4]
    its = intersection(l1, l2)
    if its[0] == np.inf:
      return [], np.inf
    final.append(its)
  final = np.array(final)

  segments = []
  for i in range(4):
    j = (i+1) % 4
    segments.append(np.linalg.norm(final[i] - final[j]))

  return np.array(final), np.std(segments)

def triangleArea(triangle):
  p1, p2, p3 = triangle
  a = np.linalg.norm(p1 - p2)
  b = np.linalg.norm(p1 - p3)
  c = np.linalg.norm(p3 - p2)
  p = (a + b+ c) / 2
  if p-a < 0 or p-b < 0 or p-c < 0:
    # print(triangle)
    return 0
  return np.sqrt(p*(p-a)*(p-b)*(p-c))

def quadArea(poly):
  poly = list(poly)
  return triangleArea(poly[:3]) + triangleArea(poly[2:] + poly[:1])

def sumAreas(poly, point):
  areas = 0
  for i in range(4):
    areas += triangleArea([point, poly[i], poly[(i+1)%4]])
  return areas

def quadInQuad(bigger, smaller, area=None):
  if area is None:
    area = quadArea(bigger)
  for point in smaller:
    if abs(sumAreas(bigger, point) - area) > 1e-4:
      return False
  return True

def findFinderPattern(quads):
  areas = np.array([quadArea(poly) for poly in quads])
  l = len(areas)
  bigFinders = []
  smolFinders = {}
  for i in range(l):
    for j in range(l):
      if i == j:
        continue
      if not quadInQuad(quads[i], quads[j], areas[i]):
        continue
      smolFinders[i] = j

  for i in range(l):
    for j in range(l):
      if i == j:
        continue
      if not quadInQuad(quads[i], quads[j], areas[i]):
        continue
      if j in smolFinders:
        bigFinders.append((i, j, smolFinders[j]))

  return bigFinders

def detectFinderPattern(filename, verbose=0):
  if not os.path.exists("logs") and verbose == 2:
    os.mkdir('logs')
  filebasename = filename.replace('\\', '/').split('/')[-1].split('.')[:-1]
  filebasename = '.'.join(filebasename)

  im = cv2.imread(filename)
  if im.shape[0] < 300:
    im = cv2.resize(im, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
  im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  im_canny = cv2.Canny(im_gray, 50, 200)
  if verbose == 2:
    cv2.imwrite(f'logs/{filebasename}_0-canny.jpg', im_canny)

  lines = cv2.HoughLinesP(im_canny, 1,np.pi/180,1)
  lines = lines.squeeze(axis=1)
  if verbose == 2:
    debugLines(im, lines, filebasename, '1-hough')

  points = []
  alines = []
  for line in lines:
    norm = np.linalg.norm(line[:2]-line[2:])
    if abs(norm - 0) < 1e-9:
      points.append(line[:2])
    else:
      alines.append(line)
  if verbose == 2:
    debugPoly(im, alines, filebasename, '2-lines')

  points = np.array(points)
  out = combine_points(points)
  alines.extend(out)
  alines = np.array(alines)
  if verbose == 2:
    debugPoly(im, alines, filebasename, '3-lines')

  alines = alines.tolist()
  out = combine_lines(alines, im.shape[:2], radius=5)
  loops = []
  opens = []
  for poly in out:
    if np.linalg.norm([poly[0] - poly[-2], poly[1] - poly[-1]]) < 5:
      loops.append(poly)
    else:
      opens.append(poly)
  out = combine_lines(opens, im.shape[:2], radius=7)
  out.extend(loops)
  if verbose == 2:
    debugPoly(im, out, filebasename, '4-combined')

  reduced = []
  stds = []
  for poly in out:
    m = np.array(poly).reshape((-1, 2))
    outShape, std = polyNormalize(m)
    if len(outShape) != 4:
      continue
    reduced.append(outShape)
    stds.append(std)

  reduced = np.array(reduced)
  stds = np.array(stds)
  accepted = np.where((stds < 10))[0]
  reduced = reduced[accepted]
  if verbose == 2:
    debugPoly(im, reduced, filebasename, '5-reduced')

  finders = findFinderPattern(reduced)
  outputFinders = []
  for finder in finders:
    group = []
    for idx in finder:
      group.append(reduced[idx])
    outputFinders.append(group)

  if verbose == 2:
    debugFinder(im, reduced, finders, filebasename, '6-final')
  return im, np.array(outputFinders)

def findPoints(PaI, A, B, t):
  if PaI[0] == np.inf:
    return (A[:, None]*(1-t) + B[:, None]*t).T

  PA = np.linalg.norm(PaI - A)
  AB = np.linalg.norm(A - B)
  out = []
  squeeze = False
  if type(t) != list and type(t) != np.ndarray:
    t = [t]
    squeeze = True
  if type(t) != np.ndarray:
    t = np.array(t)

  if np.dot(B-A, PaI-A) < 0:
    r = -t*AB / (PA + (1-t)*AB)
  else:
    r = t*AB / (PA - (1-t)*AB)
  out = (PaI - A)[None, :]*r[:, None] + A
  return out[0] if squeeze else out

def perspective_points(points, resolution):
  lines = [lineEq(points[i], points[(i+1)%4]) for i in range(4)]
  PaIs = [intersection(*lines[::2]), intersection(*lines[1::2])]
  points = [np.array(point) for point in points]
  samples = np.linspace(0, 1, resolution)
  out = []
  left = findPoints(PaIs[1], points[0], points[3], samples)
  right = findPoints(PaIs[1], points[1], points[2], samples)
  for l, r in zip(left, right):
    out.append(findPoints(PaIs[0], l, r, samples))
  return np.array(out)

def ratio(P, A, E, B):
  EA = np.linalg.norm(E - A)
  BA = np.linalg.norm(B - A)
  if P[0] == np.inf:
    return EA / BA
  PB = np.linalg.norm(P - B)
  PE = np.linalg.norm(P - E)
  return (EA * PB) / (BA * PE)

def reverseMap(points, refPoint):
  lines = [lineEq(points[i], points[(i+1)%4]) for i in range(4)]
  PaIs = [intersection(*lines[::2]), intersection(*lines[1::2])]
  hLine = lineEq(PaIs[0], refPoint)
  vLine = lineEq(PaIs[1], refPoint)
  A1 = intersection(hLine, lines[3])
  B1 = intersection(hLine, lines[1])
  A2 = intersection(vLine, lines[0])
  B2 = intersection(vLine, lines[2])
  return ratio(PaIs[0], A1, refPoint, B1), ratio(PaIs[1], A2, refPoint, B2)

def perspective_transform(img, points, resolution):
  minigrid = perspective_points(points, 26).reshape((-1,  2))
  plt.imshow(img)
  plt.scatter(minigrid[:, 0], minigrid[:, 1])
  plt.show()

  grid = perspective_points(points, resolution)
  fl_grid = np.floor(grid).astype(int)
  unit = np.ones((resolution, resolution), dtype=int)
  ul = img[fl_grid[:,:, 1], fl_grid[:,:, 0]]
  ur = img[fl_grid[:,:, 1], fl_grid[:,:, 0]+unit]
  dl = img[fl_grid[:,:, 1]+unit, fl_grid[:,:, 0]]
  dr = img[fl_grid[:,:, 1]+unit, fl_grid[:,:, 0]+unit]
  d = grid - fl_grid
  dy = d[:,:,1:]
  dx = d[:,:,:1]
  out = (ul * (1-dy) + dl * dy) * (1-dx) + dx * ((ur * (1-dy) + dr * dy))
  out = out.astype(np.uint8)
  return out

def order_finders(finders):
  centers = finders.mean(axis=2).mean(axis=1)
  angles = []
  for finder in finders:
    for rect in finder:
      for i in range(4):
        x1, y1 = rect[i]
        x2, y2 = rect[(i+1)%4]
        angles.append(np.arctan2(y1-y2, x1-x2) % (np.pi))
        
  angles = np.array(angles)
  diffs = []
  for i in range(3):
    x1, y1 = centers[i]
    x2, y2 = centers[(i+1)%3]
    alpha = np.arctan2(y1-y2, x1-x2) % np.pi
    t = (angles - alpha) % np.pi / np.pi * 180
    diffs.append(np.abs(t - 90).min() + np.abs(t).min())
  
  idx = (np.argmax(diffs) - 1) % 3
  if idx != 0:
    centers[[0, idx]] = centers[[idx, 0]]
    finders[[0, idx]] = finders[[idx, 0]]

  a1 = np.arctan2(centers[1][1] - centers[0][1], centers[1][0] - centers[0][0])
  a2 = np.arctan2(centers[2][1] - centers[0][1], centers[2][0] - centers[0][0])
  if (a1 - a2) % (2*np.pi) < np.pi:
    centers[[1, 2]] = centers[[2, 1]]
    finders[[1, 2]] = finders[[2, 1]]

  return centers

def cossim(vec1, vec2):
  return np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)

def findAnchors(finders, centers):
  ul_center = (centers[1] + centers[2]) / 2
  outer = finders[:, 0]
  output = []
  for ring, center in zip(outer, centers):
    vec = center - ul_center
    sims = []
    for p in ring:
      vecR = p - center
      sims.append(abs(cossim(vec, vecR) - 1))
    idx = np.argmin(sims)
    output.append(ring[idx])
  return output

def sort_ring(ring, anchor):
  out = []
  center = ring.mean(axis=0)
  angles = []
  ref = np.arctan2(anchor[1] - center[1], anchor[0] - center[0])
  for p in ring:
    angles.append((np.arctan2(p[1] - center[1], p[0] - center[0]) - ref) % (2*np.pi))
  indices = np.argsort(angles)
  for idx in indices:
    out.append(ring[idx])
  return np.array(out)

def orderFinderRing(finder):
  ring1, ring2, ring3 = finder
  out = []
  idx = np.argmin(((ring1 * (1-ring1)) ** 2).sum(axis=1))
  out.append(sort_ring(ring1, ring1[idx]))
  idx = np.argmin(np.linalg.norm(ring2 - ring1[None, idx], axis=1))
  out.append(sort_ring(ring2, ring2[idx]))
  idx = np.argmin(np.linalg.norm(ring3 - ring2[None, idx], axis=1))
  out.append(sort_ring(ring3, ring3[idx]))
  return np.array(out)

def finderMatrix(finder):
  outer = (finder[0] + finder[1]) / 2
  middle = (finder[1] + finder[2]) / 2
  center = finder[2]

def scatter(p1, p2, pointBw):
  v = p2 - p1
  return np.linspace(0, 1, pointBw+2)[:, None] * v[None, :] + p1[None, :]

def alignmentLines(referPoints, left, right, other):
  leftP = referPoints[left]
  rightP = referPoints[right]
  lines = []
  for i in range(3):
    ringL = leftP[i]
    ringR = rightP[i]
    nextLines = [
      lineEq(ringL[0], ringR[0]),
      lineEq(ringL[1], ringR[3]) if right == 2 else lineEq(ringL[3], ringR[1])
    ]
    lines = lines[:i] + nextLines + lines[-i:]
  lines = lines[:3] + [lineEq(leftP[3][0], rightP[3][0])] + lines[-3:]

  point1, point2 = referPoints[left][0][2], referPoints[other][0][2]
  d = np.linalg.norm(point2 - point1) 
  d /= np.linalg.norm(referPoints[left][0][2] - referPoints[left][0][3 if other == 1 else 1])
  d = d * 6 - 1
  d = int(np.round((d - 3) / 4))
  scattered = scatter(point1, point2, 4*d + 3)
  for p in scattered[1:-1]:
    lines.append(parallelLine(lines[6], p))

  up = [3, 0] if other == 1 else [1, 0]
  down = [2, 1] if other == 1 else [2, 3]
  latter = []
  for i in range(3):
    ring = referPoints[other][i]
    nextLines = [parallelLine(lines[-1], ring[a]) for a in up]
    latter = latter[:i] + nextLines + latter[-i:]
  return lines + latter[:3] + [parallelLine(lines[-1], referPoints[other][-1][0])] + latter[-3:]

def solveForX(eq, y):
  a, b, c = eq
  return (b*y + c) / (-a)

def solveForY(eq, x):
  a, b, c = eq
  return (a*x + c) / (-b)

def drawLines(eq, minX=None, minY=None, maxX=None, maxY=None):
  minX = solveForX(eq, minY) if minX is None else minX
  minY = solveForY(eq, minX) if minY is None else minY
  maxX = solveForX(eq, maxY) if maxX is None else maxX
  maxY = solveForY(eq, maxX) if maxY is None else maxY
  plt.plot([minX, maxX], [minY, maxY], color='red', lw=0.5)

def qrLocator(imageName, resolution=400, verbose=False):
  im, finders = detectFinderPattern(imageName, verbose=verbose)
  centers = order_finders(finders)
  anchors = findAnchors(finders, centers)
  p1 = sort_ring(finders[1,0], anchors[1])[1]
  p2 = sort_ring(finders[2,0], anchors[2])[-1]
  l1 = lineEq(anchors[1], p1)
  l2 = lineEq(anchors[2], p2)
  anchors.insert(2, intersection(l1, l2))
  imout = perspective_transform(im, anchors, resolution)
  shape = finders.shape
  new_finders = []
  for finder in finders.reshape((-1, 2)):
    new_finders.append(reverseMap(anchors, finder))
  new_finders = np.array(new_finders).reshape(shape)
  ordered_finders = []
  for newF in new_finders:
    ordered_finders.append(orderFinderRing(newF)*resolution)
    
  referPoints = []
  for finder in ordered_finders:
    outer = (finder[0] + finder[1]) / 2
    middle = (finder[1] + finder[2]) / 2
    center = finder.reshape((-1, 2)).mean(axis=0)[None, :]
    inner = (finder[2]*2 + center) / 3
    referPoints.append([outer, middle, inner, center])

  hlines = alignmentLines(referPoints, 0, 2, 1)
  vlines = alignmentLines(referPoints, 0, 1, 2)
  return imout, hlines, vlines


def removeObscure(im, verbose=0):
  imH = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 0]
  if verbose >= 1:
    plt.imshow(imH, vmin=0, vmax=1)
    plt.show()
  histogram = np.zeros(180, dtype=int)
  unique, count = np.unique(imH, return_counts=True)
  histogram[unique] = count

  finder_pixels = np.array([imH[:30, :30], imH[:30, -30:], imH[-30:, :30]]).flatten()
  lw = np.quantile(finder_pixels, 0.25)
  up = np.quantile(finder_pixels, 0.75)
  iqr = up - lw
  lowerbound = lw-iqr*2.5
  upperbound = up+iqr*2.5
  mask = (imH - lowerbound) % 180 <= (upperbound - lowerbound) % 180
  return mask


def getQRData(im, hlines, vlines, mask, rd=3):
  imV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 2]
  # plt.imshow(imV)
  # plt.show()
  mn = int(imV.min())
  mx = int(imV.max())
  data = []
  for hline in hlines:
    row = []
    for vline in vlines:
      x, y = intersection(hline, vline)
      x = int(x)
      y = int(y)
      pV = imV[x-rd:x+rd+1, y-rd:y+rd+1].mean()
      accepted = mask[x-rd:x+rd+1, y-rd:y+rd+1].sum () > (rd*2+1)**2/2
      if not accepted:
        row.append(-1)
      elif pV < (mn + mx) / 2:
        row.append(1)
      else:
        row.append(0)
    data.append(row)
  
  return np.array(data, np.int8)