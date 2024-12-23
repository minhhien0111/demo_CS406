from qr_detector import qrLocator, getQRData, removeObscure, getObscureMask
from qr_decoder import rawQRParse
import matplotlib.pyplot as plt
from tqdm import tqdm

def readQR(imgName, verbose=0):
  try:
    im, hlines, vlines = qrLocator(imgName, verbose=verbose)
    mask = getObscureMask(im, verbose)
    for i in range(10, 101, 10):
      try:
        qrdata = getQRData(im, hlines, vlines, mask <= i)
        data = rawQRParse(qrdata)
        if type(data) == bytes:
          break
      except:
        pass

    if type(data) != bytes:
      return None

    print(qrdata)

    if verbose >= 1:
      fig, ax = plt.subplots(1, 2)
      ax[0].imshow(mask <= i)
      ax[0].axis('off')
      ax[1].imshow(-qrdata, cmap='gray')
      ax[1].axis('off')
      plt.show()

    return data
  except Exception as e:
    if verbose >= 1:
      print(e)
    return "An error occurred"

print("Start")
print(readQR('img/test2.jpg', verbose=2))