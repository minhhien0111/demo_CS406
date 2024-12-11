from qr_detector import qrLocator, removeObscure, getQRData
from qr_decoder import rawQRParse
import matplotlib.pyplot as plt

def readQR(imgName, verbose=0):
  try:
    im, hlines, vlines = qrLocator(imgName, verbose=verbose)
    mask = removeObscure(im)
    qrdata = getQRData(im, hlines, vlines, mask)

    if verbose >= 1:
      # fig, ax = plt.subplots(1, 2)
      # ax[0].imshow(im)
      # ax[0].axis('off')
      # ax[1].imshow(-qrdata, cmap='gray')
      plt.imshow(im)
      plt.axis('off')
      plt.show()
    return rawQRParse(qrdata)
  except Exception as e:
    if verbose >= 1:
      print(e)
    return "An error occurred"

# import os
# files = os.listdir('qrcode')

# for file in files:
#   print(file, readQR('qrcode/' + file))

print(readQR('img/test3.png', verbose=1))

# 7785-v4.png 5382-v3.png