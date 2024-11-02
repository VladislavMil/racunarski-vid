import cv2
import numpy as np
import matplotlib.pyplot as plt

def crniKvadrati(matrix, x, y):
    for i in range(-2, 2):
        for j in range(-2, 2):
            matrix[x + i, y + j] = 0

img = cv2.imread('slika.png', 0)

f = np.fft.fft2(img)

fshift = np.fft.fftshift(f)

magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnituda spektra pre uklanjanja šuma')
plt.savefig('fft_mag.png')
plt.show()

x1, y1 = 230, 230
x2, y2 = 280, 280
x3, y3 = 356, 156
x4, y4 = 156, 356

crniKvadrati(fshift, x1, y1)
crniKvadrati(fshift, x2, y2)
crniKvadrati(fshift, x3, y3)
crniKvadrati(fshift, x4, y4)

magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnituda spektra nakon uklanjanja šuma')
plt.savefig('fft_mag_filtered.png')
plt.show()

f_ishift = np.fft.ifftshift(fshift)

img_filtered = np.fft.ifft2(f_ishift).real

cv2.imshow('Final image', img_filtered.astype(np.uint8))
cv2.imwrite('output.png', img_filtered)