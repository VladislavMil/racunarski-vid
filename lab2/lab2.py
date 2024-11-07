import numpy as np
import cv2 as cv

# Učitavamo sliku
image = cv.imread('coins.png')

# Prebacujemo sliku u HSV prostor boja
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Fino podešeni opseg boja za bakarni novčić
# Pokušavamo sa svetlijim opsegom i povećanom saturacijom i vrednostima
lower_bronze = np.array([1, 60, 60])  # Donji prag za bakarnu boju
upper_bronze = np.array([25, 255, 255])  # Gornji prag za bakarnu boju

# Kreiramo masku
mask = cv.inRange(hsv, lower_bronze, upper_bronze)

# Dodatno obrađujemo masku za bolje rezultate
# Koristimo "closing" operaciju za spajanje malih regiona i uklanjanje rupa
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

# Primena "dilate" operacije da se popune eventualne rupe
mask = cv.dilate(mask, kernel, iterations = 1)

# Prikaz maske za bakarni novčić
cv.imshow('Maska', mask)
cv.imwrite("maska.png", mask)

# Primena maske na originalnu sliku
result = cv.bitwise_and(image, image, mask=mask)
cv.imshow('Izdvojen', result)
cv.imwrite("resultat.png", result)

cv.waitKey(0)
cv.destroyAllWindows()
