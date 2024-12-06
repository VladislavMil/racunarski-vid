import cv2
import numpy as np

# Učitavanje slike
image = cv2.imread("ulaz.png")
if image is None:
    print("Greška pri učitavanju slike.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Korak 1: Binarizacija slike
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Binarizovana slika", binary)
cv2.waitKey(0)

# Korak 2: Pronalaženje kontura
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Prikaz kontura na binarizovanoj slici
contour_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imshow("Konture", contour_image)
cv2.waitKey(0)


# Funkcija za prepoznavanje slova P (koristeći oblik konture)
def is_letter_P(contour):
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) > 5:  # Provera da li ima dovoljno tačaka za slovo
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.3 < aspect_ratio < 0.7:  # Provera proporcija slova "P"
            return True
    return False


# Detekcija i obeležavanje slova P
detected = False
for contour in contours:
    if is_letter_P(contour):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(
            image, (x, y), (x + w, y + h), (0, 0, 255), 2
        )  # Crveni pravougaonik
        cv2.putText(
            image, "P", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
        )
        detected = True

if not detected:
    print("Nijedno slovo 'P' nije detektovano.")

# Prikaz rezultata
cv2.imshow("Detected Letters", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sačuvaj sliku sa obeleženim slovima P
cv2.imwrite("detected_P.png", image)
