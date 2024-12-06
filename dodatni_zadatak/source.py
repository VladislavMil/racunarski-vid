import cv2
import numpy as np

# Učitavanje slike
image = cv2.imread("ulaz.png")
if image is None:
    print("Greška pri učitavanju slike.")
    exit()

# Konverzija u grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Adaptivna binarizacija
binary = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# Pronalaženje kontura
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Funkcija za prepoznavanje slova P
def is_letter_P(contour):
    # Oblast konture
    area = cv2.contourArea(contour)
    if area < 500:  # Povećanje minimalne površine
        return False

    # Proporcije pravougaonika
    rect = cv2.minAreaRect(contour)
    (x, y), (width, height), angle = rect
    aspect_ratio = min(width, height) / max(width, height)
    if not (0.2 < aspect_ratio < 1.5):  # Proširenje granica proporcija
        return False

    # Rotacija konture za poravnanje
    rot_mat = cv2.getRotationMatrix2D((x + width // 2, y + height // 2), angle, 1)
    rotated_binary = cv2.warpAffine(binary, rot_mat, (binary.shape[1], binary.shape[0]))

    # Pronaći konture nakon rotacije
    rotated_contours, _ = cv2.findContours(
        rotated_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Proveriti postojanje unutrašnje konture (rupa)
    for rotated_contour in rotated_contours:
        # Kreiramo masku za konturu
        mask = np.zeros(rotated_binary.shape, dtype=np.uint8)
        cv2.drawContours(mask, [rotated_contour], -1, 255, -1)

        # Pronaći unutrašnje konture
        inner_contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Provera da li postoji rupa unutar konture (prostor sa 0 vrednostima)
        if len(inner_contours) > 0:
            return True

    return False


# Detekcija i obeležavanje slova P
detected = False
for contour in contours:
    if is_letter_P(contour):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            image, "P", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
        )
        detected = True

# Debug: Prikaz svih kontura radi provere
debug_image = image.copy()
cv2.drawContours(debug_image, contours, -1, (255, 0, 0), 2)
cv2.imshow("All Contours", debug_image)
cv2.waitKey(0)

if not detected:
    print("Nijedno slovo 'P' nije detektovano.")
else:
    print("Detektovano je slovo 'P'.")

# Prikaz rezultata
cv2.imshow("Detected Letters", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Čuvanje slike sa obeleženim slovima P
cv2.imwrite("detected_P.png", image)
