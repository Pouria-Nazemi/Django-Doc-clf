# myapp/views.py
from django.shortcuts import render
from django.http import HttpResponse
import os


from django.shortcuts import render
from .forms import ImageUploadForm
import pytesseract
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' # for windows

id2label = {0: 'Ahkam Ghazaei',
            1: 'Amlak',
            2: 'Asnad Malekiat',
            3: 'Bazdid',
            4: 'Ekhtarie',
            5: 'Estelam Boron Sazmani',
            6: 'Estelam Dron Sazmani',
            7: 'Estelam Senfi',
            8: 'Fish Ensheab',
            9: 'Mabaghi Komisionha',
            10: 'Mafasahesab',
            11: 'Nezam Mohandesi',
            12: 'Parvane',
            13: 'Peyvast Mohandesin Nazer',
            14: 'Sima Manzar',
            15: 'Sorat Majles',
            16: 'Tahodat Malek',
            17: 'Tarh Tafsili',
            18: 'Tatbigh va Tavafoghat',
            19: 'fish pardakhti',
            20: 'komision band 20',
            21: 'komision made 100',
            22: 'komision made 12',
            23: 'komision made 7',
            24: 'komision made 77',
            25: 'label',
            26: 'namaye mosavab',
            27: 'payankar',
            28: 'tahator'}

os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    model = AutoModelForSequenceClassification.from_pretrained("Pouria02/DocClassifier")
    # model = torch.load("DocClassifierModel.pkl")

    tokenizered = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")

except Exception as e:
    print(e)

langs = "fas"  # Languages for OCR eng+fas
def ImageToText(imageUploaded):
    img = process_image(imageUploaded)

    # Recognize the text as string in image using pytesserct
    config = '--oem 1'
    text = str(pytesseract.image_to_string(img, lang=langs, config=config))
    # Remove empty lines of text - s.strip() removes lines with spaces
    text = os.linesep.join([s for s in text.splitlines() if s.strip()])

        # Creating a text file to write the output
    return text

        # Evaluate the result based on Levenshtein distance
        # evaluate_result(text)


# def write_output(filename, text):
#     outfile = os.path.join(output_dir, f'out_{os.path.splitext(filename)[0]}.txt')
#     with open(outfile, 'w', encoding='utf-8') as text_file:
#         text_file.write(text)


import os
import re

import cv2
import numpy as np
import pytesseract
from PIL import Image


def process_image(imageUploaded):
    temp_filename = resize_image(imageUploaded)
    img = remove_noise_and_smooth(temp_filename)
    img = fix_rotation(img)
    # img = remove_lines(img)

    return img


def resize_image(imageUploaded):
    try:
        img = imageUploaded
        length_x, width_y = img.size
        factor = max(1, int(1800 / length_x))  # 1800 for tesserect
        size = factor * length_x, factor * width_y
        im_resized = img.resize(size, Image.LANCZOS)

        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".TIFF")
        temp_filename = temp_file.name
        im_resized.save(temp_filename, dpi=(300, 300))  # best for OCR

        return temp_filename
    except IOError:
        print("Error while reading the file.")


def remove_noise_and_smooth(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Applying erosion and dilation to remove the noise
        # img = cv2.bitwise_not(img)
        # kernel = np.ones((5, 5), np.uint8)
        # img = cv2.erode(img, kernel, iterations=1)
        # img = cv2.dilate(img, kernel, iterations=1)
        # show_wait_destroy('dilate', img)
        # img = cv2.bitwise_not(img)

        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        img = cv2.bitwise_or(img, closing)
        show_wait_destroy('bitwise_or', img)

        # img = apply_threshold(img, 1)
        # show_wait_destroy('threshold', img)
        # img = smooth_image(img)

        return img
    except IOError:
        print("Error while reading the file.")


def apply_threshold(img, argument):
    switcher = {
        1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        4: cv2.threshold(cv2.medianBlur(img, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        5: cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        6: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 2),
        7: cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31,
                                 2),
    }
    return switcher.get(argument, "Invalid method")


def smooth_image(img):
    # Apply blur to smooth out the edges
    blur_img = cv2.GaussianBlur(img, (1, 1), 0)
    show_wait_destroy('blur', blur_img)

    return blur_img


def fix_rotation(img):
    rotated_img = img
    # osd: orientation and script detection
    tess_data = pytesseract.image_to_osd(img, nice=1)
    angle = int(re.search(r"(?<=Rotate: )\d+", tess_data).group(0))
    # print("angle: " + str(angle))

    if angle != 0 and angle != 360:
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)

        # Perform the rotation
        rotation_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # Fixing the image cut-off by calculating the new center
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)

        rotation_mat[0, 2] += bound_w / 2 - center[0]
        rotation_mat[1, 2] += bound_h / 2 - center[1]

        rotated_img = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))

    return rotated_img


def remove_lines(img):
    # Remove lines to improve accuracy of tabular documents
    # https://stackoverflow.com/questions/33949831/whats-the-way-to-remove-all-lines-and-borders-in-imagekeep-texts-programmatic?answertab=votes#tab-top
    result = img.copy()
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255, 255, 255), 5)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255, 255, 255), 5)

    show_wait_destroy('nolines', result)

    return result


def show_wait_destroy(winname, img, active=False):
    if active:
        cv2.imshow(winname, cv2.resize(img, (960, 540)))
        cv2.moveWindow(winname, 500, 0)
        cv2.waitKey(0)
        cv2.destroyWindow(winname)


def save_image(img, img_path, method, active=False):
    if active:
        # Save the filtered image in the output directory
        filename = os.path.basename(img_path).split('.')[0]
        filename = filename.split()[0]
        save_path = os.path.join("../output", filename + "_filter_" + method + ".jpg")
        cv2.imwrite(save_path, img)

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = Image.open(request.FILES['image'])
            text = ImageToText(image)
            # Step  3: Tokenize the text
            tokens = tokenizered.encode(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

            # Step  5: Classify the text
            output = model(tokens)

            # Step  6: Interpret the output
            _, predicted = torch.max(output.logits, dim=1)


            predicted_label = id2label[predicted.item()]
            # Print the predicted class
            print(f"Predicted class: {predicted_label}")

            # Here, you would add your text classification logic
            # For simplicity, we'll just return the text
            return render(request, 'result.html', {'text': text, 'label': predicted_label})
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})
