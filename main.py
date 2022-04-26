from csv import reader
import os
import glob
import cv2
import pytesseract
from pytesseract import Output
import easyocr
import PIL
import numpy
import torch
from PIL import ImageDraw
from PIL import Image
from pathlib import Path
import time
from tqdm import tqdm
import pandas as pd
import matplotlib as plt

font = cv2.FONT_HERSHEY_SIMPLEX
color = (255,0,0)
# user filepaths settings. 
# current users, nijhum and lameya. 
user = "nijhum"

if(user == "lameya"):
    path = "C:/Users/HP/OCR/OCR/Tesseract/*.jpg"
    tesseract_files_path = "C:/Users/HP/OCR/OCR/Tesseract"
    easyocr_files_path = "C:/Users/HP/OCR/OCR/EasyOCR"
elif(user == "nijhum"): 
    path = "C:/Users/Asus/JP/OCRcomparison/Input/*.*"
    tesseract_files_path = "C:/Users/Asus/JP/OCRcomparison/Tesseract"
    easyocr_files_path = "C:/Users/Asus/JP/OCRcomparison/EasyOCR"
    # Tesseract executable path    
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# apply tesseract OCR and return time duration for each image
def Tess(file):

    start = time.time()
    img = cv2.imread(file, 0)
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    for i, word in enumerate(data['text']):
        if word!= "":
            x,y,w,h = data['left'][i],data['top'][i],data['width'][i],data['height'][i]
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

    image_name = Path(file).name
    f = os.path.join(tesseract_files_path, image_name)
    cv2.imwrite(f, img)
    
    end = time.time()
    duration = end - start
    return duration

# apply easyocr OCR and return time duration for each image
def EasyOCR(file):
    
    start = time.time()

    img = cv2.imread(file)
    reader = easyocr.Reader(['en'], gpu=False)
    bound = reader.readtext(file)
    for detection in bound:
        #print(detection)
        top_left = (detection[0][0])
        top_left = tuple([int(x) for x in top_left])
        bottom_right = detection[0][2]
        bottom_right = tuple([int(x) for x in bottom_right])
        text = detection[1]
        #print(text)
        img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
        img = cv2.putText(img,text, bottom_right, font, 2, color, 2 ,cv2.LINE_AA)
    

    print(bound[1])

    # image = numpy.array(im)
    image_name = Path(file).name
    f = os.path.join(easyocr_files_path, image_name)
    cv2.imwrite(f, img)

    print("")
    end = time.time()
    duration = end - start
    return duration,reader

# main function
def main():
    # lists to store time duration of performing ocr on each image
    tess_durations = []
    easy_durations = []

    # iterate and perform ocr on all the files in the input directory. 
    # append the time duration to respective list
    for file in tqdm(glob.glob(path), desc="Processing"):
        tess_duration = Tess(file)
        easyocr_duration, reader = EasyOCR(file)
        print(easy_durations)
        print(reader)
        tess_durations.append(tess_duration)
        easy_durations.append(easyocr_duration)
        break

    # save results to dataframe
    results = [["Total Duration: ", sum(tess_durations), sum(easy_durations)],
                ["Average Duration: ", sum(tess_durations)/len(tess_durations), sum(easy_durations)/len(easy_durations)], 
                ["Max Duration: ", max(tess_durations), max(easy_durations)], 
                ["Min Duration: ", min(tess_durations), min(easy_durations)]]

    df = pd.DataFrame(results,columns=['Calculation','Tesseract','EasyOCR'])
    print(df)

    # save dataframe to csv file
    df.to_csv('results.csv', index=False, header=True)

if __name__ == '__main__':
    main()  
