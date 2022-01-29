import numpy as np
import os
import pandas as pd
import modules.feature_extraction as fe
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, log_loss

import DecisionTree as dt

def load_images(path):
    relImgPath = path

    imgPaths = []
    for dirpath, dirnames, filenames in os.walk(relImgPath):
        if dirnames:
            classes = {}
            for index, name in enumerate(dirnames):
                classes[name]=index
        for filename in filenames:# [f for f in filenames if f.endswith(suportedImgFomats)]:
            imgPaths.append(os.path.join(dirpath, filename))
    
    return (imgPaths, classes)

def extract_features(path):
    relImgPath = path

    imgPaths = []
    for dirpath, dirnames, filenames in os.walk(relImgPath):
        if dirnames:
            classes = {}
            for index, name in enumerate(dirnames):
                classes[name]=index
        for filename in filenames:# [f for f in filenames if f.endswith(suportedImgFomats)]:
            imgPaths.append(os.path.join(dirpath, filename))

    # TODO delete SPLITIT folder
    print("Class names and indices:", classes)

    features = ["Relative Image Path",
            "Class Name",
            "Class Index",
            "Aspect Ratio",
            "Number of Corners (Harris)",
            "Number of Corners (Shi-Tomasi)",
            "Perimeter Area Ratio"]

    df = pd.DataFrame(columns=features)

    for i, path in enumerate(imgPaths):
        c = path.split(os.sep)[-2]
        img = fe.prepared_image(path)
        ratio = fe.aspect_ratio(img)#path) 
        numCornersH = fe.num_corners(img, detector="harris")#fe.harris_corner_detection(img)#,path)#path)
        numCornersST = fe.num_corners(img, detector="shi-tomasi")#fe.shi_tomasi_corner_detection(img)#path)
        ratioPerim = fe.perimeter_area_ratio(img)
        
        row = pd.Series([path, c, classes[c], ratio, numCornersH, numCornersST, ratioPerim], index = features)
        df = df.append(row, ignore_index=True)
        
        #for imgName in ["original", "prepared", "canny", "canny closed gaps", 
        #                "contours", "max area contour", "harris", "shi-tomasi"]:
        #    fe.display_image(fe.images[imgName], title=imgName)
        
        print(str(format((100./len(imgPaths))*i, ".2f"))+" %", end="\r")

    return df

def process_feature_dataframe(df):
    df = df.reindex(columns=["Aspect Ratio", "Number of Corners (Harris)",
           "Number of Corners (Shi-Tomasi)",
           "Perimeter Area Ratio", "Class Index"])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df[df.iloc[:, -1] < 3]
    df = df.sample(frac=1)

    return df

def get_scores(y_test, y_pred, y_proba=None):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    loss = log_loss(y_test, y_proba) if y_proba != None else -1
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return f"Accuracy:\t{accuracy}\nPrecision:\t{precision}\nLoss:\t\t{loss}\nRecall:\t\t{recall}\nF1-Score:\t{f1}"
