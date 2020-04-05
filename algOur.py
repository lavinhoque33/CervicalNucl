import cv2 as cv
import os, sys
from os import listdir
import imutils
import numpy as np
from scipy import signal

def getNucleus(mask):
    cntrs = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cntrs = imutils.grab_contours(cntrs)
    isl = []
    for i in range(len(cntrs)):
        c = cntrs[i]
        isl.append([])
        tmpImg = np.zeros_like(mask)
        cv.drawContours(tmpImg, cntrs, i, color=255, thickness=-1)
        pts = np.where(tmpImg == 255)
        for j in range(pts[0].shape[0]):
            isl[i].append((pts[0][j], pts[1][j]))
    return isl


def isafe2(img, i, j, avg, visited):
    return (i >= 0 and i < img.shape[0] and j >= 0 and j < img.shape[1] and not visited[i][j] and
            abs(avg - img[i][j]) < ranges)


def DFS2(img, i, j, avg, visited, mask, islands, index, neighbs):
    rowN = [-1, -1, -1, 0, 0, 1, 1, 1]
    colN = [-1, 0, 1, -1, 1, -1, 0, 1]
    for k in range(8):
        if isafe2(img, i + rowN[k], j + colN[k], avg, visited):
            mask[i + rowN[k]][j + colN[k]] = 255
            islands[index].append((i + rowN[k], j + colN[k]))
            visited[i + rowN[k]][j + colN[k]] = True
            neighbs.append([[j + colN[k], i + rowN[k]]])
    del neighbs[0]


def recoverNucl(img, mask):
    cntrs = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cntrs = imutils.grab_contours(cntrs)
    indexes = []
    for i in range(len(cntrs)):
        c = cntrs[i]
        if (len(c) < 5 or len(c) > 70):
            indexes.append(i)
            continue
        area = cv.contourArea(c)
        hull = cv.convexHull(c)
        hullarea = cv.contourArea(hull)
        if(hullarea == 0):
            solidity = 0
        else:
            solidity = area / float(hullarea)
        # print(i, "Solidity: ", solidity)
        if (solidity < 0.8):
            indexes.append(i)
    for i in reversed(indexes):
        del cntrs[i]
    avgInt = [0 for m in range(len(cntrs))]
    isl = []
    for i in range(len(cntrs)):
        c = cntrs[i]
        isl.append([])
        tmpImg = np.zeros_like(mask)
        cv.drawContours(tmpImg, cntrs, i, color=255, thickness=-1)
        pts = np.where(tmpImg == 255)
        for j in range(pts[0].shape[0]):
            isl[i].append((pts[0][j], pts[1][j]))
        pixInts = []
        pixInts.append(img[pts[0], pts[1]])
        avgInt[i] = np.mean(pixInts)

    visited = [[False for j in range(mask.shape[1])] for i in range(mask.shape[0])]
    for i in range(len(isl)):
        for j in range(len(isl[i])):
            visited[isl[i][j][0]][isl[i][j][1]] = True
    delInd = []
    for i in range(len(cntrs)):
        c = cntrs[i].tolist()
        tImg = np.zeros_like(mask)
        cv.drawContours(tImg, cntrs, i, color=255, thickness=-1)
        j = 0
        while(len(c)>0):
            tIsl = isl[i].copy()
            # tImg = np.zeros_like(mask)
            # cv.drawContours(tImg, cntrs, i, color=255, thickness=-1)
            tImCopy = tImg.copy()
            DFS2(img, c[j][0][1], c[j][0][0], avgInt[i], visited, tImg, isl, i, c)
            recd = cv.findContours(tImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            recd = imutils.grab_contours(recd)
            area = cv.contourArea(recd[0])
            hull = cv.convexHull(recd[0])
            hullarea = cv.contourArea(hull)
            if (hullarea == 0):
                solidity = 0
            else:
                solidity = area / float(hullarea)
            if(solidity >= max_solid):
                cntrs[i] = recd[0]
            else:
                isl[i] = tIsl
                break
        if(cntrs[i].shape[0]>=40):
            delInd.append(i)
    for i in reversed(delInd):
        del cntrs[i]
    cimg = np.zeros_like(mask)
    for (i,c) in enumerate(cntrs):
        cv.drawContours(cimg, cntrs, i, color=255, thickness=-1)
    return cimg, isl

# Parameters
th = 110
ranges = 50
max_solid = 0.75

file = open("ResultOur.txt", "a+")
file.write("\n\n")
total_score = 0
total_precision = 0
total_recall = 0
ind = 0
basepath = 'Data'
for f in listdir(basepath + '/segments'):
    ind += 1
    img = cv.imread(os.path.join(basepath, 'segments', f))
    fin2 = cv.imread(os.path.join(basepath, 'masks', f.split('.')[0]+'_Mask.jpg'))
    img1 = np.copy(img)
    img = img[:, :, 1]
    img2 = img
    fin2 = fin2[:, :, 0]
    reta, fin2 = cv.threshold(fin2, 127, 255, cv.THRESH_BINARY)
    # img = cv.medianBlur(img,5)
    # img = cv.GaussianBlur(img, (5, 5), 0)
    # img = cv.bilateralFilter(img,5,75,75)
    th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 35, 50)
    th3 = th2

    H = [[[5, -3, -3], [5, 0, -3], [5, -3, -3]],
         [[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],
         [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
         [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]],
         [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]],
         [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
         [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]],
         [[5, 5, -3], [5, 0, -3], [-3, -3, -3]]]

    H = np.array(H)
    gradient = []
    for i in range(8):
        grad = signal.convolve2d(th3, H[i], fillvalue=0)
        gradient.append(grad)

    gradient = np.array(gradient)
    final_grad = []

    for i in range(th3.shape[0]):
        a = []
        for j in range(th3.shape[1]):
            maxi = -9999
            for p in range(8):
                if gradient[p][i][j] > maxi:
                    maxi = gradient[p][i][j]
            a.append(maxi)
        final_grad.append(a)

    final_grad = np.array(final_grad)

    temp = th3.copy()

    for i in range(th3.shape[0]):
        for j in range(th3.shape[1]):
            if (final_grad[i][j] < 80):
                temp[i][j] = 0

    fin = th3 - temp

    fin1 = fin

    (final, nuclD) = recoverNucl(img2, fin1)
    nuclA = getNucleus(fin2)

    tp = 0
    fp = 0
    fn = 0
    i = 0
    while(len(nuclA)>0 and i<len(nuclA)):
        j = 0
        flag = 0
        flag2 = 0
        while(len(nuclD)>0 and j<len(nuclD)):
            cnt = len(set(nuclA[i]).intersection(nuclD[j]))
            if (cnt > 0):
                dice = cnt / max(len(nuclA[i]), len(nuclD[j]))
                # print(i, "Dice: ", dice)
                if (dice > 0.6):
                    tp += 1
                else:
                    fp += 1
                del nuclD[j]
                del nuclA[i]
                flag = 1
                flag2 = 1
                break
            j += 1
        if(flag2 == 0):
            fn += 1
        if(flag == 0):
            i += 1
    if (len(nuclA) < len(nuclD)):
        fp += abs(len(nuclA) - len(nuclD))
    fscore = (2 * tp / (2 * tp + fp + fn)) * 100
    if (tp == 0):
        precision = 0
        recall = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    total_score += fscore
    total_precision += precision
    total_recall += recall

    print(f, " Fscore: ", fscore, " Average: ", total_score / ind)
    file.write(str(f) + " Fscore: " + str(fscore) + " Recall: " + str(recall) + " Precision: " + str(precision) + '\n')

avg_precision = total_precision/ind
avg_recall = total_recall/ind
avg_score = 2/((1/avg_precision)+(1/avg_recall))

print("Average score: ", avg_score)
print("Average precision: ", avg_precision)
print("Average recall: ", avg_recall)

file.write("Average score: "+ str(avg_score)+"\n")
file.write("Average precision: "+ str(avg_precision)+"\n")
file.write("Average recall: "+ str(avg_recall)+"\n\n")
#file.write("Threshold: "+ str(th)+ " Range of Intensity: "+ str(ranges)+" Solidity: "+ str(max_solid) +"\n")

# cv.imwrite('mask.png',cv.cvtColor(fin1,cv.COLOR_GRAY2RGB))