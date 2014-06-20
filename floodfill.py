__author__ = 'RBertram'
import numpy as np
import shapefile
from operator import itemgetter
import math
from osgeo import gdal
from osgeo.gdalconst import *
from osgeo import osr
import datetime
from scipy import spatial
import time

def uniquelist(list):
    output = []
    for x in list:
        if x not in output:
            output.append(x)
    return output

def nearestPoints(currentPoint, treeBottom, treeTop, b, t):
    currentPoint = [[currentPoint[1], currentPoint[0], currentPoint[2]]]

    distanceBottom, indicesBottom = treeBottom.query(currentPoint, k=1, p=2)
    distanceTop, indicesTop = treeTop.query(currentPoint, k=1, p=2)
    closestPoints = [b[indicesBottom[0]], t[indicesTop[0]], distanceBottom, distanceTop]
    return closestPoints


def nearestPoint(currentPoint, tM):
    distance = []
    for i in range(0, len(tM)):
        x1 = tM[i][0]
        x2 = currentPoint[1]
        y1 = tM[i][1]
        y2 = currentPoint[0]
        dist = HigherMath(x1, x2, y1, y2)
        distance.append((tM[i][1], tM[i][0], tM[i][2], dist))
    dist = min(distance, key=itemgetter(3))
    return distance.index(dist)


def HigherMath(x, y, a, b):
    NumberOne = x - y
    PowerNumber1 = math.pow(NumberOne, 2)
    NumberTwo = a - b
    PowerNumber2 = math.pow(NumberTwo, 2)
    OnePlueTwo = PowerNumber1 + PowerNumber2
    return math.sqrt(OnePlueTwo)


def WeightedEvelvation(bottom, top, currentPoint):
    distTOBottom = HigherMath(bottom[0], currentPoint[1], bottom[1], currentPoint[0])
    distToTop = HigherMath(top[0], currentPoint[1], top[1], currentPoint[0])

    big = 0
    small = 0
    closeElev = 0.00
    farElev = 0.00

    if distTOBottom > distToTop:
        big = distTOBottom
        small = distToTop
        closeElev =  top[2]
        farElev =bottom[2]

    else:
        big = distToTop
        small = distTOBottom
        closeElev =  bottom[2]
        farElev = top[2]

    weight = big+small
    if small > 0 and weight > 0:
        stdEle = ((closeElev * big) + (farElev * small))/weight
    elif weight > 0:
        stdEle = (closeElev * big) / weight

    else:
        stdEle = top[2]
    return stdEle


def floodFillIndex(img, idxBottom, idxTop, bbox, sp):
    fill = set()
    for p in sp:
        fill.add((p[0],p[1], img[p[0]][p[1]]))
    CycleCount = 0
    treeBottom = spatial.cKDTree(idxBottom)
    treeTop = spatial.cKDTree(idxTop)
    edgePlusCount = 0
    between = False
    while fill:
        y,x,z = fill.pop()
        currentPoint = [y, x, z]
        closestPoint = nearestPoints(currentPoint, treeBottom, treeTop, idxBottom, idxTop)
        bottom = [closestPoint[0][0], closestPoint[0][1], closestPoint[0][2]]
        top = [closestPoint[1][0], closestPoint[1][1], closestPoint[1][2]]
        CurrentElev = round(WeightedEvelvation(bottom, top, currentPoint), 4)
        switchMe = False
        if distanceArray[y][x] > 0:
            if (closestPoint[2] + closestPoint[3]) / 2 < distanceArray[y][x]:
                distanceArray[y][x] = (closestPoint[2] + closestPoint[3]) / 2
                switchMe = True
        else:
            distanceArray[y][x] = (closestPoint[2] + closestPoint[3]) / 2
            switchMe = True
        filled.add((y, x))
        CycleCount += 1
        print CycleCount
        edge = 0
        if bbox[2] <= x <= bbox[3] and bbox[0] <= y <= bbox[1]:
            west = [y, x - 1, img[y][x - 1]]
            WestCord = (y, x - 1)
            east = [y, x + 1, img[y][x + 1]]
            EastCord = (y, x + 1)
            north = [y + 1, x, img[y + 1][x]]
            NorthCord = (y + 1, x)
            south = [y - 1, x, img[y - 1][x]]
            SouthCord = (y - 1, x)
            if north[2] <= CurrentElev and not NorthCord in filled:
               fill.add((north[0], north[1], north[2]))
            elif north[2] > CurrentElev and not NorthCord in filled:
                edge += 1
            if west[2] <= CurrentElev and not WestCord in filled:
                fill.add((west[0], west[1], west[2]))
            elif west[2] > CurrentElev and not WestCord in filled:
                edge += 1
            if east[2] <= CurrentElev and not EastCord in filled:
                fill.add((east[0], east[1], east[2]))
            elif east[2] > CurrentElev and not EastCord in filled:
                edge += 1
            if south[2] <= CurrentElev and not SouthCord in filled:
                fill.add((south[0], south[1], south[2]))
            elif south[2] > CurrentElev and not SouthCord in filled:
                edge += 1


        if edge == 1 or edge == 2 or edge == 3 and switchMe:
            if closestPoint[2] > closestPoint[3]:
                idxBottom.append([currentPoint[1], currentPoint[0], CurrentElev])
                edgePlusCount +=1
            else:
                idxTop.append([currentPoint[1], currentPoint[0], CurrentElev])
                edgePlusCount += 1
            if edgePlusCount >= 45:
                treeBottom = spatial.cKDTree(idxBottom)
                treeTop = spatial.cKDTree(idxTop)
                edgePlusCount = 0
        switchMe = False
    return idxBottom, idxTop



def floodFill(img, idxBottom, idxTop, bbox, sp):
    fill = set()
    for p in sp:
        fill.add((p[0],p[1], img[p[0]][p[1]]))
    CycleCount = 0
    treeBottom = spatial.cKDTree(idxBottom)
    treeTop = spatial.cKDTree(idxTop)
    edgePlusCount = 0
    between = False
    while fill:
        y,x,z = fill.pop()
        currentPoint = [y, x, z]
        closestPoint = nearestPoints(currentPoint, treeBottom, treeTop, idxBottom, idxTop)
        bottom = [closestPoint[0][0], closestPoint[0][1], closestPoint[0][2]]
        top = [closestPoint[1][0], closestPoint[1][1], closestPoint[1][2]]
        CurrentElev = round(WeightedEvelvation(bottom, top, currentPoint), 4)
        emptyArray[y][x] = CurrentElev
        filled.add((y, x))
        CycleCount += 1
        #print bbox
        #print currentPoint
        #time.sleep(23489)
        print CycleCount
        edge = 0
        if bbox[2] <= x <= bbox[3] and bbox[0] <= y <= bbox[1]:
            west = [y, x - 1, img[y][x - 1]]
            WestCord = (y, x - 1)
            east = [y, x + 1, img[y][x + 1]]
            EastCord = (y, x + 1)
            north = [y + 1, x, img[y + 1][x]]
            NorthCord = (y + 1, x)
            south = [y - 1, x, img[y - 1][x]]
            SouthCord = (y - 1, x)
            if north[2] <= CurrentElev and not NorthCord in filled:
               fill.add((north[0], north[1], north[2]))
            if west[2] <= CurrentElev and not WestCord in filled:
                fill.add((west[0], west[1], west[2]))
            if east[2] <= CurrentElev and not EastCord in filled:
                fill.add((east[0], east[1], east[2]))
            if south[2] <= CurrentElev and not SouthCord in filled:
                fill.add((south[0], south[1], south[2]))


def savingGrid(img, geoTrans, output):
     print "Saving Grid"
     proj = osr.SpatialReference()
     proj.ImportFromEPSG(102724)
     proj.SetLinearUnits("FEET", 0.3048)
     Ny, Nx = img.shape
     driver = gdal.GetDriverByName("GTiff")
     ds = driver.Create(output, Nx, Ny, 1, GDT_Float32)
     ds.SetProjection(proj.ExportToWkt())
     ds.GetRasterBand(1).WriteArray(img)
     ds.SetGeoTransform(geoTrans)
     ds = None
     print "Done"


if __name__ == '__main__':
    source = r'M:\2012\12TUL02_East_Tulsa_MDP\GIS\Data\Spunky\Spunky_BE_Linear.img'
    target = r"M:\2012\12TUL02_East_Tulsa_MDP\GIS\Data\Spunky\2014\Ryan_B\spunky500.tif"
    pointFile = r'M:\2012\12TUL02_East_Tulsa_MDP\GIS\Data\Spunky\2014\Ryan_B\spunkyNodes.shp'
    pointFileIndex = r'M:\2012\12TUL02_East_Tulsa_MDP\GIS\Data\Spunky\2014\Ryan_B\SpunkyIntersect500.shp'
    filled = set()
    print "Opening image..."
    img = gdal.Open(source, GA_ReadOnly)
    cols = img.RasterXSize
    rows = img.RasterYSize
    distanceArray = np.ndarray(shape=(rows,cols), dtype=float)
    emptyArray = np.ndarray(shape=(rows, cols), dtype=float)
    srs_wkt = osr.SpatialReference()
    srs_wkt.ImportFromEPSG(102724)
    srs = srs_wkt.ExportToWkt()
    band = img.GetRasterBand(1)
    data = band.ReadAsArray(0, 0, cols, rows)
    points = []
    shp = shapefile.Reader(pointFile)
    for feature in shp.shapeRecords():
        point = feature.shape.points[0]
        rec = float(feature.record[13])
        points.append([point[0], point[1], rec])
    shp = shapefile.Reader(pointFileIndex)
    index = []
    for feature in shp.shapeRecords():
        point = feature.shape.points[0]
        rec = float(feature.record[1])
        index.append([point[0], point[1], rec])

    index = sorted(index, key=itemgetter(2))
    indexPoints = []
    for i in index:
        indexPoints.append([])
    geoTrans = img.GetGeoTransform()
    originX = geoTrans[0]
    originY = geoTrans[3]
    for l in points:
        l[0] = int((l[0] - originX)/geoTrans[1]) -1
        l[1] = int((l[1] - originY)/geoTrans[5]) -1
    validPoints = []
    for l in points:
        if l[2] >= data[l[1], l[0]]:
            validPoints.append([l[0], l[1], l[2]])
    validPoints = sorted(validPoints, key=itemgetter(2))
    bbox = []
    minY = min(validPoints, key=itemgetter(1))
    maxY = max(validPoints, key=itemgetter(1))
    minX = min(validPoints, key=itemgetter(0))
    maxX = max(validPoints, key=itemgetter(0))
    bbox.append(minY[1])
    bbox.append(maxY[1])
    bbox.append(minX[0])
    bbox.append(maxX[0])
    index = uniquelist(index)
    for i in range(0, len(index)):
        for v in validPoints:
            if index[i][2] == round(v[2], 4):
                indexPoints[i].append(v)
    indexPoints = [x for x in indexPoints if x != []]
    even, odd = indexPoints[0::2], indexPoints[1::2]
    EvenTree = []
    OddTree = []
    for e in even:
        for i in e:
            EvenTree.append(i)
    for o in odd:
        for i in o:
            OddTree.append(i)
    startingpoints = []
    for l in index:
        l[0] = int((l[0] - originX)/geoTrans[1]) -1
        l[1] = int((l[1] - originY)/geoTrans[5]) -1
    for v in index:
        startingpoints.append([v[1], v[0]])
    print "Initializing Matrix"
   

    print "Began flood fill at {0}".format(datetime.datetime.now())
    print "Building Index Layer"

    EvenTree, OddTree = floodFillIndex(data, EvenTree, OddTree, bbox, startingpoints)
    filled.clear()
    print "Building Surface"
    floodFill(data, EvenTree, OddTree, bbox, startingpoints)
    print "Finished flood fill at {0}".format(datetime.datetime.now())
    savingGrid(emptyArray, geoTrans, target)