from os.path import join, isfile
from os import listdir

from xmljson import badgerfish as bf
from xml.etree.ElementTree import fromstring
from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2
import numpy as np                                  # (pip install numpy)
from skimage import measure                         # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)
from uuid import uuid4


def create_mask(fpath, show_plot: bool = False):
    img = cv2.imread(fpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (30, 30)
    )
    res = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    if show_plot:
        plt.imshow(res, 'gray')
    return res


def create_sub_mask_annotation(sub_mask,
                               image_id,
                               category_id,
                               annotation_id,
                               is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation


def to_dict(fpath):
    with open(fpath, 'r') as F:
        X = bf.data(fromstring(F.read()))
    DATA = dict(X['Image'])
    data = {}
    for key in DATA:
        if DATA[key].get("$"):
            data[key] = DATA[key]['$']
    return data


def create_annotations(data_path: str="data"):
    """

    :param data_path:
    :return:
    """

    XMLS = [
        join(data_path, f) for f in listdir(data_path)
        if f[-3:] == "xml"
    ]
    IMGS = [
        join(data_path, f) for f in listdir(data_path)
        if f[-3:] != "xml"
    ]
    ImageData = []

    for xml in tqdm(XMLS):
        ImageData.append(to_dict(xml))

    Content = {}
    for img in ImageData:
        content = img['Content']
        if content not in Content:
            Content[content] = [img]
        else:
            Content[content].append(img)

    LeafScan = Content['LeafScan']

    Annotations = []
    Exceptions = []

    for leafscan in tqdm(LeafScan):
        fpath = join(data_path, f"{leafscan['MediaId']}.jpg")
        mask = create_mask(fpath)
        try:
            annotation = create_sub_mask_annotation(
                mask,
                fpath,
                1,
                f"{uuid4()}",
                False
            )
            Annotations.append(annotation)
        except Exception as e:
            Exceptions.append(fpath)
            pass

    return Annotations
