from flask import Flask, request, jsonify
import numpy as np
import cv2
import requests
from skimage.io import imread
from sklearn.cluster import KMeans
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = img / 255
    return img


def kMeans_cluster(img):
    image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2D)
    clustOut = kmeans.cluster_centers_[kmeans.labels_]
    clustered_3D = clustOut.reshape(img.shape[0], img.shape[1], img.shape[2])
    clusteredImg = np.uint8(clustered_3D*255)
    return clusteredImg


def edgeDetection(clusteredImage):
    edged1 = cv2.Canny(clusteredImage, 0, 255)
    edged = cv2.dilate(edged1, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged


def getBoundingBox(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    return boundRect, contours, contours_poly, img


def calcFeetSize(img, fboundRect):
    y2 = int(img.shape[0] / 10)
    x2 = int(img.shape[1] / 10)

    fh = y2 + fboundRect[3]
    fw = x2 + fboundRect[2]

    ph = img.shape[0]
    pw = img.shape[1]

    opw = 210
    oph = 297

    ofs = 0.0
    if fw > fh:
        ofs = (oph / pw) * fw
    else:
        ofs = (oph / ph) * fh

    return ofs


@app.route('/measure-feet', methods=['POST'])
def measure_feet():
    try:
        file = request.files['image'].read()
        npimg = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        preprocessed_img = preprocess(img)
        clustered_img = kMeans_cluster(preprocessed_img)
        edged_img = edgeDetection(clustered_img)
        boundRect, contours, contours_poly, _ = getBoundingBox(edged_img)

        if len(boundRect) > 0:
            paperbb = boundRect[0]
            feet_size = {
                "width": paperbb[2],
                "height": paperbb[3]
            }
            summary_feet_size = calcFeetSize(img, paperbb) / 10
            return jsonify({"success": True, "feet_size": feet_size, "summary": summary_feet_size})
        else:
            return jsonify({"error": "No contours found"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health-check', methods=['GET'])
def health_check():
    return jsonify("hello")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8081)
