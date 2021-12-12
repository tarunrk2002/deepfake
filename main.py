import cv2
import dlib
import numpy as np

# print(cv2.__version__)
# print(dlib.__version__)

front_face = dlib.get_frontal_face_detector()
face_p = dlib.shape_predictor("dataset/shape_predictor_68_face_landmarks.dat")

tarun_image = cv2.imread("images/tarun.jpg")
tarun_image_copy = tarun_image
tarun_grey = cv2.cvtColor(tarun_image, cv2.COLOR_BGR2GRAY)

# print(tarun_image)
# print(tarun_image_copy)
# print(tarun_grey)

kushvith_image = cv2.imread("images/1.jpg")
kushvith_image_copy = kushvith_image
kushvith_grey = cv2.cvtColor(kushvith_image, cv2.COLOR_BGR2GRAY)

# cv2.imshow("source",kushvith_grey)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("kushvith", kushvith_grey)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# def asd():
#     a = 5+5
#     b = 9+9
#     return a,b
#
# addition,sub = asd()
# print(addition,sub)

tarun_image_can = np.zeros_like(tarun_grey)
h, w, c = kushvith_image.shape
kushvith_image_can = np.zeros((h, w, c), np.uint8)

# cv2.imshow("can",tarun_image_can)
# cv2.waitKey(0)

tarun_face = front_face(tarun_grey)
for a in tarun_face:
    pred = face_p(tarun_grey,a)
    tarun_faces = []
    for j in range(0,68):
        pointx=pred.part(j).x
        pointy=pred.part(j).y
        tarun_faces.append((pointx,pointy))

#         cv2.circle(tarun_grey,(pointx,pointy),2,(255,0,0),-1)
#         cv2.imshow("points",tarun_grey)
# cv2.waitKey(0)


kushvith_face = front_face(kushvith_image)
for a in kushvith_face:
    pred = face_p(kushvith_image,a)
    kushvith_faces = []
    for j in range(0,68):
        pointx=pred.part(j).x
        pointy=pred.part(j).y
        kushvith_faces.append((pointx,pointy))

    #         cv2.circle(kushvith_image,(pointx,pointy),2,(255,0,0),-1)
    #         cv2.imshow("points",kushvith_image)
# cv2.waitKey(0)

    np_array = np.array(tarun_faces,np.int32)
    ch = cv2.convexHull(np_array)
    # cv2.polylines(tarun_image,[ch],True,(255,0,0),2)
    # cv2.imshow("tarun hull",tarun_image)
    # cv2.waitKey(0)

    np_arrays = np.array(kushvith_faces,np.int32)
    chs = cv2.convexHull(np_arrays)
    # cv2.polylines(kushvith_image,[chs],True,(255,0,0),2)
    # cv2.imshow("kushvith hull",kushvith_image)
    # cv2.waitKey(0)
    cv2.fillConvexPoly(tarun_image_can,ch,(255,0,0))
#     cv2.imshow('convexhull',tarun_image_can)
# cv2.waitKey(0)

    in_face = cv2.bitwise_and(tarun_image,tarun_image,mask=tarun_image_can)
    cv2.imshow("inface",in_face)
cv2.waitKey(0)
















