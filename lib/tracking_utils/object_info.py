import cv2
import numpy as np

def object_info(tlwhs, obj_ids, frame_id=0):

    frame_info = {"frame_id" : frame_id, "person_info" : []}
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh # mi da le coordinate del punto e la larghezza,altezza della bb
        #intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        print("Frame #%d"%(frame_id))
        print("object with ID : %d"%(obj_id))
        print("Coordinate [x,y] : [%d,%d]" % (int(x1), int(y1)))
        print("Coordinate [x+w,y+h] : [%d,%d]" % (int(x1+w), int(y1+h)))

        info = {"ID" : obj_id,
                "x"  : int(x1),
                "y"  : int(y1),
                "z"  : 0,
                "w"  : int(w),
                "h"  : int(h),
                "q0" : 0,
                "q1" : 0,
                "q2" : 0,
                "q3" : 0}

        frame_info["person_info"].append(info)

    return frame_info
