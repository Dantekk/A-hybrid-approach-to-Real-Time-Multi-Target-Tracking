from traceback import print_tb
import cv2
import numpy as np
import time

class Flow:
    def __init__(self,):
        self.old_gray_img = None
        self.p0 = None
        self.unique_idx = []
        self.shi_tomasi = False

        # Parameters for lucas kanade optical flow
        # self.lk_params = dict(winSize  = (15, 15),
        #                       maxLevel = 3,
        #                       criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.lk_params = dict(winSize  = (5, 5),
                              maxLevel = 5,
                              criteria = [3, 10, 0.03])

    def init(self, image):
        #print(image.shape)
        self.old_gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #print(self.old_gray_img.shape)


    # Return True if features are scoveded in the image, False otherwise
    def computeGoodFeatures(self, tracks, online_ids):
        
        #start_time = time.time()
        keypoints = []
        # Compute mask and put it into gray scale image
        edge_mask = np.zeros((self.old_gray_img.shape[0], self.old_gray_img.shape[1]), dtype=np.uint8)
        for i, track in enumerate(tracks):
            
            img_to_input = cv2.rectangle(edge_mask, (track[0], track[1]), (track[2], track[3]), 255, cv2.FILLED)
            if self.shi_tomasi:            
                ## Shi-Tomasi corner points
                p_add = cv2.goodFeaturesToTrack(self.old_gray_img, mask=img_to_input, maxCorners=1500, qualityLevel=0.06, minDistance=3, blockSize=3)
                p_add = np.c_[p_add, np.full((p_add.shape[0],1,1), online_ids[i])]
                ##
            else:
                fast = cv2.FastFeatureDetector_create(threshold=20)
                #find keypoints
                kp = fast.detect(self.old_gray_img, mask=img_to_input)
                
                # Get all Keypoints in a numpy array
                p_add = []
                tmp = []
                for k in kp:
                    p_add.append([[k.pt]])
                    tmp.append(k.pt)
                
                if len(tmp)==0:
                    return False
                p_add = np.concatenate(p_add)
                
            ### Add a column that contains the track id  
            p_add = np.c_[p_add, np.full((p_add.shape[0],1,1), online_ids[i])]
            self.unique_idx.append(online_ids[i])
            keypoints.append(p_add)

        #print("--- computeGoodFeatures time %s seconds ---" % (time.time() - start_time))
        ## Convert the list of np array in a single np array and convert in float32 (because the function np.c_ returns float64)
        if len(keypoints)==0:
            return False
        else:
            p0 = np.concatenate(keypoints)
            p0 = np.float32(p0)
            self.p0 = p0
            return True

    def get_inliers(self, prev_pts, cur_pts, inlier_mask):
        keep = np.where(inlier_mask.ravel())
        return prev_pts[keep], cur_pts[keep]

    def computeNextBBox(self, currentFrame, coordinates):
        # List that will contain the new coordinates of the bounding box
        coordinate_bb_final = []

        frame_gray = cv2.cvtColor(currentFrame, cv2.COLOR_RGB2GRAY)
        frame_gray = np.array(frame_gray, dtype=np.uint8)
        self.old_gray_img = np.array(self.old_gray_img, dtype=np.uint8)

        ####start_time = time.time()
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray_img, frame_gray, self.p0[:,:,:2], None, **self.lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = self.p0[st==1]
        
        ## Partition the good_old array in different tracks by unique id
        ## https://stackoverflow.com/questions/31863083/python-split-numpy-array-based-on-values-in-the-array
        partizioned_old_points = np.split(good_old, np.where(np.diff(good_old[:,-1]))[0]+1)
        start = 0

        tmp_good_old = []
        tmp_good_new = []
        ####print("--- calcOpticalFlowPyrLK time %s seconds ---" % (time.time() - start_time))

        ####start_time = time.time()
        for i in range (len(partizioned_old_points)):
            #print(partizioned_old_points[i][:,:2])
            len_partizion = len(partizioned_old_points[i])
            end = start + len_partizion
            
            ## Compute the transformation matrix of new points
            affine_mat, inlier_mask = cv2.estimateAffinePartial2D(np.array(partizioned_old_points[i][:,:2]), np.array(good_new[start:end]),
                                                                  method=cv2.RANSAC,
                                                                  maxIters=500, # default is 2000
                                                                  confidence=0.99)

            if affine_mat is None:
                print(partizioned_old_points[i][:,:2].shape)
                print(good_new[start:end].shape)
                print("***** IT'S NOT POSSIBILE ESTIMATE MATRIX TRANSFORMATION *****")
                return False

            # Remove the points that are not inliers
            good_old_tmp, good_new_tmp = self.get_inliers(partizioned_old_points[i], good_new[start:end], inlier_mask)

            # Add filtered points in a list of numpy array
            tmp_good_old.append(good_old_tmp)
            tmp_good_new.append(good_new_tmp)
            start = end

            # get omogenee cordinate of bouding box
            coordinate_bb = []
            coordinate_bb.append([coordinates[i][:2][0], coordinates[i][:2][1], 1])
            coordinate_bb.append([coordinates[i][2:4][0], coordinates[i][2:4][1], 1])
            
            #coordinate_bb = np.concatenate(coordinate_bb)
            coordinate_bb = np.array(coordinate_bb, dtype=np.uint16)

            # Multiply the affine matrix with the coordinates of the bbox for estimate its new positions 
            # nuove_coordinate = Matrice_affine * [x,y,1]'
            # https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html

            coordinate_bb = affine_mat.dot(coordinate_bb.transpose())
            coordinate_bb = coordinate_bb.transpose()

            ## compute w,h of the bounding box because the function vis.lot_tracking(...) requires 
            ## coordinate of bbox as x1,y1,h,w as a list of np.array()
            
            i=0
            x1 = coordinate_bb[i][0]
            y1 = coordinate_bb[i][1]
            x2 = coordinate_bb[i+1][0]
            y2 = coordinate_bb[i+1][1]
            coordinate_bb_final.append(np.array([x1, y1, x2-x1, y2-y1]))
        ####print("--- estimateAffinePartial2D time %s seconds ---" % (time.time() - start_time))
        ####print()  

        # Transform list of np.array in a single np.array
        good_old = np.concatenate(tmp_good_old)
        good_new = np.concatenate(tmp_good_new)

        # Now update the previous frame and previous points
        self.old_gray_img = frame_gray.copy()

        ## Add another column that contains the track id
        self.p0 = good_new.reshape(-1, 1, 2)
        self.p0 = np.c_[self.p0, good_old[:,2:3, None]]
        self.p0 = np.float32(self.p0)

        return coordinate_bb_final

        
        
        

