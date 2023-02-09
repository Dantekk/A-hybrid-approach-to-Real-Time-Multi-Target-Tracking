from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.multitracker import JDETracker
from tracker.multitracker_byte import Byte_JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets
###
from tracking_utils.object_info import object_info as object_info
###

from tracking_utils.utils import mkdir_if_missing
from opts import opts

##
from lib.optical_flow.flow import Flow

##
import random
import decimal

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, save_text=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)

    ### Qui decido se utilizzare Byte oppure no
    print(opt.byte_mode)
    #byte_mode = False
    if opt.byte_mode==1:
        tracker = Byte_JDETracker(opt, frame_rate=frame_rate)
        print("*** BYTE MODE ATTIVA")
    else:
        tracker = JDETracker(opt, frame_rate=frame_rate)
        print("*** BYTE MODE NON ATTIVA")
    ###

    timer = Timer()
    results = []
    frame_id = 0
    

    # Define lkt counter
    lkt_counter = 0
    init_lkt_image = None
    flow = Flow()
    tlwhs_to_lkt = []

    for i, (path, img, img0) in enumerate(dataloader):
        
        #if i % 8 != 0:
            #continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()

        ## Detection mode
        # For init frame, we use detection model to get the initial bounding box
        if lkt_counter == 0:
            # Clone init frame
            init_lkt_image = img0.copy()
            # init lkt
            flow.init(init_lkt_image)
            
            # Forward image to model
            if use_cuda:
                blob = torch.from_numpy(img).cuda().unsqueeze(0)#.half()
            else:
                blob = torch.from_numpy(img).unsqueeze(0)
            online_targets = tracker.update(blob, img0)
            
            online_tlwhs = []
            online_ids = []
            #online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    #online_scores.append(t.score)

            if int(opt.lkt_num_frame)>0:
                ###
                # frame id : contains current frame id
                # online_tlwhs : contains current frame bounding boxes
                # online_ids : contains current frame bounding boxes ids
                ###
                tlwhs_to_lkt = [] # list of bounding boxes to be sent to lkt
                for i, tlwh in enumerate(online_tlwhs):
                    x1, y1, w, h = tlwh # x1, y1 is top left corner, w, h is width and height of bbox
                    x2, y2 = x1 + w, y1 + h # x2, y2 is bottom right corner of bbox
                    #print(int(x1), int(y1), int(x2), int(y2))
                    tlwhs_to_lkt.append([int(x1), int(y1), int(x2), int(y2), ])

                # There are bounding boxes to be sent to lkt
                if len(tlwhs_to_lkt)!=0:
                    res = flow.computeGoodFeatures(tlwhs_to_lkt, online_ids)
                    ## If res==False -> no good features found -> repeat detection mode
                    if res:
                        lkt_counter += 1
                    else:
                        print("***** NO GOOD FEATURES FOUND *****") 

            
        ## Optical Flow mode
        elif lkt_counter > 0 and int(opt.lkt_num_frame)>0:
            online_tlwhs = False
            #print(lkt_counter)
            online_tlwhs = flow.computeNextBBox(img0, tlwhs_to_lkt)

            ## Update Kalman Filter
            tracker.update_with_optical_flow()

            # If is not possible compute optical flow, repeat detection mode
            if online_tlwhs==False:
                flow = Flow()
                lkt_counter = 0
                continue
            
            lkt_counter += 1
            #print(lkt_counter)
            if lkt_counter>int(opt.lkt_num_frame):
                lkt_counter = 0
                flow = Flow()

              
        ###
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))


        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=(1. / timer.average_time))
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        
        frame_id += 1
      

    # save results
    if save_text:
        write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls



"""
View src/lib/datasets/dataset/jde.py file for mod res of output image
row 100 -> self.w, self.h = img_size[0], img_size[1] #1920, 1080
"""

def main(opt, data_root='/data/MOT15/train', det_root=None, seqs=('MOT15-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True,
         result_root = "/home/ciro.panariello/tracking/FairMOT/FairMOT/results"):
    logger.setLevel(logging.INFO)
    print("****** ",data_root)

    data_root = "/home/ciro.panariello/tracking/dataset/data/data/MOT15/train/"
    print("****** ",data_root)
    result_root = result_root+exp_name#os.path.join(data_root, '..', 'results', exp_name)
    print("****** ",result_root)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = result_root+"/outputs" if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        
        result_filename =os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))





if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    ##
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='convnext_tiny_skip2.pth',
         show_image=False,
         save_images=False,
         save_videos=False,
         result_root = "output_folder)
