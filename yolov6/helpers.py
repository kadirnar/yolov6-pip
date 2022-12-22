import os
import cv2
import math
import torch
import numpy as np
import os.path as osp
import time
from tqdm import tqdm
from pathlib import Path
from collections import deque
import sys

from yolov6.utils.downloads import attempt_download_from_hub
from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.datasets import LoadData
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer


class CalcFPS:
    def __init__(self, nsamples: int = 50):
        self.framerate = deque(maxlen=nsamples)

    def update(self, duration: float):
        self.framerate.append(duration)

    def accumulate(self):
        if len(self.framerate) > 1:
            return np.average(self.framerate)
        else:
            return 0.0
        
def check_img_size(img_size, s=32, floor=0):
    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size if isinstance(img_size,list) else [new_size]*2

def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor  


def model_switch(model):
    ''' Model switch to deploy status '''
    from yolov6.layers.common import RepVGGBlock
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()

    LOGGER.info("Switch model to deploy modality.")  
    
    
class YOLOV6:
    def __init__(
        self, 
        weights = 'weights/yolov6s.pt',
        device = 'cpu',
        hf_model = False,
    ):

        self.__dict__.update(locals())
        self.device = device
        self.half = False

        # Load model
        if hf_model:
            self.weights = attempt_download_from_hub(weights, hf_token=None)
        else:
            self.weights = weights
            
        model = self.load_model()
        self.stride = model.stride
        
        # Model Parameters
        self.conf = 0.25
        self.iou = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000
        self.save_dir = 'inference/output'
        self.save_txt = False
        self.save = True
        self.hide_labels = False
        self.hide_conf = False
        self.show = False


    
    def load_model(self):
        # Init model
        model = DetectBackend(self.weights, device=self.device)
        
        # Switch model to deploy status
        model_switch(model.model)

        # Half precision
        if self.half & (self.device != 'cpu'):
            model.model.half()
        else:
            model.model.float()
            self.half = False

        self.model = model
        return model


    def predict(
        self, 
        source,
        yaml,
        img_size,
    ):
        ''' Model Inference and results visualization '''
        files = LoadData(source)
        class_names = load_yaml(yaml)['names']
        img_size = check_img_size(img_size, s=self.stride)
        if self.device != 'cpu':
            self.model(torch.zeros(1, 3, *img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

        vid_path, vid_writer, windows = None, None, []
        fps_calculator = CalcFPS()
        for img_src, img_path, vid_cap in tqdm(files):
            img, img_src = Inferer.precess_image(img_src, img_size, self.stride, self.half)
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img[None]
                # expand for batch dim
            
            t1 = time.time()
            pred_results = self.model(img)
            det = non_max_suppression(pred_results, self.conf, self.iou, classes=self.classes, agnostic=self.agnostic_nms, max_det=self.max_det)[0]
            t2 = time.time()
            
            # Create output files in nested dirs that mirrors the structure of the images' dirs
            rel_path = osp.relpath(osp.dirname(img_path), osp.dirname(source))
            save_path = osp.join(self.save_dir, rel_path, osp.basename(img_path))  # im.jpg
            txt_path = osp.join(self.save_dir, rel_path, osp.splitext(osp.basename(img_path))[0])
            os.makedirs(osp.join(self.save_dir, rel_path), exist_ok=True)

            gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            img_ori = img_src.copy()

            # check image and font
            assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
            Inferer.font_check()

            det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
            for *xyxy, conf, cls in reversed(det):
                if self.save_txt:  # Write to file
                    xywh = (Inferer.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf)
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if self.save or self.show:  # Add bbox to image
                    class_num = int(cls)  # integer class
                    label = None if self.hide_labels else (class_names[class_num] if self.hide_conf else f'{class_names[class_num]} {conf:.2f}')

                    Inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))

 
            img_src = np.asarray(img_ori)

            # FPS counter
            fps_calculator.update(1.0 / (t2 - t1))
            avg_fps = fps_calculator.accumulate()
            if files.type == 'video':
                Inferer.draw_text(
                    img_src,
                    f'FPS: {avg_fps:.2f}',
                    pos=(20, 20),
                    font_scale=1.0,
                    text_color=(204, 85, 17),
                    text_color_bg=(255, 255, 255),
                    font_thickness=2,
                )

            if self.show:
                if img_path not in windows:
                    windows.append(img_path)
                    cv2.namedWindow(str(img_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(img_path), img_src.shape[1], img_src.shape[0])
                cv2.imshow(str(img_path), img_src)
                cv2.waitKey(0)  # 1 millisecond
                
            
            # Save results (image with detections)
            if self.save:
                if files.type == 'image':
                    cv2.imwrite(save_path, img_src)
                else:  # 'video' or 'stream' 
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, img_ori.shape[1], img_ori.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(img_src)

    
if __name__ == '__main__':
    model = YOLOV6(
        weights='kadirnar/yolov6t-v2.0',
        device='cuda:0',
        hf_model=True,
    )
    model = model.predict(
        source='data/images/',
        yaml='data/coco.yaml',
        img_size=640,
    )
