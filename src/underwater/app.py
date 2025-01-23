import gradio as gr
import cv2
import argparse
import sys
import numpy as np
import torch

import options as option
from models import create_model
sys.path.insert(0, "../")
import utils as util

# options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default='test.yml', help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

# load pretrained model by default
model = create_model(opt)
device = model.device

mondri = util.IRMondri(max_sigma=opt["mondri"]["max_sigma"], T=opt["mondri"]["T"], schedule=opt["mondri"]["schedule"], eps=opt["mondri"]["eps"], device=device)
mondri.set_model(model.model)

def mondri_water(image):
    image = image[:, :, [2, 1, 0]] / 255.
    LQ_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    noisy_tensor = mondri.noise_state(LQ_tensor)
    model.feed_data(noisy_tensor, LQ_tensor)
    model.test(mondri)
    visuals = model.get_current_visuals(need_GT=False)
    output = util.tensor2img(visuals["Output"].squeeze())
    return output

interface = gr.Interface(fn=mondri_water, inputs="image", outputs="image", title="MondriAI underwater test app")
interface.launch()
