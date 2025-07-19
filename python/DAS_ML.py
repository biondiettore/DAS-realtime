import sys
import os
path_dasml = os.path.split(os.path.abspath(__file__))[0]
EqNetPath = os.path.join(path_dasml, "../external/EQNet/")
sys.path.insert(0,EqNetPath)
import eqnet
from eqnet.utils import detect_peaks, extract_picks
from dataclasses import dataclass
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Union
import torch
import pandas as pd
import numpy as np

@dataclass
class Config():
    model = "phasenet_das"
    backbone = "unet"
    phases = ["P", "S"]
    device = "cuda"  
    min_prob = 0.5  
    amp = True
    dtype = torch.float32
    area = None

class Data(BaseModel):
    id: List[str]
    timestamp: List[str]
    vec: Union[List[List[List[float]]], List[List[float]]]
    dt_s: Optional[float] = 0.01

def predict(meta: Data):

    with torch.inference_mode():

        with torch.amp.autocast(device_type=args.device, enabled=args.amp):
            
            scores = torch.softmax(model(meta), dim=1)  # [batch, nch, nt, nsta]
            topk_scores, topk_inds = detect_peaks(scores, vmin=args.min_prob, kernel=21)

            picks = extract_picks(
                topk_inds,
                topk_scores,
                file_name=meta["id"],
                begin_time=meta["timestamp"] if "timestamp" in meta else None,
                dt=meta["dt_s"] if "dt_s" in meta else 0.01,
                vmin=args.min_prob,
                phases=args.phases,
            )

    return {"picks": picks}

def load_model(args):

    model = eqnet.models.__dict__[args.model](
        backbone=args.backbone,
        in_channels=1,
        out_channels=(len(args.phases) + 1),
    )

    if args.model == "phasenet" and (not args.add_polarity):
        raise ("No pretrained model for phasenet, please use phasenet_polarity instead")
    elif (args.model == "phasenet") and (args.add_polarity):
        model_url = "https://github.com/AI4EPS/models/releases/download/PhaseNet-Polarity-v3/model_99.pth"
    elif args.model == "phasenet_das":
        if args.area is None:
            model_url = "https://github.com/AI4EPS/models/releases/download/PhaseNet-DAS-v5/model_29.pth"
        elif args.area == "forge":
            model_url = (
                "https://github.com/AI4EPS/models/releases/download/PhaseNet-DAS-ConvertedPhase/model_99.pth"
            )
        else:
            raise ("Missing pretrained model for this area")
    else:
        raise
    state_dict = torch.hub.load_state_dict_from_url(
        model_url, model_dir="./", progress=True, check_hash=True, map_location="cpu"
    )
    model.load_state_dict(state_dict["model"], strict=True)

    return model

###################### FastAPI ######################
app = FastAPI()
args = Config()
model = None#load_model(args)
# model.to(args.device)
# model.eval()


def preload_model(**kwargs):
    global model, args
    if "device" in kwargs:
        args.device = kwargs["device"]
    else:
        args.device = "cuda"
    model = load_model(args)
    model.to(args.device)
    model.eval()
    return

async def preload_model_async(**kwargs):
    global model, args
    if "device" in kwargs:
        args.device = kwargs["device"]
    else:
        args.device = "cuda"
    model = load_model(args)
    model.to(args.device)
    model.eval()
    return


@app.post("/predict")
def predict(meta: Data):

    with torch.inference_mode():

        with torch.amp.autocast(device_type=args.device, enabled=args.amp):
            
            scores = torch.softmax(model(meta), dim=1)  # [batch, nch, nt, nsta]
            topk_scores, topk_inds = detect_peaks(scores, vmin=args.min_prob, kernel=21)

            picks = extract_picks(
                topk_inds,
                topk_scores,
                file_name=meta["id"],
                begin_time=meta["timestamp"] if "timestamp" in meta else None,
                dt=meta["dt_s"] if "dt_s" in meta else 0.01,
                vmin=args.min_prob,
                phases=args.phases,
            )

    return {"picks": picks}

def phasenet_das(das_data, timestamp, ev_id, dt):
    """Function to apply PhaseNet-DAS to given data window"""
    global args
    vec = das_data[:,:].T
    vec = vec[np.newaxis, :, :]
    data = torch.tensor(vec, dtype=args.dtype).unsqueeze(0) # [batch, nch, nt, nsta]
    meta = {"id": [ev_id], "timestamp":[timestamp], "data": data, "dt_s": dt}
    picks = predict(meta)["picks"]
    picks = picks[0] ## batch size = 1
    picks = pd.DataFrame.from_dict(picks, orient="columns")
    if "cuda" in args.device:
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()
    return picks


def extract_peak_amp(data,TT,fs,ot,f1,f2, chIDs=None, align_wind=1.0):
    """Function to compute SNR for a given earthquake"""
    maxAmp = np.zeros_like(TT)
    nch = data.shape[0] if chIDs is None else chIDs.shape[0]
    nt = data.shape[1]
    dt = 1.0/fs
    align_wind= align_wind + 1.0/(f1+f2)
    if chIDs is None:
        chIDs = range(nch)
    for ich in range(nch):
        ch = chIDs[ich]
        min_t_align = TT[ich]-align_wind*0.5
        max_t_align = TT[ich]+align_wind*0.5
        it_min_wind = max(0,int((min_t_align-ot)/dt+0.5))
        it_max_wind = min(nt-1,int((max_t_align-ot)/dt+0.5))
        maxAmp[ich] = abs(data[ch,it_min_wind:it_max_wind]).max()
    return maxAmp