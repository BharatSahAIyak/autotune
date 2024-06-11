import os 
import subprocess 
from typing import Optional, Union 

import numpy as np 
import torch 
import torch.nn.functional as F 


SAMPLE_RATE=8000


def load_audio(file:str,sr: int=SAMPLE_RATE):

    try:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out=subprocess.run(cmd,capture_output=True,check=True).stdout
    
    except subprocess.CalledProcessError as e:

        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e 
    
    return np.frombuffer(out,np.int16).flatten().astype(np.float32)/ 32768.0
