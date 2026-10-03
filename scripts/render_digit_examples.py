"""Optional Pillow contact sheet from local MNIST training images."""
from pathlib import Path
import argparse
import sys
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prototype_lab.offline import read_idx


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);a=p.parse_args()
    images=read_idx(a.raw/'train-images-idx3-ubyte');labels=read_idx(a.raw/'train-labels-idx1-ubyte')
    sheet=Image.new('RGB',(1000,315),'#14243a');draw=ImageDraw.Draw(sheet)
    draw.text((25,20),'MNIST TRAINING EXAMPLES | one observed image per digit',fill='white')
    for digit in range(10):
        image=images[np.flatnonzero(labels==digit)[0]]
        tile=Image.fromarray(image).resize((84,84),Image.Resampling.NEAREST).convert('RGB')
        sheet.paste(tile,(15+98*digit,80));draw.text((48+98*digit,180),str(digit),fill='white')
    draw.text((25,245),'Dataset preview only. These images are not selected cluster representatives.',fill='#a6cadb')
    destination=ROOT/'examples/digits.png';sheet.save(destination)
    print(destination)
