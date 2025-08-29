import numpy as np

def to_img(pic): return pic.transpose(1,2,0)
def to_tensor(pic): return pic.transpose(2,0,1)
