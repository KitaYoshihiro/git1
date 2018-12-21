from cntk.ops.functions import load_model
from PIL import Image 
import numpy as np

z = load_model("../my_cntk_model.dnn")

print(z)
