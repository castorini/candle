from PIL import Image
import numpy as np

def draw_binary_filter2d(filter, enlarge=10):
    arr = 255 * filter.clamp(0, 1).data.cpu().numpy()
    new_size = (arr.shape[0] * enlarge, arr.shape[1] * enlarge)
    Image.fromarray(arr.astype(np.uint8)).resize(new_size).show()