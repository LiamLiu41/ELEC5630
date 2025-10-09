import numpy as np
from PIL import Image
import os

def load_datadir_re(datadir, bitDepth, resize, gamma, load_imgs=True, load_mask=True, white_balance=[1,1,1]):
    """
    Load photometric stereo data from the given directory.
    """
    # If white_balance is a vector, convert it to a diagonal matrix
    if isinstance(white_balance, (list, tuple, np.ndarray)) and len(white_balance) == 3:
        white_balance = np.diag(white_balance)
    
    # Build data structure
    data = type('data', (object,), {})()  # Create an empty class instance
    data.L = np.loadtxt(os.path.join(datadir, 'light_directions.txt'))
    
    # Ensure data.L is a 2D array so we can perform matrix multiplication correctly
    if data.L.ndim == 1:
        data.L = data.L.reshape(-1, 3)
    
    # Perform matrix multiplication to apply white balance
    data.L = np.dot(data.L, white_balance).T


    # Read filename list
    with open(os.path.join(datadir, 'filenames.txt'), 'r') as f:
        data.filenames = [os.path.join(datadir, line.strip()) for line in f.readlines()]
    
    # Load mask image if needed
    if load_mask and not hasattr(data, 'mask'):
        data.mask = np.array(Image.open(os.path.join(datadir, 'mask.png')))
        data.mask = np.array(Image.fromarray(data.mask).resize(resize, Image.NEAREST))
        data.mask = data.mask[:,:,None]
        data.mask = data.mask.reshape((-1, 1))
        data.foreground_ind = np.where(data.mask != 0)[0]
        data.background_ind = np.where(data.mask == 0)[0]
        data.mask = data.mask.reshape(resize[0],resize[1])
    # Load images if needed
    if load_imgs and not hasattr(data, 'imgs'):
        data.imgs = None
        for filename in data.filenames:
            img = Image.open(filename)
            img = img.resize(resize, Image.NEAREST)
            img = np.array(img)
            
            # Apply gamma correction
            img = (img / 255.0) ** gamma
            # img = (img * 255).astype(np.uint8)
            img = img.mean(axis=2)
            
            if data.imgs is None:
                height, width = img.shape
                img = img.reshape((-1,1))
                data.imgs = img.reshape((-1, 1))
            else:
                data.imgs = np.append(data.imgs, img.reshape((-1, 1)), axis=1)
                # data.imgs.append(img)
    
    return data

