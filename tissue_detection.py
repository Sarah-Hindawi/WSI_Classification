# Code reference: https://www.nature.com/articles/s41379-021-00850-6

import os
import cv2
import gc
import shutil
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage as ski
import skimage.color as color

Image.MAX_IMAGE_PIXELS = None

class TissueDetection():
  """ Load a slide and generate mask of tissue region

  # Arguments
     slide_path 
  
  # Attributes
      path
      name
      slide
      tissue_mask
      bounding_box

  # Methods
      read_level
      generate_mask
      show_overlay
  """    
  def __init__(self, slide_obj):
    self.slide_obj = slide_obj

  def read_level(self, level):
    """ Read a given level of an openslide object 
    
    # Arguments
      level: level of slide to read
   
    # Returns: RGB image of whole slide as a numpy array  
    """
    im = self.slide_obj.read_region((0,0), level, self.slide_obj.level_dimensions[level])
    return np.asarray(im)[:,:,:3]

  def apply_otsu(self, x):
    """  Determine otsue threshold and convert image to a mask """
    threshold = ski.filters.threshold_otsu(x)
    x[x > threshold] = 1
    x[x < threshold] = 0
    return x

  def get_bounding_box(self, mask):
    """ Determines the corners of a rectangle that completely contains a given mask  
    
    # Note:
        this function returns coordinates in the axes that they are used by 
        PIL/openslide, however x and y need to be flipped in order to be      
        used in numpy/matplotlib
    
    # Args:
        mask: a numpy array

    # Returns: a tuple
     """ 
    xax = np.amax(mask, axis = 1)
    yax = np.amax(mask, axis = 0)
    xmin = np.argmax(xax)
    xmax = mask.shape[0] - np.argmax(np.flip(xax, 0))
    ymin = np.argmax(yax)
    ymax = mask.shape[1] - np.argmax(np.flip(yax, 0))
    return (ymin, ymax, xmin, xmax)

  def generate_mask(self, level=2, make_contour=False,
                    invert_background=False, return_mask=False):
    """ Generate the tissue mask 
    
    # Arguments
        level: openslide level on which it is generated
        div: downsample of each side relative to full slide
        contour: returns a contour of tissue rather than a filled mask

    # Returns
        the generated mask (as a class attribute) 
    """
    gc.collect()

    try:
      im = self.read_level(level).astype(np.float16)
    except IndexError as e:
      level -= 1
      im = self.read_level(level).astype(np.float16)
      print(os.path.basename(self.slide_obj._filename), level, e)

    d = self.slide_obj.level_downsamples[level] 
    
    # fixes slides where all the background has been made white
    if invert_background:
      temp = cv2.inRange(im, np.array([253,253,253]), np.array([255,255,255]))
      im[temp==255] = [0,0,0]
    
    im = im.astype(np.float16) 
    grey_img = color.rgb2gray(im)
    del im  # Free up the memory of the original image
    grey_img[grey_img < 0.2] = 1 # get rid of marks on the slide
    grey = self.apply_otsu(grey_img)

    # flip the values
    grey[grey==1] = 2
    grey[grey==0] = 1
    grey[grey==2] = 0
    grey[grey==2] = 0 # intentially running this twice
    
    # set the edges of the slide to 0
    xdim, ydim = grey.shape
    xclip, yclip = int(xdim*0.01), int(ydim*0.01)
    grey[:xclip,:]=0
    grey[-xclip:,:]=0
    grey[:,:yclip]=0
    grey[:,-yclip:]=0
    mask = grey

    # apply sequential binary morphology operations
    #mask = cv2.erode(mask, iterations=2, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, iterations=2,
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    mask = cv2.dilate(mask, iterations=6,
                      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    
    mask = mask.astype(np.uint8)

    self.tissue_mask = Image.fromarray(mask.astype('uint8'), mode='L')
    self.bounding_box = tuple([int(x*d) for x in self.get_bounding_box(mask)])
        
    if make_contour:
      contour = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT,
                                 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
                                 )
      self.contour = Image.fromarray(contour.astype('uint8'), mode='L')
    
    gc.collect()
    if return_mask:
      return self.tissue_mask

  def show_overlay(self, save_path=None, size=10, overlay='mask'):
    """ Generates an overlay of the slide and tissue region (either
    as mask or contours) and then either displays or saves it 
    """
    roi = self.slide_obj.get_thumbnail((5000, 5000))
    if overlay == 'mask':
      mask = self.tissue_mask.resize(roi.size)
    elif overlay == 'contour':
      mask = self.contour.resize(roi.size)
    if roi.size[1] > roi.size[0]:
      roi = roi.rotate(90, expand=True)
      mask = mask.rotate(90, expand=True)
    ratio = int(roi.size[0]/roi.size[1])
    roi = np.asarray(roi)
    mask = np.asarray(mask)
    mask = mask.astype('float16')
    fig = plt.figure(figsize = (size, size*ratio))
    plt.suptitle((os.path.basename(self.slide_obj._filename), self.slide_obj.dimensions, self.bounding_box))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    plt.imshow(roi, interpolation='none')
    if overlay == 'mask':
      plt.imshow(mask, 'viridis', interpolation='none', alpha=0.3)
    elif overlay == 'contour':
      plt.imshow(mask, 'binary', interpolation='none', alpha=0.5)
    fig.tight_layout()
    if save_path:
      plt.savefig(save_path, orientation='landscape')
      plt.close()
    else:
      plt.show()


def process_slides(slide_dir, working_dir): 
  """ Running the program as a script will iterate over every slide in a folder,
  showing first the downsampled slide with overlaid tissue region.
  """
  slide_dict = {}
  for dx in ['lgg', 'gbm']:
    folder = os.path.join(slide_dir, 'tcga_' + dx)
    slide_dict[dx] = [os.path.join(folder, x) for x in os.listdir(folder)
                       if x[-3:] == 'svs']
  save_to_dir = os.path.join(working_dir, 'tissue_masks_imgs') 
  if os.path.exists(save_to_dir):
    shutil.rmtree(save_to_dir)
    os.mkdir(save_to_dir)
  else:
    os.mkdir(save_to_dir)
  slide_list = []
  for key in slide_dict.keys():
    slide_list += slide_dict[key]
  random.shuffle(slide_list)
  if FLAGS.number:
    slides = slide_list[:FLAGS.number]
  else: slides = slide_list
  for s in slides:
    print('\n{}'.format(os.path.basename(s)))
    try:
      td = TissueDetection(s)
      td.generate_mask(make_countour=True)
      save_path = os.path.join(save_to_dir, os.path.basename(s)[:-4] + '.jpg')
      if FLAGS.save == 'mask':
        td.tissue_mask.save(save_path)
      elif FLAGS.save == 'image':
        td.show_overlay(save_path, overlay=FLAGS.overlay)
      print(td.slide_obj.dimensions, td.bounding_box)
      del td
    except IndexError as e:
      print('*** unable to process %s *** due to:' % os.path.basename(s), e)

# ———————————————————————————————————————————————————————————————————

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--number', type=int, default=None,
        help='Number of slides to view')
  parser.add_argument('--save_images', action='store_true',
        help='Save images to file (vs. just displaying')
  parser.add_argument('--overlay', choices=['mask','contour'], default='contour',
        help='Display tissue area as a contoured outline or mask')
  parser.add_argument('--save', choices=['mask', 'image'], default='image')
  FLAGS, unparsed = parser.parse_known_args()
  print(FLAGS)

  process_slides()


