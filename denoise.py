from Tkinter import Tk, Canvas, Label
import ImageTk
import Image
from copy import copy
from time import time
from sys import exit
from math import exp
from random import random, normalvariate as normal

WIDTH = 100
HEIGHT = 100

clamp = lambda x, low, high: low if x < low else (high if x > high else x)

# Load in a WIDTH x HEIGHT image from a text file that provides rows of
# space-separated floating point luminance values, and return the image
# data as a floatmap represented as a list of rows, where each row is
# represented as a list of luminance values.
def ReadImage(path):
  floatmap = []
  for line in open(path).readlines():
    row = [float(x) for x in line.split(' ') if x]
    assert(len(row) == WIDTH)
    floatmap.append(list(row))
  assert(len(floatmap) == HEIGHT)
  return floatmap

# The following functions sample a single pixel value y_ij in each of our
# three models, respectively. The IterateMCMC() function below performs a
# full round of Gibbs sampling, using one of these functions to draw the
# values at each pixel.
def _discrete(neighbors, x_ij, WP, WL):
  # Use the model to get the energy when the pixel takes value c.
  # We only look at the iterand neighbors, and not their originl noisy colors.
  potential = lambda c: sum([WP for z in neighbors if z[1] == c]) + \
                            (WL if x_ij == c else 0.0)
  # Now we can evaluate the Giibs distribution and find the probability that
  # the pixel (i,j) is white.
  Pwhite = exp(potential(1)) / sum([exp(potential(c)) for c in [0,1]])
  return 1.0 if random() < Pwhite else 0.0

def _continuous(neighbors, x_ij, WP, WL):
  # Calculate the parameters for our normal distribution (from equations
  #  8 and 12 in the assignment sheet.)
  variance = 0.5 / (WP * len(neighbors) + WL)
  mean = 2*variance * (WL*x_ij + sum([WP*z[1] for z in neighbors]))
  stdev = variance**0.5
  z = normal(mean, stdev)
  return clamp(mean + z*stdev, 0.0, 1.0)

def _continuous_with_edges(neighbors, x_ij, WP, WL):
  weight = lambda x_kl: WP / (0.01 + (x_ij - x_kl)**2)
  variance = 0.5 / (sum([weight(z[0]) for z in neighbors]) + WL)
  mean = 2*variance * (WL*x_ij + sum([weight(z[0]) * z[1] for z in neighbors]))
  stdev = variance**0.5
  z = normal(mean, stdev)
  return clamp(mean + z*stdev, 0.0, 1.0)

Models = {'Discrete': _discrete,
          'Continuous1': _continuous,
	  'Continuous2': _continuous_with_edges}

# This function takes in a noisy floatmap and performs one round
#   of Gibbs sampling, returning a triple containing:
#     0. an iterand floatmap giving the results of just this round;
#     1. an average floatmap over our iteration thus far (note that
#         this will be the same as the iterand floatmap unless we
#         get an optional 'state' argument;) and
#     2. some representation of the current state so that we can continue.
def IterateMCMC(noisy_floatmap, WP, WL, state=None, pixel_sampler=_discrete):
  if not state:
    state = {'original': copy(noisy_floatmap),
             'average': noisy_floatmap,
             'n': 0.0}
  # Sample a new WIDTH x HEIGHT floatmap
  new_floatmap = []
  for y in range(HEIGHT):
    new_floatmap.append( [ ] )  # Start a new row.
    for x in range(WIDTH):
      # Collect the original & iterand values of all neighbors of this
      # pixel, and watch for boundary conditions.
      neighbors = []
      color_pair = lambda i,j: (state['original'][j][i], noisy_floatmap[j][i])
      if y > 0:
        neighbors.append(color_pair(x, y-1))
      if y < HEIGHT - 1:
        neighbors.append(color_pair(x, y+1))
      if x > 0:
        neighbors.append(color_pair(x-1, y))
      if x < WIDTH - 1:
        neighbors.append(color_pair(x+1, y))
      new_color = pixel_sampler(neighbors, state['original'][y][x], WP, WL)
      new_floatmap[-1].append(new_color)
  # Calculate the new average from the new image, running average & iteration #.
  new_average = []
  n = state['n']
  for (new_row, old_row) in zip(new_floatmap, state['average']):
    new_row = [(n*old + new)/(n+1) for (new,old) in zip(new_row,old_row)]
    new_average.append(new_row)
  # Save our state for re-entry
  state['average'] = new_average
  state['n'] += 1
  return (new_floatmap, new_average, state)

# This function builds a PIL Image object using the color values
# stored in a floatmap. These objects can then be passed to Tk to
# draw the image in the GUI.
def PILImageFromFloatmap(image):
  # Convert the 2D array to a 1D array.
  flattened_list = sum(image, [])
  # Get the colors to integers in the range 0-255.
  discretized_colors = [int(x * 255) for x in flattened_list]
  # Build a string of the luminance values, in hex.
  as_hexstring = ''.join([('%02x' % x) for x in discretized_colors])
  from binascii import unhexlify
  # Convert that string into the binary color values.
  as_binstring = unhexlify(as_hexstring)
  # Now PIL is ready to take over.
  return Image.fromstring('L', (WIDTH,HEIGHT), as_binstring)

# This class lets us build the figures in our GUI -- each CaptionedImage
# object keeps track of its position and dimensions, and we can update
# the image and caption later. The image is scaled to the same width
# passed into the constructor, and is kept a square; the caption is then
# set below on the screen.
class CaptionedImage:
  def __init__(self, root, width, height, left, top):
    self.root = root
    self.dimensions = (width,height)
    self.position = (left,top)
    self.image_label = None
  def set_image(self, image):
    # Fit image to width.
    width = self.dimensions[0]
    image = image.resize( (width,width) )
    # Put the image on a Tk label.
    tkpi = ImageTk.PhotoImage(image)
    image_label = Label(root, image=tkpi)
    # Don't let the image get collected prematurely (from tkinterbook.)
    image_label.tkpi = tkpi
    image_label.place(x=self.position[0], y=self.position[1])
    # See if we have one we need to destroy
    if self.image_label:
      self.image_label.destroy()
    self.image_label = image_label
  def set_caption(self, text):
    # Make another label for the caption.
    self.caption_label = Label(root, text=text)
    left = self.position[0]  # We use the same width as the image, then center.
    width = self.dimensions[0]
    top = self.position[1] + self.dimensions[1] # Set it below the image.
    self.caption_label.place(x=left, y=top, width=width)

# Calculate the MAE between two floatmaps.
def FloatmapMAE(map1, map2):
  total_error = 0.0
  for row_pair in zip(map1, map2):
    total_error += sum([abs(x-y) for (x,y) in zip(*row_pair)])
  return total_error / (WIDTH*HEIGHT)

if __name__ == '__main__':
  from optparse import OptionParser
  parser = OptionParser(usage='usage: %prog test-case [options]',
                        version='CS691 HW 3.0, Joshua Trask',
                        epilog="""
test-case is the path to the text representation of the target image.
The noisy image is assumed to be in the same location, with '-noise'
added before the file extension.""")

  models = Models.keys()
  models.sort()
  models_str = '/'.join(models)
  parser.add_option('-m', '--model', help='model to use (%s)' % models_str,
                    type='choice', choices=Models.keys(), dest='model')
  parser.add_option('-P', help='pairwise parameter WP (default=1.0)',
                    metavar='WP', type='float', default=1.0, dest='WP')
  parser.add_option('-L', help='pixel parameter WL (default=1.0)',
                    metavar='WL', type='float', default=1.0, dest='WL')
  parser.add_option('-i', '--iterations', metavar='max',
                    help='number of iterations (default=100)',
                    type=int, default=100, dest='max_iterations')
  parser.add_option('-w', '--wait', metavar='t', type='int', default=0,
                    help='seconds to wait after each iteration (default=0)',
                    dest='seconds_per_update')
  parser.add_option('-o', '--out', metavar='file', type='string',
                    help="""optional output path in any PIL-supported format"""
                         """(png recommended)""", dest='output_path')
  parser.add_option('-f', metavar='k', type='int',
                    help='if set, save t-step images at each t=nk',
                    dest='output_freq')
  (options, args) = parser.parse_args()
  if len(args) < 1 or options.model is None:
    parser.print_help()
    exit()
  
  testcase = args[0]
  iteration_state = None
  iteration = 0
  model = Models[options.model]
  # Set up the UI.
  root = Tk()
  root.geometry('%dx%d' % (850, 235))
  root.title('MCMC Denoiser - %s' % testcase)
  # Set up the figures in the UI. First the noisy version:
  import os
  path_parts = list(os.path.splitext(testcase))
  path_parts.insert(1, '-noise')
  noisy_floatmap = ReadImage(''.join(path_parts))
  noisy_figure = CaptionedImage(root, 200, 200, 10, 10)
  noisy_figure.set_image(PILImageFromFloatmap(noisy_floatmap))
  noisy_figure.set_caption('Original image')
  # Then the current iterand (which just begins as a copy of the noisy image.)
  iterand_floatmap = copy(noisy_floatmap)
  iterand_figure = CaptionedImage(root, 200, 200, 220, 10)
  iterand_figure.set_image(PILImageFromFloatmap(iterand_floatmap))
  iterand_figure.set_caption('Current iterand (i=%d)' % iteration)
  # Next, the running average (which also starts as a noisy copy.)
  average_floatmap = copy(iterand_floatmap)
  average_figure = CaptionedImage(root, 200, 200, 430, 10)
  average_figure.set_image(PILImageFromFloatmap(average_floatmap))
  average_figure.set_caption('Running average')
  # Finally, add the figure for our denoising target.
  good_floatmap = ReadImage(testcase)
  good_figure = CaptionedImage(root, 200, 200, 640, 10)
  good_figure.set_image(PILImageFromFloatmap(good_floatmap))
  good_figure.set_caption('Target image')
  next_iteration = time()
  
  while iteration < options.max_iterations:
    if next_iteration < time():
      iteration += 1
      next_iteration = time() + options.seconds_per_update
      iterand_floatmap, average_floatmap, iteration_state = \
          IterateMCMC(iterand_floatmap,
                      options.WP, options.WL,
                      iteration_state,
                      model)
      iterand_image = PILImageFromFloatmap(iterand_floatmap)
      average_image = PILImageFromFloatmap(average_floatmap)
      mae = FloatmapMAE(average_floatmap, good_floatmap)

      if options.output_path:
        if options.output_freq and iteration % options.output_freq == 0:
          path_parts = list(os.path.splitext(options.output_path))
          path_parts.insert(-1, '.%d' % iteration)
          tstep_path = ''.join(path_parts)
          average_image.save(''.join(path_parts))
          print "Saved %d-step image to %s" % (iteration, tstep_path)
          print "MAE", mae
        if iteration == options.max_iterations:
          average_image.save(options.output_path)
          print "Saved final image to", options.output_path
          print "MAE", mae
      try:
        iterand_figure.set_caption('Current iterand (i=%d)' % iteration)
        iterand_figure.set_image(iterand_image)
        mae = FloatmapMAE(average_floatmap, good_floatmap)
        average_figure.set_caption('Running average (MAE: %0.3f)' % mae)
        average_figure.set_image(average_image) 
        root.update()
      except:
        # There's nothing we can really do about exceptions from Tk, and
        # besides they mostly just come from quitting early.
        exit()
  root.mainloop()
