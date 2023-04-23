from PIL import Image
import glob
from os import walk
import shutil

src_dir = "/home/fabian/31_TrainingImages/training_set/imgs"
dest_dir = "/home/fabian/31_TrainingImages/training_set/masks"


filenames = next(walk(src_dir), (None, None, []))[2]  # [] if no file

counter = 0
for filename in filenames:
  if len(filename) < 16 and "mask" in filename:
    file_counter = int(filename[:-9])
    print(file_counter)

    # open the input image
    input_image = Image.open(src_dir+"/"+str(file_counter)+ "_mask.png")

    # get the width and height of the input image
    width, height = input_image.size

    # get the color values for each pixel in the input image
    pixels = input_image.load()

    # create a dictionary to store the output images
    output_images = {}

    # loop through each pixel in the input image
    for x in range(width):
        for y in range(height):
            # get the color value for the current pixel
            color = pixels[x, y]
            if(not color == (0,0,0)):
              print(color)
              
              # if an output image for this color value doesn't exist yet, create it
              if color not in output_images:
                  output_images[color] = Image.new("L", (width, height), color=0)
              
              # set the pixel in the output image for this color value
              output_images[color].putpixel((x, y), 5)

    # save each output image
    counter = 0
    for color, image in output_images.items():
        counter += 1
        image.save(dest_dir+"/"+str(file_counter)+"_"+str(counter)+"_segmask.png")
