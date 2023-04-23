import glob
from os import walk
import shutil

src_dir = "/home/fabian/31_TrainingImages/render_13_04/imgs"
dest_dir = "/home/fabian/31_TrainingImages/training_set/imgs"
dest_dir_label = "/home/fabian/31_TrainingImages/training_set/labels"
src_dir_label = "/home/fabian/31_TrainingImages/render_13_04/labels"

filenames = next(walk(src_dir), (None, None, []))[2]  # [] if no file

counter = 0
for filename in filenames:
  file_counter = int(filename[:-4])
  if(file_counter %4 == 0 and counter < 200):
    shutil.copyfile(src_dir+"/"+filename, dest_dir+"/"+filename)
    shutil.copyfile(src_dir_label+"/"+str(file_counter)+".txt", dest_dir_label+"/"+str(file_counter)+".txt")
    print("copied file " + filename)
    counter += 1