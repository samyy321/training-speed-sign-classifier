import collections
import os
import glob
import re
import numpy as np
from PIL import Image

def create_image_lists(img_dir):
	"""
	Builds a list of training images from the file system.

	Analyzes the sub folders in the image directory,
	uses their names as labels, get images paths and create hot encoding.

	Returns:
	A dictionary containing a list of images and hot encoding for each label.
	"""
	if not os.path.isdir(img_dir):
		print("Image directory '" + img_dir + "' not found.")
		return None

	result = collections.OrderedDict()
	sub_dirs = [os.path.join(img_dir, item) for item in os.listdir(img_dir)]
	sub_dirs = sorted(item for item in sub_dirs if os.path.isdir(item))
	classes_count = len(sub_dirs)

	for i, sub_dir in enumerate(sub_dirs):
		images = []
		extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
		file_list = []
		dir_name = os.path.basename(sub_dir)
		print("Looking for images in '" + dir_name + "'")
		for extension in extensions:
			file_glob = os.path.join(img_dir, dir_name, '*.' + extension)
			file_list.extend(glob.glob(file_glob))
		if not file_list:
			print('No files found')
			continue
		label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
		for file_name in file_list:
			images.append(file_name)
		hot_encoding = [1 if x == i else 0 for x in range(0, classes_count)]
		result[label_name] = {
			'images': images,
			'hot_encoding': hot_encoding,
		}
	return result

def load_features_and_targets(img_lists):
	"""
	Take a dictionary containing labels and corresponding images lists and
	hot encoding and return features and targets numpy arrays.

	A feature in features[i] has his corresponding hot encoding in
	targets[i].
	"""
	features = []
	targets = []

	for label in img_lists:
		for img in img_lists[label]['images']:
			features.append(np.array(Image.open(img).resize((32, 32))))
			targets.append(img_lists[label]['hot_encoding'])

	features = np.array(features)
	targets = np.array(targets)
	return features, targets
