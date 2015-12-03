import code
import numpy as np
import matplotlib.pyplot as plt
import skimage.util as skim

from skimage import exposure

import fileio

def split_kp(kp):
		x_kp = kp[0:len(kp):2]
		y_kp = kp[1:len(kp):2]
		return x_kp, y_kp

def merge_kp(x_kp, y_kp):
	return [j for i in zip(x_kp, y_kp) for j in i]

def flip(face, kp, horiz = True):
	if horiz:
		face_flip = np.fliplr(face)
		x_kp, y_kp = split_kp(kp)
		x_kp = face.shape[0] - x_kp
		y_kp = y_kp  #[::-1]
		return face_flip, merge_kp(x_kp, y_kp)
	else:
		face_flip = np.flipud(face)
		x_kp, y_kp = split_kp(kp)
		x_kp = x_kp
		y_kp = face.shape[1] - y_kp
		return face_flip, merge_kp(x_kp, y_kp)

def rotate(face, kp, clockwise = True):
	if clockwise:
		face_rot = np.rot90(face, 3)
		x_kp, y_kp = split_kp(kp)
		return face_rot, merge_kp(face.shape[0] - y_kp, x_kp)
	else:
		face_rot = np.rot90(face, 1)
		x_kp, y_kp = split_kp(kp)
		return face_rot, merge_kp(y_kp, face.shape[1] - x_kp)

def enhance_contrast(face):
	hist = np.histogram(face, bins=np.arange(0, 256))
	glob = exposure.equalize_hist(face) * 255
	return glob

### TEST ###

# faces = fileio.FaceReader("../data/training.csv", "../data/training.pkl.gz", 
# 	fast_nrows=10)
# faces.load_file()
# raw_data = faces.get_data()
# to_keep = ~(np.isnan(raw_data['Y']).any(1))
# X = raw_data['X'][to_keep]
# Y = raw_data['Y'][to_keep]

# face = X[9][0]
# kp = Y[9]
# x_kp, y_kp = split_kp(kp)

# code.interact(local=locals())

# face_flip1, x_kp_flip1, y_kp_flip1 = flip(face, kp, True)
# face_flip2, x_kp_flip2, y_kp_flip2 = flip(face, kp, False)
# face_rot1, x_kp_rot1, y_kp_rot1 = rotate(face, kp, True)
# face_rot2, x_kp_rot2, y_kp_rot2 = rotate(face, kp, False)

# fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), 
# 	(ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3)

# ax1.imshow(face, cmap=plt.cm.Greys_r)
# ax1.plot(x_kp, y_kp, 'rx')
# ax1.axis('off')
# ax1.set_title('Original', fontsize=20)

# ax2.imshow(face_flip1, cmap=plt.cm.gray)
# ax2.plot(x_kp_flip1, y_kp_flip1, 'rx')
# ax2.axis('off')
# ax2.set_title('Flipped Horizontal', fontsize=20)

# ax3.imshow(face_flip2, cmap=plt.cm.gray)
# ax3.plot(x_kp_flip2, y_kp_flip2, 'rx')
# ax3.axis('off')
# ax3.set_title('Flipped Vertical', fontsize=20)

# ax4.imshow(face_rot1, cmap=plt.cm.gray)
# ax4.plot(x_kp_rot1, y_kp_rot1, 'rx')
# ax4.axis('off')
# ax4.set_title('Rotated Clockwise', fontsize=20)

# ax5.imshow(face_rot2, cmap=plt.cm.gray)
# ax5.plot(x_kp_rot2, y_kp_rot2, 'rx')
# ax5.axis('off')
# ax5.set_title('Rotated Counter Clockwise', fontsize=20)

# ax6.imshow(skim.random_noise(face, mode='gaussian'), cmap=plt.cm.gray)
# ax6.plot(x_kp, y_kp, 'rx')
# ax6.axis('off')
# ax6.set_title('Random Noise', fontsize=20)

# ax7.imshow(skim.random_noise(face, mode='s&p'), cmap=plt.cm.gray)
# ax7.plot(x_kp, y_kp, 'rx')
# ax7.axis('off')
# ax7.set_title('Salt and Pepper', fontsize=20)

# ax8.imshow(enhance_contrast(face), cmap=plt.cm.gray)
# ax8.plot(x_kp, y_kp, 'rx')
# ax8.axis('off')
# ax8.set_title('High Contrast', fontsize=20)

# plt.show()