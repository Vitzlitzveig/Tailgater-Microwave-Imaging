#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv	# Bessel functions of 1sr kind
from scipy.signal import convolve2d


def load_data(fname):
	data =  np.loadtxt(fname)

	# Shift lines
	data[1::2] = np.roll(data[1::2], 3, axis=1)

	# Restore edges

	# Extraploate top edge
	data[0] = 2*data[1] - data[2]
	data[0,3] = (data[0,2]+data[0,4])/2
	data[0,-1] = 2*data[0,-2] - data[0,-3]

	# Interpolate checkers on the left side
	data[1:-2:2, :4] = (data[0:-2:2, :4] + data[2::2, :4])/2
	data[-1, :4] = 2*data[-2,:4] - data[-3,:4]

	# Interpolate checkers on the right side
	data[2::2, -1] = (data[1:-1:2, -1] + data[3::2, -1])/2

	# Normalize
	# data -= data.min()
	# data/=data.max()
	return data


def progressbar(i, total):
	I = i+1
	bars = 30
	pos = int(I*bars / total)
	s = '[' + '#'*pos + ' '*(bars-pos)+f'] {I}/{total}'
	print('\r' + s, end='')
	if I == total: print()


def beam_form(Ti, Tj=0, f=11e9, D=0.40):
	'''
	Radiation pattern for a parabolic antenna
	Ti, Tj - distances in i- and j- directions from centaral axis, degrees.
	f - wave frequency, Hz
	D - diameter of antenna
	'''
	c = 3e8 # Speed of light
	T0 = np.pi*D*f/c	# pi/Size of the main beam in radians
	rd = 180/np.pi 	# Radian to degree coefficient

	if Tj is None:
		T = Ti
	else:
		T = (Ti**2 + Tj**2)**0.5
	# Avoid division by zero
	if abs(T) < 1e-6: return 1.
	kaSinT = T0 * np.sin(T/rd)
	A = (1 + np.cos(T/rd)) * jv(1, kaSinT) / kaSinT
	return abs(A.real)


def beam_mask(shape, deg_in_pix=1.):
	beam = np.zeros(shape)
	i0, j0 = map(lambda x: x//2, shape)
	for i in range(shape[0]):
		for j in range(shape[1]):
			beam[i,j] = beam_form((i-i0)*deg_in_pix, (j-j0)*deg_in_pix)

	return beam


def check_beam(fname, i0, j0, N=21):
	'''
	Check if theoretical beam mathces with the image
	i0, j0 - coordinates of a beam center, pixels
	N - size of sample, pixels
	'''
	data = load_data(fname)
	beam = beam_mask((N, N))

	_i0 = i0 - beam.shape[0]//2
	_j0 = j0 - beam.shape[1]//2
	part = data[_i0:_i0+beam.shape[0],_j0:_j0+beam.shape[1]]
	part -= part.min()
	part /= part.max()

	part[N//2:,:N//2] = np.nan
	part[:N//2, N//2:] = np.nan

	part[np.isnan(part)] = beam[np.isnan(part)]
	plt.title('Theoretical beam (btn-left, top-right)\nshould match the image (tor-left,btn-right)')
	plt.imshow(part, cmap='plasma')
	plt.show()


def RL_deconv(data, kernel, iters=100, eps=1e-8, u0=None):
	'''
	Richardson-Lucy deconvolution
	'''

	# A bit better convolution
	fix = convolve2d(np.ones_like(data), kernel, mode='same')
	conv = lambda d: convolve2d(d, kernel, mode='same')/fix

	# Normalize data into range [1,2]
	# to avoid divisions by zero during iterations
	data1 = data - data.min()
	data1 /= data1.max()
	data1 += 1

	if u0 is None:
		u = data1.copy()
	else:
		u = u0.copy()

	for I in range(iters):
		progressbar(I, iters)
		k = conv(data1/(conv(u)))
		u *= k
		n = np.linalg.norm(k-1, np.inf)
		if n < eps: break

	# Restore initial scale
	u -= 1
	u *= data.max() -  data.min()
	u += data.min()

	return u


def sharpen_image(fname):
	data = load_data(fname)
	beam = beam_mask((71, 71))
	beam /= beam.sum()


	reconstruction = RL_deconv(data, beam, iters=1000)
	np.savetxt(fname[:-4]+'-reconstructed.txt', reconstruction)

	plt.subplot(211)
	plt.title('Original')
	plt.imshow(data, cmap='plasma')
	plt.xlabel("Azimuth (dish uses CCW heading)")
	plt.ylabel("Elevation")
	plt.colorbar()

	plt.subplot(212)
	plt.title('Reconstructed')
	plt.imshow(reconstruction, cmap='plasma')
	plt.xlabel("Azimuth (dish uses CCW heading)")
	plt.ylabel("Elevation")
	plt.colorbar()

	plt.show()


if __name__ == '__main__':
	fname = 'examples/raw-data-20230321-193653.txt'
	# Position of the most distinguishable sattelite - (39,48)
	check_beam(fname, i0=39, j0=48)
	print('Shaprening, wait a minute...')
	sharpen_image(fname)
