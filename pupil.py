# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 09:11:31 2018

@author: fmillour
"""

import numpy as np
from scipy.ndimage import rotate
from matplotlib import pyplot as plt


Dmain = 8.18 # Diameter of main mirror in meters

Dscnd = 1.2 # Diameter of secondary mirror

Dtowr  = 1.5 # Diameter of the tower
D_M3   = 1.1 # Diameter of tertiary mirror
Thk_M3 = 0.2 # Thickness of tertiary mirror

Spid_thk = 0.04;  # Thickness of M2 spider in meters
Spid_angle = 100. # Angle between spider arms in degrees

Duse = 30    # useful region in meters
NAXIS = 2048 # Number of pixels

# Emissivity of different parts of the telescope
em1 = 0.3 # M1 emissivity
em2 = 0.1 # Spider emissivity
em3 = 0.2 # M3 tower emissivity

chopThrow = 0.1; # Fraction of pupil movement when chopping
chopThrowInPixels = int(chopThrow / Duse * NAXIS)

# Define the axis
axis = np.linspace(-Duse/2, Duse/2, NAXIS)
# Calculate a 2D radius out of the 1D (vector) axis
radius = np.sqrt(axis[np.newaxis,:]**2 + axis[:,np.newaxis]**2)

# M1 image
M1 = radius <= Dmain / 2
#plt.imshow(M1)

# M2 image
M2 = radius <= Dscnd / 2
#plt.imshow(M2)

# M3 seen through the side image
M3_imprint = (np.abs((axis+D_M3/2)[np.newaxis,:]) < Thk_M3/2) * (np.abs(axis[:,np.newaxis]) < D_M3/2)
#plt.imshow(M3_imprint)

# M3 tower image
M3_tower = radius <= D_M3 / 2

# Addup M3 and M3 tower to make the M3 cluster
M3_cluster = M3_tower + M3_imprint
M3_cluster = M3_cluster.astype(float)
#plt.imshow(rotate(M3_cluster.astype(float),30))

# Build the spider
spiderArm1 = (np.abs((axis-D_M3/2-Spid_thk/2)[:,np.newaxis]) < Spid_thk/2) * (axis[np.newaxis,:] > 0)
spiderArm2 = (np.abs((axis+D_M3/2+Spid_thk/2)[:,np.newaxis]) < Spid_thk/2) * (axis[np.newaxis,:] < 0)
# Arms 3 and 4 are rotated
spiderArm3 = (np.abs((axis-Spid_thk/2)[np.newaxis,:]) < Spid_thk/2) * ((axis-D_M3/2)[:,np.newaxis] > 0)
spiderArm3 = rotate(spiderArm3.astype(float),90-Spid_angle,reshape=False)
spiderArm4 = (np.abs((axis-Spid_thk/2)[np.newaxis,:]) < Spid_thk/2) * ((axis-D_M3/2)[:,np.newaxis] > 0)
spiderArm4 = rotate(spiderArm4.astype(float),270-Spid_angle,reshape=False)
# The spider is just the sum of all 4 arms
spiderArm = spiderArm1.astype(float) + spiderArm2.astype(float) + spiderArm3 +spiderArm4
#plt.imshow(spiderArm)

# Pupil emissivity
Pupil_1 = em1 * M1 + em2 * M1 * spiderArm + em3 * M3_cluster
plt.figure(1)
plt.imshow(Pupil_1)

# Pupil emissivity with chop throw
Pupil_2 = em1 * M1 + M1 * np.roll(em2 * spiderArm + em3 * M3_cluster,chopThrowInPixels,axis=0)

# Compute a PSF out of pupils, first for chop in
fft_im = np.abs(np.fft.fft2(Pupil_1))
fft_im = np.fft.fftshift(fft_im)**2

# chop out
fft_imc = np.abs(np.fft.fft2(Pupil_2))
fft_imc = np.fft.fftshift(fft_imc)**2

# Display PSF
plt.figure(2)
plt.imshow(fft_im,vmin=0,vmax=1000)
#plt.imshow(fft_im)

# Display PSF 2
plt.figure(3)
plt.imshow(fft_imc,vmin=0,vmax=1000)
#plt.imshow(fft_imc)

# Display PSF differences
plt.figure(4)
plt.imshow(fft_im-fft_imc,vmin=-10,vmax=10000)
#plt.imshow(fft_im-fft_imc)

# Transmissive pupil
Pupil_1_obj = M1 *(1 - spiderArm) * (1 - M3_cluster)

# Display corresponding PSF
plt.figure(5)
plt.imshow(Pupil_1_obj)

