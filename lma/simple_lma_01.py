# -*- coding: utf-8 -*-
##############################################################################
# LMA: A Python Library for generating FT-spectra from
#         Molecular Dynamics Trajectories.
#
# Copyright 2017-2018
# Ruhr-University-Bochum Institute of biophysics and the Author
#
# Author: Matthias Massarczyk
#
# LMA is free software:
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
##############################################################################



import mdtraj
import numpy as np
from lma_core import *


#########################
TRAJ = mdtraj.load('...', top='...')
partial_charges = np.load('...')
dt = ...
win = 'tukey'
wav_save_name = '...'
data_save_name = '...'
#########################



traj_data = TRAJ.xyz * partial_charges[:,:,np.newaxis]
traj_data_zm = zero_mean(traj_data)

fin_len = round_to_next_exponent(traj_data_zm.shape[0])
wav = freq(dt, fin_len)

np.savetxt(wav_save_name, wav)

traj_data_win = apply_window(traj_data_zm, winType=win)
traj_data_pad = pad_zeros(traj_data_win)

traj_data_ft = ftrans(traj_data_pad, wav.shape[0])
traj_data_pft = project_vecs(traj_data_ft)

pm = power_matrix(traj_data_pft)
pf = power_factor(pm, wav)

dfm = distribution_factor_matrix(pf)
sua = sum_up_atoms(pf, dfm)

np.save(data_save_name, sua)

