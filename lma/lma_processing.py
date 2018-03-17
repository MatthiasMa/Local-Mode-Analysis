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
import os
from lma_core import spectrum as spec
from lma_core import mode_spectrum as mspec
from lma_core import gaussian_filter as gfilter
from lma_core import get_windowSlices as getTW
from lma_core import pos2vel2acc, mode_spectrum_traj, mode_spec, fuse_mode_spectrum
from lma_core import switch_by_bond, switch_by_dist

class LMAProcessor:

    def __init__(self, loader, args):

        self.loader = loader
        self.traj = None
        self.args = args
        self.freq = None
        self.allspec = None
        self.allspec_f = None
        self.allspec_b = None
        self.proj = True
        self.diag = False
        self.skip_full = False
        self.skip_atom = False
        self.skip_mode = False
        self.TWSlices = None
        self.Bonds = None
        self.modespec_tw_b = None
        self.rproj = False # => False
        #self.rproj = True # => False
        self.reshape()
        self.logic()

    def spectrum(self):
        switch_traj = None
        if self.args.switch_bond is not None or self.args.switch_dist is not None:
            if self.loader.TRAJ is None and self.args.switch_bond is not None:
                switch_traj = None
            elif self.loader.TRAJ is not None and self.args.switch_bond is not None:
                switch_traj = switch_by_bond(self.loader.TRAJ, self.args.switch_bond, switch_traj, guess=self.args.guess_bonds_off)
            if self.args.switch_dist is not None:
                switch_traj = switch_by_dist(self.loader.TRAJ, self.args.switch_dist, self.args.switch_dist_arg, switch_traj)
        
        if switch_traj is not None:
            sigma = 4
            print('sigma:', sigma)
            if self.args.save_switch:
                np.savetxt(self.args.out+'switch_traj_raw.txt', switch_traj)
            switch_traj = gfilter(switch_traj, sigma=sigma)
            if self.args.save_switch:
                np.savetxt(self.args.out+'switch_traj.txt', switch_traj)
        
        if (self.args.spec or self.args.spec_plot or
            self.args.atom_spec or self.args.atom_spec_plot or
            self.args.mode_spec or self.args.mode_spec_plot):
            if not self.skip_full: 

                print('\tprocessing spectra...')
                self.freq, self.allspec = spec(self.traj, win=self.args.win, proj=self.proj,
                    diag=self.diag, dt=self.args.dt, wav=self.freq, scale=self.args.fscale,
                    switch_arr=switch_traj, rproj=self.rproj)
                
                if self.args.gaussian_filter != 0:
                    try:
                        for dim in range(self.allspec.shape[2]):
                            for elem in range(self.allspec.shape[1]):
                                self.allspec[:,elem,dim] = gfilter(
                                        self.allspec[:,elem,dim], sigma=self.args.gaussian_filter)
                    except IndexError:
                        for elem in range(self.allspec.shape[1]):
                            self.allspec[:,elem] = gfilter(
                                    self.allspec[:,elem], sigma=self.args.gaussian_filter)

            if self.args.mode_spec or self.args.mode_spec_plot:
                if self.loader.TRAJ is not None:
                    print('\tprocessing modes...')
                    Vel, Acc = pos2vel2acc(self.loader.TRAJ.xyz)
                    Pos = self.loader.TRAJ.xyz
                    self.freq, self.spec_b, self.spec_p, self.spec_n, self.Bonds = mspec(
                                            self.loader.TRAJ, Pos, Vel, dt=self.args.dt,
                                            power='omega2', switch_arr=switch_traj,
                                            guess=self.args.guess_bonds_off)
                    #for bnd in self.Bonds:
                    #    print(bnd)
                    self.spec_b, self.spec_p, self.spec_n = fuse_mode_spectrum(self.allspec, self.spec_b, self.spec_p, self.spec_n, self.Bonds, self.args.gaussian_filter)
                    self.Bond_names = bond_naming(self.Bonds)
                    if self.args.gaussian_filter != 0:
                        try:
                            for dim in range(self.spec_b.shape[2]):
                                for elem in range(self.spec_b.shape[1]):
                                    self.spec_b[:,elem,dim] = gfilter(
                                            self.spec_b[:,elem,dim], sigma=self.args.gaussian_filter)
                                    self.spec_p[:,elem,dim] = gfilter(
                                            self.spec_p[:,elem,dim], sigma=self.args.gaussian_filter)
                                    self.spec_n[:,elem,dim] = gfilter(
                                            self.spec_n[:,elem,dim], sigma=self.args.gaussian_filter)
                        except IndexError:
                            for elem in range(self.spec_b.shape[1]):
                                self.spec_b[:,elem] = gfilter(
                                        self.spec_b[:,elem], sigma=self.args.gaussian_filter)
                                self.spec_p[:,elem] = gfilter(
                                        self.spec_p[:,elem], sigma=self.args.gaussian_filter)
                                self.spec_n[:,elem] = gfilter(
                                        self.spec_n[:,elem], sigma=self.args.gaussian_filter)




    def spectrum_tw(self):
        switch_traj = None
        if self.args.switch_bond is not None or self.args.switch_dist is not None:
            if self.loader.TRAJ is None and self.args.switch_bond is not None:
                switch_traj = None
            elif self.loader.TRAJ is not None and self.args.switch_bond is not None:
                switch_traj = switch_by_bond(self.loader.TRAJ, self.args.switch_bond, switch_traj, guess=self.args.guess_bonds_off)
            if self.args.switch_dist is not None:
                switch_traj = switch_by_dist(self.loader.TRAJ, self.args.switch_dist, self.args.switch_dist_arg, switch_traj)
        
        if switch_traj is not None:
            sigma = 4
            print('sigma:', sigma)
            if self.args.save_switch:
                np.savetxt(self.args.out+'switch_traj_raw.txt', switch_traj)
            switch_traj = gfilter(switch_traj, sigma=sigma)
            if self.args.save_switch:
                np.savetxt(self.args.out+'switch_traj.txt', switch_traj)
        
        if (self.args.spec or self.args.spec_plot or
            self.args.atom_spec or self.args.atom_spec_plot or
            self.args.mode_spec or self.args.mode_spec_plot):
            
            self.TWSlices = getTW(self.loader.traj_data.shape[0], self.args.n_time_windows,
                                windowwidth=self.args.time_window_length)
            if not self.skip_full:
                print('\tprocessing spectra...')
                self.allspec_tw = []
                for idx, tw in enumerate(self.TWSlices):
                    print('\t\tproc. spec. - time window', idx,'/', self.args.n_time_windows, ':', *tw)
                    self.freq, self.allspec = spec(self.traj[tw[0]:tw[1],:,:], win=self.args.win, proj=self.proj,
                        diag=self.diag, dt=self.args.dt, wav=self.freq, scale=self.args.fscale, switch_arr=(switch_traj[tw[0]:tw[1]] if switch_traj is not None else None))
                    self.allspec_tw.append(self.allspec)

                if self.args.gaussian_filter != 0:

                    for ii, tw in enumerate(self.TWSlices):
                        try:
                            for dim in range(self.allspec_tw[ii].shape[2]):
                                for elem in range(self.allspec_tw[ii].shape[1]):
                                    self.allspec_tw[ii][:,elem,dim] = gfilter(
                                            self.allspec_tw[ii][:,elem,dim], sigma=self.args.gaussian_filter)
                        except IndexError:
                            for elem in range(self.allspec_tw[ii].shape[1]):
                                self.allspec_tw[ii][:,elem] = gfilter(
                                        self.allspec_tw[ii][:,elem], sigma=self.args.gaussian_filter)

            if self.args.mode_spec or self.args.mode_spec_plot:
                if self.loader.TRAJ is not None:
                    print('\tprocessing modes...')
                    Vel, Acc = pos2vel2acc(self.loader.TRAJ.xyz)
                    Pos = self.loader.TRAJ.xyz
                    BONDLenTraj, NORMLenTraj, INPLANELenTraj, self.Bonds = mode_spectrum_traj(self.loader.TRAJ, Pos, Vel, guess=self.args.guess_bonds_off)
                    if self.args.switch_bond is not None or self.args.switch_dist is not None:
                        BONDLenTraj *= switch_traj[:,np.newaxis]
                        INPLANELenTraj *= switch_traj[:,np.newaxis]
                        NORMLenTraj *= switch_traj[:,np.newaxis]
                    self.modespec_tw_b = []
                    self.modespec_tw_p = []
                    self.modespec_tw_n = []
                    for tw_idx, tw in enumerate(self.TWSlices):
                        print('\t\tproc. modes - time window', tw_idx, '/', self.args.n_time_windows, ':', *tw)
                        self.freq, self.spec_b, self.spec_p, self.spec_n = mode_spec(
                                        BONDLenTraj[tw[0]:tw[1],:],
                                        NORMLenTraj[tw[0]:tw[1],:],
                                        INPLANELenTraj[tw[0]:tw[1],:], dt=self.args.dt, power='omega2')
                        self.spec_b, self.spec_p, self.spec_n = fuse_mode_spectrum(self.allspec_tw[tw_idx], self.spec_b, self.spec_p, self.spec_n, self.Bonds, self.args.gaussian_filter)
                        self.modespec_tw_b.append(self.spec_b)
                        self.modespec_tw_p.append(self.spec_p)
                        self.modespec_tw_n.append(self.spec_n)
                    self.Bond_names = bond_naming(self.Bonds)
                        
                    if self.args.gaussian_filter != 0:
                        for ii, tw in enumerate(self.TWSlices):
                            try:
                                for dim in range(self.modespec_tw_b[ii].shape[2]):
                                    for elem in range(self.modespec_tw_b[ii].shape[1]):
                                        self.modespec_tw_b[ii][:,elem,dim] = gfilter(
                                                self.modespec_tw_b[ii][:,elem,dim], sigma=self.args.gaussian_filter)
                                        self.modespec_tw_p[ii][:,elem,dim] = gfilter(
                                                self.modespec_tw_p[ii][:,elem,dim], sigma=self.args.gaussian_filter)
                                        self.modespec_tw_n[ii][:,elem,dim] = gfilter(
                                                self.modespec_tw_n[ii][:,elem,dim], sigma=self.args.gaussian_filter)
                            except IndexError:
                                for elem in range(self.modespec_tw_b[ii].shape[1]):
                                    self.modespec_tw_b[ii][:,elem] = gfilter(
                                            self.modespec_tw_b[ii][:,elem], sigma=self.args.gaussian_filter)
                                    self.modespec_tw_p[ii][:,elem] = gfilter(
                                            self.modespec_tw_p[ii][:,elem], sigma=self.args.gaussian_filter)
                                    self.modespec_tw_n[ii][:,elem] = gfilter(
                                            self.modespec_tw_n[ii][:,elem], sigma=self.args.gaussian_filter)





    def reshape(self):
        if self.loader.traj_data.ndim == 1:
            self.traj = self.loader.traj_data[:,np.newaxis,np.newaxis]
        elif self.loader.traj_data.ndim == 2:
            if self.loader.traj_data.shape[1] == 3:
                self.traj = self.loader.traj_data[:,np.newaxis,:]
            elif self.loader.traj_data.shape[1] == 4:
                self.traj = self.loader.traj_data[:,np.newaxis,:3]
            else:
                self.traj = self.loader.traj_data[:,:,np.newaxis]
        elif self.loader.traj_data.ndim == 3:
            self.traj = self.loader.traj_data

    def logic(self):
        if self.args.weighting is None:
            self.diag = True
            self.proj = False
        if self.args.coupling_off is False:
            self.diag = True
            self.proj = False




def bond_naming(bonds):
    return ['_'.join([str(elem).replace("'", '_1_').replace('"', '_2_') for elem in bnd]) for bnd in bonds]

