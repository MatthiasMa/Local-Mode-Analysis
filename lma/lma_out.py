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


import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 16
fig_size[1] = 8
plt.rcParams['lines.linewidth'] = 3.0
import os
from lma_core import get_band_pass_filter_traj as bpfilter
import mdtraj

def rotate(Mover, Target):
    Cov = np.dot(np.transpose(Mover), Target)
    Vmat, Smat, Wmat = np.linalg.svd(Cov)
    decide_not_rh = (np.linalg.det(Vmat) * np.linalg.det(Wmat)) < 0.0
    if decide_not_rh:
        Smat[-1] = -Smat[-1]
        Vmat[:,-1] = -Vmat[:,-1]
    RotMatrix = np.dot(Vmat, Wmat)
    return np.dot(Mover, RotMatrix)


class OutputWriter:

    def __init__(self, proc, args):
        self.proc = proc
        self.args = args
        if not self.args.bpf:
            np.save(self.args.out+'wavenum.npy', self.proc.freq)
            np.savetxt(self.args.out+'wavenum.txt', self.proc.freq)

        if self.args.n_time_windows > 1:
            with open(self.args.out+'time_windows.info', 'w') as outStream:
                for idx, tw in enumerate(self.proc.TWSlices):
                    outStream.write('{} {}\n'.format(*tw))
        
        if self.args.atom_spec:
            with open(self.args.out+'atoms.info', 'w') as outStream:
                for idx, atm in enumerate(self.proc.loader.name_data):
                    outStream.write('{}\n'.format(atm))
        
        if self.args.mode_spec:
            if self.proc.Bonds is not None:
                with open(self.args.out+'bonds.info', 'w') as outStream:
                    for idx, bnd in enumerate(self.proc.Bonds):
                        outStream.write('{} {}\n'.format(*bnd))
    
    def write(self):
        if not self.args.bpf:
            if self.args.spec:
                outarr = np.concatenate((self.proc.freq[:,np.newaxis], self.proc.allspec.sum(1)[:,np.newaxis]), axis=1)
                if self.args.save_txt: 
                    np.savetxt(self.args.out+'fullspec.txt', outarr)
                if self.args.save_npy: 
                    np.save(self.args.out+'fullspec.npy', outarr)

            if self.args.atom_spec:
                outarr = np.concatenate((self.proc.freq[:,np.newaxis], self.proc.allspec), axis=1)
                if self.args.save_txt: 
                    np.savetxt(self.args.out+'atomspec.txt', outarr)
                if self.args.save_npy: 
                    np.save(self.args.out+'atomspec.npy', outarr)

            if self.args.mode_spec:
                if self.proc.loader.TRAJ is not None:
                    outarr_b = np.concatenate((self.proc.freq[:,np.newaxis], self.proc.spec_b), axis=1)
                    outarr_p = np.concatenate((self.proc.freq[:,np.newaxis], self.proc.spec_p), axis=1)
                    outarr_n = np.concatenate((self.proc.freq[:,np.newaxis], self.proc.spec_n), axis=1)
                    if self.args.save_txt: 
                        np.savetxt(self.args.out+'modespec_b.txt', outarr_b)
                        np.savetxt(self.args.out+'modespec_p.txt', outarr_p)
                        np.savetxt(self.args.out+'modespec_n.txt', outarr_n)
                    if self.args.save_npy: 
                        np.save(self.args.out+'modespec_b.npy', outarr_b)
                        np.save(self.args.out+'modespec_p.npy', outarr_p)
                        np.save(self.args.out+'modespec_n.npy', outarr_n)

        if self.args.bpf:
            print('\tgenerating bpf trajectory...')
            BPF_TRAJ = bpfilter(self.proc.loader.TRAJ,
                    self.args.dt, *self.args.bpf_range, scale=self.args.fscale)
            par = int(round(self.args.bpf_part*BPF_TRAJ.n_frames))
            BPF_TRAJ.center_coordinates(mass_weighted=True)
            BPF_TRAJ = BPF_TRAJ[par:-par]
            for ii in range(1, BPF_TRAJ.n_frames):
                BPF_TRAJ.xyz[ii,:,:] = rotate(BPF_TRAJ.xyz[ii,:,:], BPF_TRAJ.xyz[0,:,:])
            BPF_TRAJ[0].save_pdb(self.args.out+'traj.pdb')
            BPF_TRAJ.save_dcd(self.args.out+'traj.dcd')
            for pii in range(int(np.ceil(BPF_TRAJ.n_frames/self.args.bpf_length))):
                BPF_TRAJ[pii*self.args.bpf_length:(pii+1)*self.args.bpf_length].save_pdb(
                        self.args.out+'traj({}-{})_({}_{}).pdb'.format(
                            int(round(self.args.bpf_range[0])),
                            int(round(self.args.bpf_range[1])),
                            pii*self.args.bpf_length+par,
                            (pii+1)*self.args.bpf_length+par))

    def plot(self):
        if not self.args.bpf:
            if self.args.spec_plot:
                plt.plot(self.proc.freq, self.proc.allspec.sum(1))
                plt.savefig(self.args.out+'fullspec.png')
                plt.cla()

            if self.args.atom_spec_plot:
                print('\tplotting spectra...')
                for atm_idx in range(self.proc.allspec.shape[1]):
                    plt.plot(self.proc.freq, self.proc.allspec[:,atm_idx],
                                        label=self.proc.loader.name_data[atm_idx])
                    plt.legend()
                    plt.savefig(self.args.out+'atomspec_{}.png'.format(self.proc.loader.name_data[atm_idx]))
                    plt.cla()

            if self.args.mode_spec_plot:
                if self.proc.loader.TRAJ is not None:
                    print('\tplotting modes...')
                    for bnd_idx in range(len(self.proc.Bonds)):
                        plt.plot(self.proc.freq, self.proc.spec_b[:,bnd_idx],
                                            label='b_'+self.proc.Bond_names[bnd_idx])
                        plt.legend()
                        plt.savefig(self.args.out+'modespec_b_{}.png'.format(self.proc.Bond_names[bnd_idx]))
                        plt.cla()

                        plt.plot(self.proc.freq, self.proc.spec_p[:,bnd_idx],
                                            label='p_'+self.proc.Bond_names[bnd_idx])
                        plt.legend()
                        plt.savefig(self.args.out+'modespec_p_{}.png'.format(self.proc.Bond_names[bnd_idx]))
                        plt.cla()

                        plt.plot(self.proc.freq, self.proc.spec_n[:,bnd_idx],
                                            label='n_'+self.proc.Bond_names[bnd_idx])
                        plt.legend()
                        plt.savefig(self.args.out+'modespec_n_{}.png'.format(self.proc.Bond_names[bnd_idx]))
                        plt.cla()


    def write_tw(self):
        if not self.args.bpf:

            if self.args.save_npy: 
                if self.args.spec:
                    spec = np.zeros((len(self.proc.allspec_tw), self.proc.allspec_tw[0].shape[0], 1))
                    for idx, tw in enumerate(self.proc.TWSlices):
                        spec[idx,:,0] += self.proc.allspec_tw[idx].sum(1)
                    np.save(self.args.out+'fullspec_tw.npy', spec)

                if self.args.atom_spec:
                    atom_spec = np.zeros((len(self.proc.allspec_tw), *self.proc.allspec_tw[0].shape))
                    for idx, tw in enumerate(self.proc.TWSlices):
                        atom_spec[idx,:,:] = self.proc.allspec_tw[idx]
                    np.save(self.args.out+'atomspec_tw.npy', atom_spec)

                if self.args.mode_spec and (self.proc.modespec_tw_b is not None):
                    mode_spec_b = np.zeros((len(self.proc.modespec_tw_b), *self.proc.modespec_tw_b[0].shape))
                    mode_spec_p = np.zeros((len(self.proc.modespec_tw_b), *self.proc.modespec_tw_b[0].shape))
                    mode_spec_n = np.zeros((len(self.proc.modespec_tw_b), *self.proc.modespec_tw_b[0].shape))
                    for idx, tw in enumerate(self.proc.TWSlices):
                        mode_spec_b[idx,:,:] = self.proc.modespec_tw_b[idx]
                        mode_spec_p[idx,:,:] = self.proc.modespec_tw_p[idx]
                        mode_spec_n[idx,:,:] = self.proc.modespec_tw_n[idx]

                    np.save(self.args.out+'modespec_b_tw.npy', mode_spec_b)
                    np.save(self.args.out+'modespec_p_tw.npy', mode_spec_p)
                    np.save(self.args.out+'modespec_n_tw.npy', mode_spec_n)

        for idx, tw in enumerate(self.proc.TWSlices):
            signum = str(idx).zfill(len(str(len(self.proc.TWSlices))))
            if not self.args.bpf:

                if self.args.spec:
                    if self.args.save_txt: 
                        outarr = np.concatenate((self.proc.freq[:,np.newaxis], self.proc.allspec_tw[idx].sum(1)[:,np.newaxis]), axis=1)
                        np.savetxt(self.args.out+'fullspec_par{}.txt'.format(signum), outarr)

                if self.args.atom_spec:
                    if self.args.save_txt: 
                        outarr = np.concatenate((self.proc.freq[:,np.newaxis], self.proc.allspec_tw[idx]), axis=1)
                        np.savetxt(self.args.out+'atomspec_par{}.txt'.format(signum), outarr)

                if self.args.mode_spec:
                    if self.args.save_txt: 
                        outarr_b = np.concatenate((self.proc.freq[:,np.newaxis], self.proc.modespec_tw_b[idx]), axis=1)
                        outarr_p = np.concatenate((self.proc.freq[:,np.newaxis], self.proc.modespec_tw_p[idx]), axis=1)
                        outarr_n = np.concatenate((self.proc.freq[:,np.newaxis], self.proc.modespec_tw_n[idx]), axis=1)
                        np.savetxt(self.args.out+'modespec_b_par{}.txt'.format(signum), outarr_b)
                        np.savetxt(self.args.out+'modespec_p_par{}.txt'.format(signum), outarr_p)
                        np.savetxt(self.args.out+'modespec_n_par{}.txt'.format(signum), outarr_n)

            if self.args.bpf:
                print('\tgenerating bpf trajectory...')
                BPF_TRAJ = bpfilter(self.proc.loader.TRAJ[tw[0]:tw[1]],
                        self.args.dt, *self.args.bpf_range, scale=self.args.fscale)
                par = int(round(self.args.bpf_part*BPF_TRAJ.n_frames))
                BPF_TRAJ.center_coordinates(mass_weighted=True)
                BPF_TRAJ = BPF_TRAJ[par:-par]
                for ii in range(1, BPF_TRAJ.n_frames):
                    BPF_TRAJ.xyz[ii,:,:] = rotate(BPF_TRAJ.xyz[ii,:,:], BPF_TRAJ.xyz[0,:,:])
                BPF_TRAJ[0].save_pdb(self.args.out+'traj.pdb')
                BPF_TRAJ.save_dcd(self.args.out+'traj.dcd')
                for pii in range(int(np.ceil(BPF_TRAJ.n_frames/self.args.bpf_length))):
                    print('\t\t', pii*self.args.bpf_length+par, (pii+1)*self.args.bpf_length+par)
                    BPF_TRAJ[pii*self.args.bpf_length:(pii+1)*self.args.bpf_length].save_pdb(
                            self.args.out+'traj({}-{})_({}_{})_par{}.pdb'.format(
                                int(round(self.args.bpf_range[0])),
                                int(round(self.args.bpf_range[1])),
                                pii*self.args.bpf_length+par,
                                (pii+1)*self.args.bpf_length+par, signum))

    def plot_tw(self):
        if not self.args.bpf:

            if self.args.spec_plot:
                print('\tplotting spectrum...')
                for idx, tw in enumerate(self.proc.TWSlices):
                    print('\t\tplot spec. - time window', idx, '/', self.args.n_time_windows, ':', *tw)
                    signum = str(idx).zfill(len(str(len(self.proc.TWSlices))))
                    plt.plot(self.proc.freq, self.proc.allspec_tw[idx].sum(1))
                    plt.savefig(self.args.out+'fullspec_par{}.png'.format(signum))
                    plt.cla()

            if self.args.atom_spec_plot:
                print('\tplotting spectra...')
                for idx, tw in enumerate(self.proc.TWSlices):
                    print('\t\tplot spec. - time window', idx, '/', self.args.n_time_windows, ':', *tw)
                    signum = str(idx).zfill(len(str(len(self.proc.TWSlices))))
                    for atm_idx in range(self.proc.allspec_tw[idx].shape[1]):
                        plt.plot(self.proc.freq, self.proc.allspec_tw[idx][:,atm_idx],
                                    label=self.proc.loader.name_data[atm_idx])
                        plt.legend()
                        plt.savefig(self.args.out+'atomspec_{}_par{}.png'.format(self.proc.loader.name_data[atm_idx], signum))
                        plt.cla()

            if self.args.mode_spec_plot:
                print('\tplotting modes...')
                for idx, tw in enumerate(self.proc.TWSlices):
                    print('\t\tplot modes time window', idx, '/', self.args.n_time_windows, ':', *tw)
                    signum = str(idx).zfill(len(str(len(self.proc.TWSlices))))
                    for bnd_idx in range(len(self.proc.Bonds)):

                        plt.plot(self.proc.freq, self.proc.modespec_tw_b[idx][:,bnd_idx],
                                            label='b_'+self.proc.Bond_names[bnd_idx])
                        plt.legend()
                        plt.savefig(self.args.out+'modespec_b_{}_par{}.png'.format(self.proc.Bond_names[bnd_idx], signum))
                        plt.cla()

                        plt.plot(self.proc.freq, self.proc.modespec_tw_b[idx][:,bnd_idx],
                                            label='p_'+self.proc.Bond_names[bnd_idx])
                        plt.legend()
                        plt.savefig(self.args.out+'modespec_p_{}_par{}.png'.format(self.proc.Bond_names[bnd_idx], signum))
                        plt.cla()

                        plt.plot(self.proc.freq, self.proc.modespec_tw_b[idx][:,bnd_idx],
                                            label='n_'+self.proc.Bond_names[bnd_idx])
                        plt.legend()
                        plt.savefig(self.args.out+'modespec_n_{}_par{}.png'.format(self.proc.Bond_names[bnd_idx], signum))
                        plt.cla()


    def plot_3d(self):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from collections import defaultdict
        if self.args.spec_plot_3d:
            print('\tplotting 3d spectrum...')
            fig = plt.figure(figsize=(32,16))
            ax = fig.gca(projection='3d')
            X = self.proc.freq
            Y = np.array([(ii[0]+ii[1])/2.0 for ii in self.proc.TWSlices])
            X, Y = np.meshgrid(X, Y)
            Z = np.empty(X.shape)
            for tii in range(Z.shape[0]):
                Z[tii,:] = np.nan_to_num(self.proc.allspec_tw[tii].sum(1))
            #Z[:,:] = Z[:,:]/(Z[:,:].max(axis=1))[:,np.newaxis]
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap=cm.coolwarm)
            ax.view_init(elev=35, azim=-70)
            #ax.set_xlim(2000,2500)
            #ax.set_zlim(0,1)
            plt.savefig(self.args.out+'fullspec_3d.png')
            plt.cla()

        if self.args.atom_spec_plot_3d:
            print('\tplotting 3d spectra...')
            X = self.proc.freq
            Y = np.array([(ii[0]+ii[1])/2.0 for ii in self.proc.TWSlices])
            X, Y = np.meshgrid(X, Y)
            
            for atm_idx in range(self.proc.allspec_tw[0].shape[1]):
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                Z = np.empty(X.shape)
                
                for tii in range(Z.shape[0]):
                    Z[tii,:] = np.nan_to_num(self.proc.allspec_tw[tii][:,atm_idx])
                
                surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            cmap=cm.coolwarm)
                
                ax.view_init(elev=35, azim=-70)
                
                plt.savefig(self.args.out+'atomspec3d_{}.png'.format(self.proc.loader.name_data[atm_idx]))
                plt.cla()

        if self.args.mode_spec_plot_3d:

            print('\tplotting 3d modes...')
            for bnd_idx in range(len(self.proc.Bonds)):
                
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                X = self.proc.freq
                Y = np.array([(ii[0]+ii[1])/2.0 for ii in self.proc.TWSlices])
                X, Y = np.meshgrid(X, Y)
                Z = np.empty(X.shape)
                for tii in range(Z.shape[0]):
                    Z[tii,:] = np.nan_to_num(self.proc.modespec_tw_b[tii][:,bnd_idx])
                surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            cmap=cm.coolwarm)

                ax.view_init(elev=35, azim=-70)
                plt.savefig(self.args.out+'modespec3d_b_{}.png'.format(self.proc.Bond_names[bnd_idx]))
                plt.cla()

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                X = self.proc.freq
                Y = np.array([(ii[0]+ii[1])/2.0 for ii in self.proc.TWSlices])
                X, Y = np.meshgrid(X, Y)
                Z = np.empty(X.shape)
                for tii in range(Z.shape[0]):
                    Z[tii,:] = np.nan_to_num(self.proc.modespec_tw_p[tii][:,bnd_idx])
                surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            cmap=cm.coolwarm)
                ax.view_init(elev=35, azim=-70)
                plt.savefig(self.args.out+'modespec3d_p_{}.png'.format(self.proc.Bond_names[bnd_idx]))
                plt.cla()

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                X = self.proc.freq
                Y = np.array([(ii[0]+ii[1])/2.0 for ii in self.proc.TWSlices])
                X, Y = np.meshgrid(X, Y)
                Z = np.empty(X.shape)
                for tii in range(Z.shape[0]):
                    Z[tii,:] = np.nan_to_num(self.proc.modespec_tw_n[tii][:,bnd_idx])
                surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            cmap=cm.coolwarm)

                ax.view_init(elev=35, azim=-70)
                plt.savefig(self.args.out+'modespec3d_n_{}.png'.format(self.proc.Bond_names[bnd_idx]))
                plt.cla()


