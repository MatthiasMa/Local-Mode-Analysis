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

class LMALoader:

    def __init__(self, data_args):

        self.args = data_args
        self.data_file = data_args.input
        self.top_file = data_args.top
        self.weighting_file = data_args.weighting
        self.traj_data = None
        self.name_data = None
        self.bond_data = None
        self.bond_traj_data = None
        self.bond_name_data = None
        self.weighting_data = None
        self.top_data = None
        self.TRAJ = None
        self.resave = False
        self.load()

    def load(self):

        try:
            self.try_mdtraj()
        except OSError:
            fak = 10.0 if self.args.pos_data_scale == 'nm' else 1.0
            self.traj_data = self.np_load(self.data_file)*fak
            if self.args.time_range != [0.0, 0.0]:
                if self.traj_data.ndim == 3:
                    self.traj_data = self.traj_data[
                            self.args.time_range[0]:self.args.time_range[1],:,:]
                elif self.traj_data.ndim == 2:
                    self.traj_data = self.traj_data[
                            self.args.time_range[0]:self.args.time_range[1],:]
                elif self.traj_data.ndim == 1:
                    self.traj_data = self.traj_data[
                            self.args.time_range[0]:self.args.time_range[1]]
        
        if self.weighting_file is not None:
            if 'dcd' in self.weighting_file.split('.')[-1]:
                #weighting_data_temp = mdtraj.load(self.weighting_file, top=self.top_file).xyz[:,:,:-1]
                #self.weighting_data = np.empty((weighting_data_temp.shape[0], weighting_data_temp.shape[1]))
                self.weighting_data = mdtraj.load_dcd(self.weighting_file, top=self.top_file).xyz[:,:,1]
            else:
                self.weighting_data = self.np_load(self.weighting_file)
            
            #self.weighting_data = self.np_load(self.weighting_file)
            
            if self.weighting_data.ndim == 1:
                self.weighting_data = self.weighting_data[np.newaxis,:,np.newaxis]
            elif self.weighting_data.ndim == 2:
                if self.args.time_range != [0.0, 0.0]:
                    self.weighting_data = self.weighting_data[
                            self.args.time_range[0]:self.args.time_range[1],:]
                self.weighting_data = self.weighting_data[:self.traj_data.shape[0],:,np.newaxis]
            self.traj_data *= self.weighting_data
        self.name_data = get_full_atom_names(self.TRAJ, self.traj_data.shape[0])

    def try_mdtraj(self):

        if self.top_file is not None and self.top_file[-3:] == 'psf':
            self.convert_psf2pdb()
        try:
            self.TRAJ = mdtraj.load(self.data_file, top=self.top_file)
        except (OSError, TypeError):
            try:
                self.TRAJ = mdtraj.load(self.data_file)
            except OSError:
                if self.top_file is None: raise OSError
                self.TRAJ = mdtraj.load(self.top_file)
                self.traj_data = self.np_load(self.data_file)
                if self.traj_data.ndim == 2:
                    self.traj_data = self.traj_data.reshape(self.traj_data.shape[0],-1,3)
                fak = 1.0 if self.args.pos_data_scale == 'nm' else 10.0
                if self.args.time_range != [0.0, 0.0]:
                    self.traj_data = self.traj_data[
                            self.args.time_range[0]:self.args.time_range[1],:,:]
                self.TRAJ.xyz = self.traj_data/fak
                self.TRAJ.time = np.array(list(range(self.traj_data.shape[0])))
        self.TRAJ.center_coordinates(mass_weighted=True)
        self.traj_data = self.TRAJ.xyz*10.0
        if self.resave:
            self.TRAJ[0].save_pdb(self.top_file)


    def np_load(self, infile):
        try:
            return np.load(infile)
        except OSError:
            return np.loadtxt(infile)

    def convert_psf2pdb(self):
        with open(self.top_file, 'r') as inStream:
            with open(self.top_file+'.pdb', 'w') as outStream:
                for line in inStream:
                    if '!NATOM' in line:
                        natoms = int(line.split()[0])
                        for idx, line in enumerate(inStream):
                            if natoms == idx: break
                            #print(line.strip())
                            splitline = line.replace("''","'").split()
                            outStream.write(
                                'ATOM  {:>5d}  {:<3s} {:<3s}{:>6d}       0.000   0.000   0.000  0.00  0.00         {:>2s} \n'.format(
                                idx+1, splitline[4], splitline[1][:3], int(splitline[2]), splitline[4][0]))
                        self.resave = True
                        self.top_file = self.top_file+'.pdb'


def prep_dir(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

def get_full_atom_names(Traj=None, n_atms=0):
    names = []
    if Traj is not None:
        for atm in range(Traj.n_atoms):
            atom = Traj.top.atom(atm)
            #names.append( [ atom.name, atom.serial, atom.residue.name,
            #                atom.residue.resSeq, atom.residue.chain.index ] )
            names.append('-'.join([str(ss) for ss in [atm, atom.name, atom.residue.name, atom.residue.resSeq]]))
    else:
        for ii in range(n_atms):
            names.append(str(ii))

    return names
