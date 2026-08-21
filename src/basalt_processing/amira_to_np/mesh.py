#!/usr/bin/python3
# SPDX-License-Identifier: EUPL-1.2
# Incorporated from Biomedisa into basalt_processing on 2026-07-22.
##########################################################################
##                                                                      ##
##  Copyright (c) 2019-2025 Philipp Lösel. All rights reserved.         ##
##                                                                      ##
##  This file is part of the open source project biomedisa.             ##
##                                                                      ##
##  Licensed under the European Union Public Licence (EUPL)             ##
##  v1.2, or - as soon as they will be approved by the                  ##
##  European Commission - subsequent versions of the EUPL;              ##
##                                                                      ##
##  You may redistribute it and/or modify it under the terms            ##
##  of the EUPL v1.2. You may not use this work except in               ##
##  compliance with this Licence.                                       ##
##                                                                      ##
##  You can obtain a copy of the Licence at:                            ##
##                                                                      ##
##  https://joinup.ec.europa.eu/page/eupl-text-11-12                    ##
##                                                                      ##
##  Unless required by applicable law or agreed to in                   ##
##  writing, software distributed under the Licence is                  ##
##  distributed on an "AS IS" basis, WITHOUT WARRANTIES                 ##
##  OR CONDITIONS OF ANY KIND, either express or implied.               ##
##                                                                      ##
##  See the Licence for the specific language governing                 ##
##  permissions and limitations under the Licence.                      ##
##                                                                      ##
##########################################################################

import os
import numpy as np
import re

def get_voxel_spacing(header, extension):
    if extension == '.am':
        # read header as string
        b = header[0].tobytes()
        try:
            s = b.decode("utf-8")
        except:
            s = b.decode("latin1")
        # get physical size from image header
        lattice = re.search('define Lattice (.*)\n', s)
        bounding_box = re.search('BoundingBox (.*),\n', s)
        if bounding_box and lattice:
            # get number of voxels
            lattice = lattice.group(1)
            xsh, ysh, zsh = lattice.split(' ')
            xsh, ysh, zsh = float(xsh), float(ysh), float(zsh)
            # get bounding box
            bounding_box = bounding_box.group(1)
            i0, i1, i2, i3, i4, i5 = bounding_box.split(' ')
            # calculate voxel spacing
            xres = round((float(i1)-float(i0)) / (xsh - 1), 7) # bugfix to make it correct
            yres = round((float(i3)-float(i2)) / (ysh - 1), 7) # buffix to make it correct
            zres = round((float(i5)-float(i4)) / (zsh - 1), 7) # buffix to make it correct

        else:
            xres, yres, zres = 1, 1, 1
    elif extension in ['.hdr', '.mhd', '.mha', '.nrrd', '.nii', '.nii.gz']:
        xres, yres, zres = header.GetSpacing()
    elif extension == '.zip':
        header = header[0][0]
        try:
            xres, yres, zres = header.get_voxel_spacing()
        except:
            xres, yres = header.get_voxel_spacing()
            zres = 1.0
    else:
        print('Warning: could not get voxel spacing. Using x_spacing, y_spacing, z_spacing = 1, 1, 1 instead.')
        xres, yres, zres = 1, 1, 1
    return xres, yres, zres
