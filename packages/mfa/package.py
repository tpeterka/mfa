# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Mfa(CMakePackage):
    """Multivariate functional approximation library"""

    homepage = "https://github.com/tpeterka/mfa"
    url      = "https://github.com/tpeterka/mfa"
    git      = "https://github.com/tpeterka/mfa.git"

    version('master', branch='master')

    depends_on('mpi')
    depends_on('diy@master')
    depends_on('eigen')

    def cmake_args(self):
        args = ['-DCMAKE_CXX_COMPILER=%s' % self.spec['mpi'].mpicxx,
                '-DDIY_INCLUDE_DIRS=%s' %  self.spec['diy'].prefix,
                '-DEIGEN_INCLUDE_DIRS=%s' % self.spec['eigen'].prefix]
        return args
