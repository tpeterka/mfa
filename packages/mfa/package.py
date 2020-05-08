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

    variant('thread', values=str, default='serial', description='Threading model: serial, tbb, sycl, kokkos (default = serial).')

    depends_on('mpi')
    depends_on('diy@master')
    depends_on('eigen')
    depends_on('tbb', when='thread=tbb')
    depends_on('kokkos', when='thread=kokkos')

    def cmake_args(self):
        args = ['-DDIY_INCLUDE_DIRS=%s/include' %  self.spec['diy'].prefix,
                '-DEIGEN_INCLUDE_DIRS=%s/include/eigen3' % self.spec['eigen'].prefix]

        thread = str(self.spec.variants['thread'].value)
        print('Building with thread =', thread)

        args.extend(['-Dmfa_thread=%s' % thread])

        return args
