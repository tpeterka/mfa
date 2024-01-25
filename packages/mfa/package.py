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

    variant('build_type', default='Release', description='CMake build type', values=('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel'))
    variant('thread', values=str, default='serial', description='Threading model: serial, tbb, sycl, kokkos (default = serial).')
    variant('examples', default=True, description='Build MFA examples')
    variant('tests', default=True, description='Build MFA unit tests')

    depends_on('mpich')
    depends_on('tbb', when='thread=tbb')
    depends_on('kokkos', when='thread=kokkos')

    def cmake_args(self):
        args = ['-DCMAKE_BUILD_TYPE=%s' % self.spec.variants['build_type'].value,
                '-DCMAKE_C_COMPILER=%s' % self.spec['mpich'].mpicc,
                '-DCMAKE_CXX_COMPILER=%s' % self.spec['mpich'].mpicxx,
                self.define_from_variant('examples', 'mfa_build_examples'),
                self.define_from_variant('tests', 'mfa_build_tests'),
                '-Dmfa_thread=%s' % self.spec.variants['thread']]

        # thread = str(self.spec.variants['thread'].value)
        print('MFA build type: ', self.spec.variants['build_type'].value)
        print('Building MFA examples: ', self.spec.variants['examples'].value)
        print('Building MFA tests: ', self.spec.variants['tests'].value)
        print('Building MFA with threading type: ', self.spec.variants['thread'].value)
        # args.extend(['-Dmfa_thread=%s' % thread])

        return args
