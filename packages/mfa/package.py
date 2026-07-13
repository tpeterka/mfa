# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *

class Mfa(CMakePackage):
    """Multivariate functional approximation library"""

    homepage = "https://github.com/tpeterka/mfa"
    url      = "https://github.com/tpeterka/mfa"
    git      = "file:///home/tpeterka/software/mfa"

    version('master', branch='master')
    version("local", git="file:///home/tpeterka/software/mfa", branch="master")

    variant('build_type', default='Release', description='CMake build type', values=('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel'))
    variant('thread', values=str, default='serial', description='Threading model: serial, tbb, sycl, kokkos (default = serial).')
    variant('examples', default=True, description='Build MFA examples')
    variant('tests', default=True, description='Build MFA unit tests')
    variant('python', default=False, description='Python bindings')

    depends_on('mpich')
    depends_on('tbb', when='thread=tbb')
    depends_on('kokkos', when='thread=kokkos')
    depends_on('python+shared', type=('build', 'run'), when='+python')
    depends_on('py-numpy', type=('build', 'run'), when='+python')
    depends_on('py-mpi4py', type=('build', 'run'), when='+python')

    def cmake_args(self):
        args = ['-DCMAKE_BUILD_TYPE=%s' % self.spec.variants['build_type'].value,
                '-DCMAKE_C_COMPILER=%s' % self.spec['mpich'].mpicc,
                '-DCMAKE_CXX_COMPILER=%s' % self.spec['mpich'].mpicxx,
                self.define_from_variant('mfa_build_examples', 'examples'),
                self.define_from_variant('mfa_build_tests', 'tests'),
                self.define('mfa_thread', self.spec.variants['thread'].value),
                self.define_from_variant('mfa_python', 'python'),
                self.define_from_variant('python', 'python')]

        print('MFA build type: ', self.spec.variants['build_type'].value)
        print('Building MFA examples: ', self.spec.variants['examples'].value)
        print('Building MFA tests: ', self.spec.variants['tests'].value)
        print('Building MFA with threading type: ', self.spec.variants['thread'].value)

        return args

    def install(self, spec, prefix):
        super().install(spec, prefix)

        if '+python' in spec:
            pyver = spec['python'].version.up_to(2)
            site_packages = join_path(prefix.lib, 'python{0}'.format(pyver), 'site-packages')
            mkdirp(site_packages)

            install_tree(
                join_path(self.build_directory, 'python', 'mfa'),
                join_path(site_packages, 'mfa')
            )

            install_tree(
                join_path(self.build_directory, 'include', 'diy', 'lib', 'diy'),
                join_path(site_packages, 'diy')
            )
