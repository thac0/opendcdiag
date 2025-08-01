# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0

project(
    'OpenDCDiag',
    ['c', 'cpp'],
    default_options: [
        'b_ndebug=if-release',
        'c_std=gnu17',
        'cpp_std=gnu++23',
    ],
    meson_version : '>=1.3',
)

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

bash = find_program('bash')
perl = find_program('perl')
python = find_program('python3')
shell = find_program('sh')

if cpp.get_id() != 'gcc' or cc.get_id() != 'gcc'
    warning(
        'We recommend building OpenDCDiag with the GNU Compiler Collection (gcc). Your build may fail or your executable may not behave as expected.',
    )
endif

if cpp.get_linker_id() != 'ld.bfd' or cc.get_linker_id() != 'ld.bfd'
    warning(
        'We recommend linking OpenDCDiag with the binutils linker (ld). Your build may fail or your executable may not behave as expected.',
    )
endif

device_type = get_option('device_type')

march_generic_flags = []
march_flags = []
march_base = get_option('march_base')
if march_base != ''
    march_flags += '-march=' + march_base
endif
if host_machine.cpu_family() == 'x86_64'
    if march_base == ''
        march_base = 'haswell'
    endif
    march_generic_flags = [
        '-march=core2',
        '-mno-sse3',
    ]
    march_flags = [
        '-march=' + march_base,
        '-mtune=skylake',
        '-maes',
        '-mpclmul',
        '-mno-rdrnd',
        '-falign-loops=32',
    ]
    if cc.has_argument('-Wa,-muse-unaligned-vector-move')
        march_flags += ['-Wa,-muse-unaligned-vector-move']
    elif target_machine.system() == 'windows' and cpp.get_id() == 'gcc'
        warning(
            'GCC for Windows is unable to generate 32-byte-aligned stacks necessary for AVX content. '
            + 'Please upgrade to a version of binutils that supports the option -muse-unaligned-vector-move (2.38 or later). '
            + 'The application will randomly crash at runtime otherwise.',
        )
    endif
endif

debug_nasm_flags = ''
if get_option('buildtype') == 'debug'
    debug_c_flags = [
        '-ggdb3',
        '-O0',
    ]
elif get_option('buildtype') == 'debugoptimized'
    debug_c_flags = [
        '-ggdb3',
        '-O3',
        '-D_FORTIFY_SOURCE=2',
    ]
else
    debug_c_flags = [
        '-O3',
        '-g',
        '-D_FORTIFY_SOURCE=2',
    ]
endif

if get_option('static_libstdcpp') == true or target_machine.system() == 'windows'
    add_project_link_arguments(
        [
            '-static-libstdc++',
        ],
        language: 'cpp',
    )
endif

nasm_system_flags = [
    '-felf64',
    '-Xgnu',
    '-DLINUX',
    '-D__LINUX__',
]
default_c_flags = []
default_cpp_flags = []

foreach ccflag : [
    '-fstack-clash-protection',
    '-fstack-protector-strong',
    '-fcf-protection=full',
    '-mcet-switch',
]
    if cc.has_argument(ccflag)
        add_project_arguments(ccflag, language: ['c', 'cpp'])
    endif
endforeach

foreach lflag : [
    '-Wl,-z,relro',
    '-Wl,-z,noexecstack',
    '-pie',
]
    if cc.links('int main() { return 0; }', args: [lflag], name: lflag)
        add_project_link_arguments(lflag, language: ['c', 'cpp'])
    endif
endforeach

if target_machine.system() == 'windows'
    add_project_arguments(
        [
            '-D__STDC_FORMAT_MACROS',
            '-D__USE_MINGW_ANSI_STDIO',
            '-D_POSIX',
            '-D_WIN32_WINNT=0x0A00',
            '-D_NO_SYS_GUID_OPERATOR_EQ_',
            '-Wa,-mbig-obj',
            '-include', meson.current_source_dir()
            + '/framework/sysdeps/windows/win32_stdlib.h',
        ],
        language: ['c', 'cpp'],
    )
    add_project_link_arguments(
        [
            '-mconsole',
            '-Wl,--stack,8388608',
            '-static-libgcc',
            '-Wl,-static',
            '-lpthread',
            '-lm',
            '-lssp',
            '-Wl,-Bdynamic',
            '-lshlwapi',
            '-lntdll',
        ],
        language: ['c', 'cpp'],
    )
    nasm_system_flags = [
        '-fwin64',
        '-Xgnu',
        '-DWIN_ABI',
    ]
elif target_machine.system() == 'darwin'
    add_project_arguments(
        [
            '-D__unix__', # Why doesn't Apple think their OS is Unix?
            '-D_DARWIN_C_SOURCE',
        ],
        language: ['c', 'cpp'],
    )
    nasm_system_flags = ['-fmacho64', '-Xgnu', '-DAPPLE', '-D__APPLE__', '--prefix=_']
else
    if get_option('buildtype') == 'debug'
        debug_nasm_flags = '-gdwarf'
    endif
    if get_option('buildtype') == 'debug'
        debug_nasm_flags = '-gdwarf'
    endif
endif

default_c_flags += [
    '-DSANDSTONE',
    '-fno-associative-math',
    '-fno-asynchronous-unwind-tables',
]

default_cpp_flags += [
    '-DSANDSTONE',
    '-fno-associative-math',
    '-fvisibility-inlines-hidden',
]

if get_option('b_sanitize') != 'none'
    default_c_flags += [
        '-g',
        '-fno-omit-frame-pointer',
    ]
    default_cpp_flags += [
        '-g',
        '-fno-omit-frame-pointer',
    ]
endif

default_c_warn = [
    '-Wall',
    '-Wextra',
    '-Werror=format-security',
    '-Werror=incompatible-pointer-types',
    '-Werror=implicit-function-declaration',
    '-Werror=int-conversion',
    '-Wno-unused-parameter',
    '-Wno-sign-compare',
]
if cc.has_argument('-Werror=discarded-qualifiers')
    default_c_warn += '-Werror=discarded-qualifiers'
endif

default_cpp_warn = [
    '-Wall',
    '-Wextra',
    '-Wformat',
    '-Wformat-security',
    '-Wno-unused-parameter',
    '-Wno-sign-compare',
    '-Wno-missing-field-initializers',
    '-Wno-non-virtual-dtor',
]

# Subdirs can add target dependencies if they wish
target_deps = []

boost_dep = dependency('boost', version : '>=1.69', required : false)
# Check for required headers
# (Also accomodates distributions which do not package a pkg-config file for boost)
boost_has_hdrs = cpp.check_header('boost/algorithm/string.hpp') and cpp.check_header('boost/type_traits/is_complex.hpp')
has_boost = boost_dep.found() or boost_has_hdrs

if not has_boost
    assert(false, 'Could not find boost or its headers')
elif not boost_dep.found()
    boost_dep = []
endif

pthread_dep = dependency('threads')

dl_dep = []
gtest_dep = []
if target_machine.system() != 'windows'
    dl_dep = cc.find_library('dl')
    gtest_dep = dependency('gtest_main', required: false)
endif

if get_option('coverage') == true
    gcov_dep = cc.find_library('gcov')
    target_deps += [gcov_dep]
    default_c_flags += [
        '-fprofile-arcs',
        '-ftest-coverage',
    ]
    default_cpp_flags += [
        '-fprofile-arcs',
        '-ftest-coverage',
    ]
endif

target_deps += [
    boost_dep,
    dl_dep,
    pthread_dep,
]

top_incdir = include_directories('.')

# Subdirs add (static) libraries to these arrays for linking
sandstone_tests = []
sandstone_tests_link = []
sandstone_tests_link_whole = []

unittests_sources = []

subdir('framework')
subdir('tests')

sandstone_tests_link_whole += [sandstone_tests]

executable(
    get_option('executable_name'),
    install: true,
    dependencies: target_deps,
    link_with: [
        sandstone_tests_link,
    ],
    link_whole: [
        framework_a,
        sandstone_tests_link_whole,
    ],
    include_directories: [
        framework_incdir,
    ],
    cpp_args: [
        default_cpp_warn,
        default_cpp_flags,
    ],
)

executable(
    'unittests',
    files(
        'framework/sandstone_unittests_utils.cpp',
        'framework/sysdeps/linux/malloc.cpp',
    ),
    unittests_sources,
    dependencies: [gtest_dep, pthread_dep],
    include_directories: [
        framework_incdir,
    ],
    cpp_args: [
        '-march=haswell',
        debug_c_flags,
        default_cpp_warn,
        '-USANDSTONE',
    ],
    build_by_default: false,
    native: true,
)

summary(
    {
        'Logging format': get_option('logging_format'),
        'Using -march': march_base,
        'Static libstdc++': get_option('static_libstdcpp')
        or target_machine.system() == 'windows',
        'Executable name': get_option('executable_name'),
        'Fallback executable name': get_option('fallback_exec'),
        'Device type:': device_type,
    },
    section: 'Configuration',
    bool_yn: true,
)
