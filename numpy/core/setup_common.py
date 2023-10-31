import copy
import pathlib
import sys
import textwrap
from numpy.distutils.misc_util import mingw32
C_ABI_VERSION = 16777225
C_API_VERSION = 16


class MismatchCAPIError(ValueError):
    pass


def get_api_versions(apiversion, codegen_dir):
    """
    Return current C API checksum and the recorded checksum.

    Return current C API checksum and the recorded checksum for the given
    version of the C API version.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup_common.get_api_versions', 'get_api_versions(apiversion, codegen_dir)', {'sys': sys, 'apiversion': apiversion, 'codegen_dir': codegen_dir}, 2)

def check_api_version(apiversion, codegen_dir):
    """Emits a MismatchCAPIWarning if the C API version needs updating."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.core.setup_common.check_api_version', 'check_api_version(apiversion, codegen_dir)', {'get_api_versions': get_api_versions, '__file__': __file__, 'MismatchCAPIError': MismatchCAPIError, 'apiversion': apiversion, 'codegen_dir': codegen_dir}, 0)
FUNC_CALL_ARGS = {}

def set_sig(sig):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.core.setup_common.set_sig', 'set_sig(sig)', {'FUNC_CALL_ARGS': FUNC_CALL_ARGS, 'sig': sig}, 0)
for file in ['feature_detection_locale.h', 'feature_detection_math.h', 'feature_detection_cmath.h', 'feature_detection_misc.h', 'feature_detection_stdio.h']:
    with open(pathlib.Path(__file__).parent / file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            if not line.strip():
                continue
            set_sig(line)
MANDATORY_FUNCS = ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'fabs', 'floor', 'ceil', 'sqrt', 'log10', 'log', 'exp', 'asin', 'acos', 'atan', 'fmod', 'modf', 'frexp', 'ldexp', 'expm1', 'log1p', 'acosh', 'asinh', 'atanh', 'rint', 'trunc', 'exp2', 'copysign', 'nextafter', 'strtoll', 'strtoull', 'cbrt', 'log2', 'pow', 'hypot', 'atan2', 'creal', 'cimag', 'conj']
OPTIONAL_LOCALE_FUNCS = ['strtold_l']
OPTIONAL_FILE_FUNCS = ['ftello', 'fseeko', 'fallocate']
OPTIONAL_MISC_FUNCS = ['backtrace', 'madvise']
OPTIONAL_VARIABLE_ATTRIBUTES = ['__thread', '__declspec(thread)']
OPTIONAL_FUNCS_MAYBE = ['ftello', 'fseeko']
C99_COMPLEX_TYPES = ['complex double', 'complex float', 'complex long double']
C99_COMPLEX_FUNCS = ['cabs', 'cacos', 'cacosh', 'carg', 'casin', 'casinh', 'catan', 'catanh', 'cexp', 'clog', 'cpow', 'csqrt', 'csin', 'csinh', 'ccos', 'ccosh', 'ctan', 'ctanh']
OPTIONAL_HEADERS = ['xmmintrin.h', 'emmintrin.h', 'immintrin.h', 'features.h', 'xlocale.h', 'dlfcn.h', 'execinfo.h', 'libunwind.h', 'sys/mman.h']
OPTIONAL_INTRINSICS = [('__builtin_isnan', '5.'), ('__builtin_isinf', '5.'), ('__builtin_isfinite', '5.'), ('__builtin_bswap32', '5u'), ('__builtin_bswap64', '5u'), ('__builtin_expect', '5, 0'), ('__builtin_mul_overflow', '(long long)5, 5, (int*)5'), ('_m_from_int64', '0', 'emmintrin.h'), ('_mm_load_ps', '(float*)0', 'xmmintrin.h'), ('_mm_prefetch', '(float*)0, _MM_HINT_NTA', 'xmmintrin.h'), ('_mm_load_pd', '(double*)0', 'emmintrin.h'), ('__builtin_prefetch', '(float*)0, 0, 3'), ('__asm__ volatile', '"vpand %xmm1, %xmm2, %xmm3"', 'stdio.h', 'LINK_AVX'), ('__asm__ volatile', '"vpand %ymm1, %ymm2, %ymm3"', 'stdio.h', 'LINK_AVX2'), ('__asm__ volatile', '"vpaddd %zmm1, %zmm2, %zmm3"', 'stdio.h', 'LINK_AVX512F'), ('__asm__ volatile', '"vfpclasspd $0x40, %zmm15, %k6\\n"                                             "vmovdqu8 %xmm0, %xmm1\\n"                                             "vpbroadcastmb2q %k0, %xmm0\\n"', 'stdio.h', 'LINK_AVX512_SKX'), ('__asm__ volatile', '"xgetbv"', 'stdio.h', 'XGETBV')]
OPTIONAL_FUNCTION_ATTRIBUTES = [('__attribute__((optimize("unroll-loops")))', 'attribute_optimize_unroll_loops'), ('__attribute__((optimize("O3")))', 'attribute_optimize_opt_3'), ('__attribute__((optimize("O2")))', 'attribute_optimize_opt_2'), ('__attribute__((nonnull (1)))', 'attribute_nonnull')]
OPTIONAL_FUNCTION_ATTRIBUTES_AVX = [('__attribute__((target ("avx")))', 'attribute_target_avx'), ('__attribute__((target ("avx2")))', 'attribute_target_avx2'), ('__attribute__((target ("avx512f")))', 'attribute_target_avx512f'), ('__attribute__((target ("avx512f,avx512dq,avx512bw,avx512vl,avx512cd")))', 'attribute_target_avx512_skx')]
OPTIONAL_FUNCTION_ATTRIBUTES_WITH_INTRINSICS_AVX = [('__attribute__((target("avx2,fma")))', 'attribute_target_avx2_with_intrinsics', '__m256 temp = _mm256_set1_ps(1.0); temp =     _mm256_fmadd_ps(temp, temp, temp)', 'immintrin.h'), ('__attribute__((target("avx512f")))', 'attribute_target_avx512f_with_intrinsics', '__m512i temp = _mm512_castps_si512(_mm512_set1_ps(1.0))', 'immintrin.h'), ('__attribute__((target ("avx512f,avx512dq,avx512bw,avx512vl,avx512cd")))', 'attribute_target_avx512_skx_with_intrinsics', '__mmask8 temp = _mm512_fpclass_pd_mask(_mm512_set1_pd(1.0), 0x01);    __m512i unused_temp =         _mm512_castps_si512(_mm512_set1_ps(1.0));    _mm_mask_storeu_epi8(NULL, 0xFF, _mm_broadcastmb_epi64(temp))', 'immintrin.h')]

def fname2def(name):
    return 'HAVE_%s' % name.upper()

def sym2def(symbol):
    define = symbol.replace(' ', '')
    return define.upper()

def type2def(symbol):
    define = symbol.replace(' ', '_')
    return define.upper()

def check_long_double_representation(cmd):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup_common.check_long_double_representation', 'check_long_double_representation(cmd)', {'LONG_DOUBLE_REPRESENTATION_SRC': LONG_DOUBLE_REPRESENTATION_SRC, 'sys': sys, 'mingw32': mingw32, 'long_double_representation': long_double_representation, 'pyod': pyod, 'cmd': cmd}, 1)
LONG_DOUBLE_REPRESENTATION_SRC = '\n/* "before" is 16 bytes to ensure there\'s no padding between it and "x".\n *    We\'re not expecting any "long double" bigger than 16 bytes or with\n *       alignment requirements stricter than 16 bytes.  */\ntypedef %(type)s test_type;\n\nstruct {\n        char         before[16];\n        test_type    x;\n        char         after[8];\n} foo = {\n        { \'\\0\', \'\\0\', \'\\0\', \'\\0\', \'\\0\', \'\\0\', \'\\0\', \'\\0\',\n          \'\\001\', \'\\043\', \'\\105\', \'\\147\', \'\\211\', \'\\253\', \'\\315\', \'\\357\' },\n        -123456789.0,\n        { \'\\376\', \'\\334\', \'\\272\', \'\\230\', \'\\166\', \'\\124\', \'\\062\', \'\\020\' }\n};\n'

def pyod(filename):
    """Python implementation of the od UNIX utility (od -b, more exactly).

    Parameters
    ----------
    filename : str
        name of the file to get the dump from.

    Returns
    -------
    out : seq
        list of lines of od output

    Notes
    -----
    We only implement enough to get the necessary information for long double
    representation, this is not intended as a compatible replacement for od.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup_common.pyod', 'pyod(filename)', {'filename': filename}, 1)
_BEFORE_SEQ = ['000', '000', '000', '000', '000', '000', '000', '000', '001', '043', '105', '147', '211', '253', '315', '357']
_AFTER_SEQ = ['376', '334', '272', '230', '166', '124', '062', '020']
_IEEE_DOUBLE_BE = ['301', '235', '157', '064', '124', '000', '000', '000']
_IEEE_DOUBLE_LE = _IEEE_DOUBLE_BE[::-1]
_INTEL_EXTENDED_12B = ['000', '000', '000', '000', '240', '242', '171', '353', '031', '300', '000', '000']
_INTEL_EXTENDED_16B = ['000', '000', '000', '000', '240', '242', '171', '353', '031', '300', '000', '000', '000', '000', '000', '000']
_MOTOROLA_EXTENDED_12B = ['300', '031', '000', '000', '353', '171', '242', '240', '000', '000', '000', '000']
_IEEE_QUAD_PREC_BE = ['300', '031', '326', '363', '105', '100', '000', '000', '000', '000', '000', '000', '000', '000', '000', '000']
_IEEE_QUAD_PREC_LE = _IEEE_QUAD_PREC_BE[::-1]
_IBM_DOUBLE_DOUBLE_BE = ['301', '235', '157', '064', '124', '000', '000', '000'] + ['000'] * 8
_IBM_DOUBLE_DOUBLE_LE = ['000', '000', '000', '124', '064', '157', '235', '301'] + ['000'] * 8

def long_double_representation(lines):
    """Given a binary dump as given by GNU od -b, look for long double
    representation."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup_common.long_double_representation', 'long_double_representation(lines)', {'_AFTER_SEQ': _AFTER_SEQ, 'copy': copy, '_BEFORE_SEQ': _BEFORE_SEQ, '_INTEL_EXTENDED_12B': _INTEL_EXTENDED_12B, '_MOTOROLA_EXTENDED_12B': _MOTOROLA_EXTENDED_12B, '_INTEL_EXTENDED_16B': _INTEL_EXTENDED_16B, '_IEEE_QUAD_PREC_BE': _IEEE_QUAD_PREC_BE, '_IEEE_QUAD_PREC_LE': _IEEE_QUAD_PREC_LE, '_IBM_DOUBLE_DOUBLE_LE': _IBM_DOUBLE_DOUBLE_LE, '_IBM_DOUBLE_DOUBLE_BE': _IBM_DOUBLE_DOUBLE_BE, '_IEEE_DOUBLE_LE': _IEEE_DOUBLE_LE, '_IEEE_DOUBLE_BE': _IEEE_DOUBLE_BE, 'lines': lines}, 1)

def check_for_right_shift_internal_compiler_error(cmd):
    """
    On our arm CI, this fails with an internal compilation error

    The failure looks like the following, and can be reproduced on ARM64 GCC 5.4:

        <source>: In function 'right_shift':
        <source>:4:20: internal compiler error: in expand_shift_1, at expmed.c:2349
               ip1[i] = ip1[i] >> in2;
                      ^
        Please submit a full bug report,
        with preprocessed source if appropriate.
        See <http://gcc.gnu.org/bugs.html> for instructions.
        Compiler returned: 1

    This function returns True if this compiler bug is present, and we need to
    turn off optimization for the function
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup_common.check_for_right_shift_internal_compiler_error', 'check_for_right_shift_internal_compiler_error(cmd)', {'textwrap': textwrap, 'cmd': cmd}, 1)

