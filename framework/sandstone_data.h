/*
 * Copyright 2022 Intel Corporation.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SANDSTONE_DATA_H
#define SANDSTONE_DATA_H

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "fp_vectors/Floats.h"

enum DataType {
    //SizeMask = 0x3f,
    UInt8Data = 0,
    UInt16Data = 1,
    UInt32Data = 3,
    UInt64Data = 7,
    UInt128Data = 15,

    DataIsSigned = 0x80,
    Int8Data = UInt8Data | DataIsSigned,
    Int16Data = UInt16Data | DataIsSigned,
    Int32Data = UInt32Data | DataIsSigned,
    Int64Data = UInt64Data | DataIsSigned,
    Int128Data = UInt128Data | DataIsSigned,

    DataIsFloatingPoint = 0x40,
    Float16Data = UInt16Data | DataIsFloatingPoint,
    BFloat16Data = 2 | DataIsFloatingPoint,
    Float32Data = UInt32Data | DataIsFloatingPoint,
    Float64Data = UInt64Data | DataIsFloatingPoint,
    Float80Data = 9 | DataIsFloatingPoint,
    Float128Data = UInt128Data | DataIsFloatingPoint,
    HFloat8Data,
    BFloat8Data,
};

/* Shared scalar-type mappings. */
#if ULONG_MAX == ULLONG_MAX /* LP64 (e.g. Linux, BSD) */
#   define SANDSTONE_ULONG_TYPE UInt64Data
#   define SANDSTONE_LONG_TYPE Int64Data
#else                       /* LLP64 (e.g. Windows) */
#   define SANDSTONE_ULONG_TYPE UInt32Data
#   define SANDSTONE_LONG_TYPE Int32Data
#endif

#define SANDSTONE_SCALAR_TYPES(T) \
    T(_Bool, UInt8Data) \
    T(char, UInt8Data) \
    T(uint8_t, UInt8Data) \
    T(uint16_t, UInt16Data) \
    T(uint32_t, UInt32Data) \
    T(unsigned long, SANDSTONE_ULONG_TYPE) \
    T(unsigned long long, UInt64Data) \
    T(__uint128_t, UInt128Data) \
    T(int8_t, Int8Data) \
    T(int16_t, Int16Data) \
    T(int32_t, Int32Data) \
    T(long, SANDSTONE_LONG_TYPE) \
    T(long long, Int64Data) \
    T(__int128_t, Int128Data) \
    T(HFloat8, HFloat8Data) \
    T(BFloat8, BFloat8Data) \
    T(BFloat16, BFloat16Data) \
    T(Float16, Float16Data) \
    T(float, Float32Data) \
    T(double, Float64Data)

#ifdef __SIZEOF_FLOAT128__
struct Float128
{
    __float128 payload;

#ifndef __f128
#   define __f128(x) x##q
#endif
#ifdef __cplusplus
    Float128() = default;
    Float128(long double f) : payload(f) {}

    static constexpr int digits = 113;
    static constexpr int digits10 = 33;
    static constexpr int max_digits10 = 6;  // log2(digits)
    static constexpr int min_exponent = -16381;
    static constexpr int min_exponent10 = -4931;
    static constexpr int max_exponent = 16384;
    static constexpr int max_exponent10 = 4932;

    static constexpr bool radix = 2;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = has_quiet_NaN;
    static constexpr std::float_denorm_style has_denorm = std::denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr std::float_round_style round_style =
            std::round_toward_zero;   // unlike std::numeric_limits<float>::round_style

    static constexpr __float128 min()
    { return __f128(0x1p-16382); }
    static constexpr __float128 max()
    { return __f128(1.18973149535723176508575932662800702e+4932); }
    static constexpr __float128 lowest()
    { return -max(); }
    static constexpr __float128 denorm_min()
    { return __f128(6.47517511943802511092443895822764655e-4966); }
    static constexpr __float128 epsilon()
    { return __f128(1.92592994438723585305597794258492732e-34); }
    static constexpr __float128 round_error()
    { return __f128(0.5); }
    static constexpr __float128 infinity()
    { return __builtin_inff128(); }
    static constexpr __float128 neg_infinity()
    { return -__builtin_inff128(); }
    static constexpr __float128 quiet_NaN()
    { return __builtin_nanf128(""); }
    static constexpr __float128 signaling_NaN()
    { return __builtin_nansf128(""); }
#endif
};
#endif // __FLT128_MAX__

#ifdef __cplusplus

namespace SandstoneDataDetails {
enum { MaxDataTypeSize = 16 };

static constexpr const char *type_name(DataType type)
{
    switch (type) {
    case UInt8Data: return "uint8_t";
    case UInt16Data: return "uint16_t";
    case UInt32Data: return "uint32_t";
    case UInt64Data: return "uint64_t";
    case UInt128Data: return "uint128_t";

    case Int8Data: return "int8_t";
    case Int16Data: return "int16_t";
    case Int32Data: return "int32_t";
    case Int64Data: return "int64_t";
    case Int128Data: return "int128_t";

    case HFloat8Data: return "HFloat8";
    case BFloat8Data: return "BFloat8";
    case Float16Data: return "_Float16";
    case BFloat16Data: return "_BFloat16";
    case Float32Data: return "float";
    case Float64Data: return "double";
    case Float80Data: return "_Float64x";   // long double is IEEE-754 extended precision binary64
    case Float128Data: return "_Float128";

    //case DataIsSigned:
    case DataIsFloatingPoint:
        __builtin_unreachable();
    }
    return nullptr;
}

template <DataType V> struct TypeToDataType_helper : std::true_type
{
    static constexpr DataType Type = V;
    static constexpr bool IsValid = true;
    static const char *name() { return type_name(Type); }
};

template <typename T> struct TypeToDataType
{ static constexpr bool IsValid = false; };

/* Generated from the shared list. */
#define SANDSTONE_DATATYPE_SPECIALIZATION(ctype, tag) \
    template<> struct TypeToDataType<ctype> : TypeToDataType_helper<tag> {};
SANDSTONE_SCALAR_TYPES(SANDSTONE_DATATYPE_SPECIALIZATION)
#undef SANDSTONE_DATATYPE_SPECIALIZATION

/* Exceptions stay handwritten below. */
template<> struct TypeToDataType<void>  : TypeToDataType_helper<UInt8Data> {};
template<> struct TypeToDataType<Float32> : TypeToDataType_helper<Float32Data> {};
template<> struct TypeToDataType<Float64> : TypeToDataType_helper<Float64Data> {};
template<> struct TypeToDataType<long double> :
        TypeToDataType_helper<sizeof(long double) == sizeof(double) ? Float64Data : Float80Data> {};
#ifdef __SIZEOF_FLOAT128__
template<> struct TypeToDataType<Float128> : TypeToDataType_helper<Float128Data> {};
template<> struct TypeToDataType<__float128> : TypeToDataType_helper<Float128Data> {};
#endif
#ifdef SANDSTONE_FP16_TYPE
template<> struct TypeToDataType<fp16_t> : TypeToDataType_helper<Float16Data> {};
#endif

template <typename T> concept ValidDataType = TypeToDataType<T>::IsValid;

static constexpr size_t type_real_size(DataType type)
{
    constexpr unsigned SizeMask = 0x3f;
    switch (type) {
        case HFloat8Data:
        case BFloat8Data:
            return 1;
        case BFloat16Data:
            return 2;
        default:
            return (type & SizeMask) + 1;
    }
}

static constexpr size_t type_size(DataType type)
{
    // special case: long double has 10 bytes of data but occupies 16 bytes
    if (type == Float80Data)
        return sizeof(long double);
    return type_real_size(type);
}

static constexpr size_t type_alignment(DataType type)
{
    return type_size(type);
}
} // namespace SandstoneDataDetails

#else
/* for C mode, we'll have to use _Generic */
#define SANDSTONE_DATATYPE_ASSOC(ctype, tag) ctype: tag,
#define DATATYPEFORTYPE(X) _Generic((X), \
        SANDSTONE_SCALAR_TYPES(SANDSTONE_DATATYPE_ASSOC) \
        long double: (sizeof(long double) == sizeof(double) ? Float64Data : Float80Data) \
    )

#endif /* __cplusplus */

#endif /* SANDSTONE_DATA_H */
