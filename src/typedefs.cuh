#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

using u64 = unsigned long long;
static_assert(sizeof(u64) == 8, "u64 must alias a 64-bit unsigned integer type.");

using f32 = float;
static_assert(sizeof(f32) == 4, "f32 must alias a 32-bit floating point type.");

using u32 = unsigned int;
static_assert(sizeof(u32) == 4, "u32 must alias a 32-bit unsigned integer type.");

using i32 = int;
static_assert(sizeof(i32) == 4, "i32 must alias a 32-bit signed integer type.");

using u8 = unsigned char;
static_assert(sizeof(u8) == 1, "u8 must alias an 8-bit unsigned integer type.");

#endif
