#include "prng.hpp"

namespace prng {

Prng& Global() noexcept
{
    thread_local Prng instance;
    return instance;
}

void Seed(u64 instance, u64 initseq) noexcept { Global().Seed(initstate, initseq); }
u32 Rand() noexcept                           { return Global().Next(); }
u32 Rand(u32 range) noexcept                  { return Global().NextBelow(range); }
f32 RandF() noexcept                          { return Global().NextFloat(); }
}