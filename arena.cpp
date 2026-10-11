#include "arena.hpp"

#include <limits>

Arena::Arena(u64 capacity)
{
    if (capacity > std::numeric_limits<std::size_t>::max()) {
        throw std::bad_alloc{};
    } 

    memory_.reset(static_cast<u8*>(::operator new(
        static_cast<std::size_t>(capacity), std::align_val_t{kBaseAlign})));
    capacity_ = capacity;
    pos_ = 0;
}