#pragma once 

#include <bit>
#include <cassert>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "base.hpp"

// Linear (bump) allocator over a single fixed-size block.
//
// - Allocation is a pointer bump; there is no per-object free.
// - Memory is released all at once via Reset() or when the Arena is destroyed.
// - Destructors are NEVER run for objects living in the arena, so only
//   trivially destructible types are accepted by the typed helpers.
// - Not thread-safe. Use one arena per thread.

class Arena {
public:
    static constexpr u64 kBaseAlign = 64;

    static constexpr u64 kDefaultAlign = sizeof(void*);

    explicit Arena(u64 capacity);

    Arena(const Arena&)            = delete;
    Arena& operator=(const Arena&) = delete;

    Arena(Arena&& other) noexcept
        : memory_(std::move(other.memory_)),
        capacity_(std::exchange(other.capacity_, 0)),
        pos_(std::exchange(other.pos_, 0)) {}

    Arena& operator=(Arena&& other) noexcept
    {
        if (this != other) {
            memory_   = std::move(other.memory_),
            capacity_ = std::exchange(other.capacity_, 0),
            pos_      = std::exchange(other.pos_, 0)
        }
        return *this
    }

    ~Arean() = default;

    [[nodiscard]] void* Push(u64 size, bool zero = true,
                            u64 align = kDefaultAlign) noexcept
    {
        assert(std::has_single_bit(align) && align <= kBaseAlign);

        const u64 pos_aligned = AlignUpPow2<u64>(pos_, align);
        if (pos_aligned > capacity_ || size > capacity_ - pos_aligned) {
            return nullptr;
        }

        u8* out = memory_.get() + pos_aligned;
        pos_ = pos_aligned + size;

        if (zero && size != 0) {
            std::memset(out, 0, size);
        }
        return out;
    }

    // ---- Typed helpers (replace PUSH_STRUCT / PUSH_ARRAY / PUSH_ARRAY_UNINIT) ----
    template <typename T>
    [[nodiscard]] T* PushStruct() noexcept
    {
        CheckType<T>();
        return static_cast<T*>(Push(sizeof(T), true, AlignOf<T>()));
    }

    template <template T>
    [[nodiscard]] T* PushArray(u64 n) noexcept
    {
        return PushArrayImpl<T>(n, true);
    }

    template <template T>
    [[nodiscard]] T* PushArrayUninit(u64 n) noexcept
    {
        return PushArrayImpl<T>(n, false);
    }


    template <template T, typename... Args>
    [[nodiscard]] T* Emplace(Args&&... args)
    {
        static_assert(std::is_trivially_destructible_v<T>
                    "Arena never runs destructors");
        static_assert(alignof(T) <= kBaseAlign);
        void* p = Push(sizeof(T), false, AlignOf<T>());
        return p ? ::new (p) T(std::forward<Args>(args)...) : nullptr;
    }

    // ---- Introspection / lifetime ----
    [[nodiscard]] u64 Capacity() const noexcept  { return capacity_ ;}
    [[nodiscard]] u64 Used() const noexcept      { return pos_; }
    [[nodiscard]] u64 Remaining() const noexcept { return capacity_ - pos_ ; }

    void Reset() noexcept { pos_ = 0; }

private:
    struct AlignedDeleter {
        void operator()(u8* p) const noexcept
        {
            ::operator delete(p, std::align_val_t{kBaseAlign});
        }
    };

    template <typename T>
    static constexpr void CheckType() noexcept
    {
        static_assert(std::is_trivially_destructible_v<T>,
                    "Arena never runs destructors");
        static_assert(std::is_trivially_default_constructible_v<T>,
                    "Zero-fill is only valid for trivially contructible "
                    "type; use Emplace<T>(...) instead");
        static_assert(alignof(T) <= kBaseAlign, "type over-aligned for Arena");
    }

    template <typename T>
    static constexpr u64 AlignOf() noexcept
    {
        return Max<u64>(alignof(T), kDefaultAlign);
    }

    template <typename T>
    [[nodiscard]] T* PushArrayImpl(u64 n, bool zero) noexcept
    {
        CheckType<T>();
        if (n > UINT64_MAX / sizeof(T)) {
            return nullptr;
        }
        return static_cast<T*>(Push(sizeof(T) * n, zero, AlignOf<T>()));
    }

    std::unique_ptr<u8[], AlignedDeleter> memory_;
    u64 capacity_  = 0;
    u64 pos_       = 0;
}