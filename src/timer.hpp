#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "base.hpp"

namespace rl {

struct ProfileEntry {
    u64 count      = 0;
    u64 total_ns   = 0;
    u64 min_ns     = std::numeric_limits<u64>::max();
    u64 max_ns     = 0;

    void Record(u64 duration_ns) noexcept
    {
        count++;
        total_ns += duration_ns;
        if (duration_ns < min_ns) { min_ns = duration_ns; }
        if (duration_ns > max_ns) { max_ns = duration_ns; }
    }

    [[nodiscard]] f64 TotalMillis() const noexcept { return static_cast<f64>(total_ns) / 1e6; }
    [[nodiscard]] f64 AvgMicros() const noexcept
    {
        return count > 0 ? (static_cast<f64>(total_ns) / (count * 1e3)) : 0.0;
    }
    [[nodiscard]] f64 MinMicros() const noexcept
    {
        return min_ns == std::numeric_limits<u64>::max() ? 0.0 : (static_cast<f64>(min_ns) / 1e3);
    }
    [[nodiscard]] f64 MaxMicros() const noexcept { return static_cast<f64>(max_ns) / 1e3; }
};

class ProfileRegistry {
public:
    static ProfileRegistry& Global() noexcept
    {
        static ProfileRegistry instance;
        return instance;
    }

    void Record(const std::string& name, u64 duration_ns)
    {
        entries_[name].Record(duration_ns);
    }

    void Clear() noexcept { entries_.clear(); }

    [[nodiscard]] const std::unordered_map<std::string, ProfileEntry>& Entries() const noexcept
    {
        return entries_;
    }

    void PrintSummary(std::ostream& os = std::cout) const
    {
        if (entries_.empty()) {
            os << "No profiling entries recorded.\n";
            return;
        }

        os << "\n"
           << "+--------------------------------+--------+------------+------------+---------------------+\n"
           << "| Scope                          | Calls  | Total (ms) | Avg (us)   | Min / Max (us)      |\n"
           << "+--------------------------------+--------+------------+------------+---------------------+\n";

        for (const auto& [name, e] : entries_) {
            os << "| " << std::left << std::setw(30) << name << " | "
               << std::right << std::setw(6) << e.count << " | "
               << std::fixed << std::setprecision(3) << std::setw(10) << e.TotalMillis() << " | "
               << std::fixed << std::setprecision(2) << std::setw(10) << e.AvgMicros() << " | "
               << std::setw(8) << e.MinMicros() << " / "
               << std::setw(8) << e.MaxMicros() << " |\n";
        }

        os << "+--------------------------------+--------+------------+------------+---------------------+\n\n";
    }

private:
    std::unordered_map<std::string, ProfileEntry> entries_;
};

class Timer {
public:
    using Clock = std::chrono::steady_clock;

    Timer() noexcept : start_(Clock::now()) {}

    void Reset() noexcept { start_ = Clock::now(); }

    [[nodiscard]] u64 ElapsedNanoseconds() const noexcept
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_).count();
    }

    [[nodiscard]] f64 ElapsedMicroseconds() const noexcept
    {
        return static_cast<f64>(ElapsedNanoseconds()) / 1e3;
    }

    [[nodiscard]] f64 ElapsedMilliseconds() const noexcept
    {
        return static_cast<f64>(ElapsedNanoseconds()) / 1e6;
    }

    [[nodiscard]] f64 ElapsedSeconds() const noexcept
    {
        return static_cast<f64>(ElapsedNanoseconds()) / 1e9;
    }

private:
    Clock::time_point start_;
};

class ScopedProfile {
public:
    explicit ScopedProfile(std::string name)
        : name_(std::move(name)), timer_() {}

    ~ScopedProfile()
    {
        ProfileRegistry::Global().Record(name_, timer_.ElapsedNanoseconds());
    }

private:
    std::string name_;
    Timer       timer_;
};

} // namespace rl

#define RL_CONCAT_IMPL(a, b) a##b
#define RL_CONCAT(a, b) RL_CONCAT_IMPL(a, b)
#define RL_PROFILE_SCOPE(name) ::rl::ScopedProfile RL_CONCAT(_profile_scope_, __COUNTER__)(name)

