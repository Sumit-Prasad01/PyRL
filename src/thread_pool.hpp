#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "base.hpp"

class ThreadPool {
public:
    explicit ThreadPool(u32 num_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Enqueue an arbitrary task.
    void Enqueue(std::function<void()> task);

    // Parallel-for over [start, end) split across worker threads.
    // Blocks until all chunks have finished executing.
    void ParallelFor(u32 start, u32 end, const std::function<void(u32 chunk_start, u32 chunk_end)>& fn);

    [[nodiscard]] u32 ThreadCount() const noexcept { return static_cast<u32>(workers_.size()); }

    // Global default thread pool instance.
    static ThreadPool& Global() noexcept;

private:
    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex                        queue_mutex_;
    std::condition_variable           cv_;
    bool                              stop_ = false;
};
