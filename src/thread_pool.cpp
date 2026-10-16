#include "thread_pool.hpp"

#include <algorithm>
#include <atomic>

ThreadPool::ThreadPool(u32 num_threads)
{
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) { num_threads = 4; }
    }

    workers_.reserve(num_threads);
    for (u32 i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::Enqueue(std::function<void()> task)
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
}

void ThreadPool::ParallelFor(u32 start, u32 end, const std::function<void(u32 chunk_start, u32 chunk_end)>& fn)
{
    if (start >= end) { return; }
    const u32 total_items = end - start;
    const u32 num_workers = static_cast<u32>(workers_.size());

    if (num_workers <= 1 || total_items <= 1) {
        fn(start, end);
        return;
    }

    const u32 chunks = std::min(num_workers, total_items);

    struct SyncState {
        explicit SyncState(u32 c) : remaining(c) {}
        std::atomic<u32> remaining;
        std::mutex mutex;
        std::condition_variable cv;
    };

    auto sync = std::make_shared<SyncState>(chunks);

    for (u32 c = 0; c < chunks - 1; ++c) {
        const u32 c_start = start + static_cast<u32>((u64{c} * total_items) / chunks);
        const u32 c_end   = start + static_cast<u32>((u64{c + 1} * total_items) / chunks);

        Enqueue([c_start, c_end, &fn, sync] {
            if (c_start < c_end) {
                fn(c_start, c_end);
            }
            if (sync->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(sync->mutex);
                sync->cv.notify_one();
            }
        });
    }

    // Run the last chunk on calling thread
    const u32 last_start = start + static_cast<u32>((u64{chunks - 1} * total_items) / chunks);
    const u32 last_end   = end;
    if (last_start < last_end) {
        fn(last_start, last_end);
    }

    if (sync->remaining.fetch_sub(1, std::memory_order_acq_rel) > 1) {
        std::unique_lock<std::mutex> lock(sync->mutex);
        sync->cv.wait(lock, [&sync] {
            return sync->remaining.load(std::memory_order_acquire) == 0;
        });
    }
}

ThreadPool& ThreadPool::Global() noexcept
{
    static ThreadPool global_pool;
    return global_pool;
}
