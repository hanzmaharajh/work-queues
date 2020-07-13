#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>

#include <boost/iterator/transform_iterator.hpp>

template <size_t NumWorkers, size_t MaxBatchSize, typename WorkReturnType,
          typename WorkArg>
class BatchWorkStealingQueue {
 protected:
  // The details required for one task
  struct Task {
    using promise_type = std::promise<WorkReturnType>;

    promise_type promise;
    WorkArg arg;

    Task(promise_type promise, WorkArg t)
        : promise(std::move(promise)), arg(std::move(t)) {}

    // transform_iterator doesn't quite work right with lamdas, so we need these
    // functors.
    // TODO Consider some other transform iterator.
    struct PromiseFunc {
      [[nodiscard]] promise_type& operator()(Task& t) const {
        return t.promise;
      }
    };

    struct ArgFunc {
      [[nodiscard]] WorkArg& operator()(Task& t) const { return t.arg; }
    };
  };

  // A batch of tasks
  using batch_type = std::vector<Task>;

  // A queue consiting of our batches. Insert to the back, pop from the front.
  std::deque<batch_type> task_queue;

 public:
  // An iterator to a WorkArg
  using batch_args_iterator_type =
      boost::transform_iterator<typename Task::ArgFunc,
                                typename batch_type::iterator>;
  // An iterator to a promise
  using batch_promise_iterator_type =
      boost::transform_iterator<typename Task::PromiseFunc,
                                typename batch_type::iterator>;

  // The first parameter is an iterator to the beginning of a batch of work
  // arguments. The second is the end. The final argument is an iterator to a
  // range of promises that must be populated. The range is the same length as
  // the batch length.
  using batch_function_type = std::function<batch_promise_iterator_type(
      batch_args_iterator_type, batch_args_iterator_type,
      batch_promise_iterator_type)>;

  using future_type = std::future<WorkReturnType>;

  // The batch function.
  // After a batch is ready, this function is called.
  batch_function_type batch_function;

  constexpr static const size_t NUM_WORKERS = NumWorkers;

 protected:
  mutable std::mutex mutex;
  std::condition_variable condition_variable;
  bool stopped = false;
  std::array<std::thread, NumWorkers> workers;

  void pro_worker_func() {
    const typename Task::ArgFunc args_func;
    const typename Task::PromiseFunc promise_func;

    while (true) {
      std::vector<Task> batch;
      {
        std::unique_lock lock(mutex);
        condition_variable.wait(lock,
                                [&] { return stopped || !task_queue.empty(); });

        if (stopped) return;

        // Get the next batch of tasks
        batch.swap(task_queue.front());
        task_queue.pop_front();
      }

      // Call the batch function
      auto promise_end_it = batch_function(
          boost::make_transform_iterator(std::begin(batch), args_func),
          boost::make_transform_iterator(std::end(batch), args_func),
          boost::make_transform_iterator(std::begin(batch), promise_func));

      // TODO Should we do this? If the user doesn't populate the promises an
      // exception will be thrown in the future
      assert(promise_end_it ==
             boost::make_transform_iterator(std::end(batch), promise_func));
    }
  }

  void pro_stop() {
    {
      std::lock_guard lock(mutex);
      if (stopped) {
        // We're already stopped
        return;
      }
      stopped = true;

      // TODO Should we clear the queue? Or let it finish?
      task_queue.clear();
    }

    condition_variable.notify_all();

    for (auto&& w : workers) {
      w.join();
    }
  }

 public:
  BatchWorkStealingQueue(batch_function_type bf)
      : batch_function(std::move(bf)) {
    // Fire up our worker threads
    std::generate(std::begin(workers), std::end(workers), [&] {
      return std::thread(std::mem_fn(&BatchWorkStealingQueue::pro_worker_func),
                         this);
    });

    static_assert(NumWorkers > 0 && MaxBatchSize > 0);
  }

  ~BatchWorkStealingQueue() { pro_stop(); }

  // Put a WorkArg into the queue
  template <typename... Args>
  future_type emplace(Args&&... arg) {
    auto future = [&] {
      std::lock_guard lock(mutex);
      // If there's no room in this batch, add a new one.
      if (task_queue.empty() || std::size(task_queue.back()) == MaxBatchSize) {
        auto& v = task_queue.emplace_back();
        v.reserve(MaxBatchSize);
      }

      // Add our task to the last batch
      auto& task = task_queue.back().emplace_back(
          std::promise<WorkReturnType>(), WorkArg(std::forward<Args>(arg)...));

      return task.promise.get_future();
    }();

    condition_variable.notify_one();
    return future;
  }

  // Put a WorkArgs into the queue
  template <typename InputIterator, typename OutputIterator>
  OutputIterator add(InputIterator begin, const InputIterator& end,
                     OutputIterator out) {
    {
      std::lock_guard lock(mutex);
      if (task_queue.empty()) {
        auto& v = task_queue.emplace_back();
        v.reserve(MaxBatchSize);
      }

      while (begin != end) {
        // Do we need to add a new batch?
        if (std::size(task_queue.back()) == MaxBatchSize) {
          auto& v = task_queue.emplace_back();
          v.reserve(MaxBatchSize);
        }

        auto& v = task_queue.back();

        // Fill up this batch, as much as we can
        const auto remaining_in_batch = std::min<size_t>(
            MaxBatchSize - std::size(v), std::distance(begin, end));
        const auto batch_end = begin + remaining_in_batch;
        for (; begin != batch_end; ++begin) {
          *(out++) = v.emplace_back(std::promise<WorkReturnType>(), *begin)
                         .promise.get_future();
        }
      }
    };

    // TODO Should we count the number of batches we touched and notify_one if
    // we only touched one?
    condition_variable.notify_all();
    return out;
  }

  // TODO size()? It should be constant time - more cost under the lock
};

template <size_t NumWorkers>
class WorkStealingQueue {
  std::queue<std::function<void()>> tasks;
  std::array<std::thread, NumWorkers> workers;
  mutable std::mutex mutex;
  std::condition_variable condition_variable;
  std::atomic_bool stopped = false;

  void pro_worker_func() {
    std::function<void()> task;
    while (true) {
      {
        std::unique_lock lock(mutex);
        condition_variable.wait(lock,
                                [&] { return stopped || !tasks.empty(); });
        if (stopped) return;

        task = std::move(tasks.front());
        tasks.pop();
      }
      task();
    }
  }

 public:
  WorkStealingQueue() {
    std::generate(std::begin(workers), std::end(workers), [&] {
      return std::thread(std::mem_fn(&WorkStealingQueue::pro_worker_func),
                         this);
    });
  }

  ~WorkStealingQueue() { pro_stop(); }

  template <typename F, typename... Args>
  auto add(F&& f, Args&&... arg) {
    // This must be a shared_ptr, because the conversion from lambda to
    // std::function requires types that are copy constructable. Lame!
    auto ptr = std::make_shared<
        std::packaged_task<std::invoke_result_t<F, Args...>()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(arg)...));
    auto future = ptr->get_future();

    {
      std::lock_guard lock(mutex);
      // TODO: Should we construct the std::function outside the lock and then
      // move it into place? So as to reduce the locked time?
      tasks.emplace([task = std::move(ptr)] { task->operator()(); });
    }

    condition_variable.notify_one();
    return future;
  }

  void pro_stop() {
    if (stopped.exchange(true)) {
      // We're already stopped
      return;
    }

    condition_variable.notify_all();

    for (auto&& w : workers) {
      w.join();
    }
  }

  [[nodiscard]] size_t size() const {
    std::lock_guard lock(mutex);
    return std::size(tasks);
  }
};
