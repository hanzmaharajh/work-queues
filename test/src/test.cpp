#include "work-queues/work_queues.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <map>
#include <numeric>
#include <random>
#include <vector>

#include <boost/thread/synchronized_value.hpp>

using clock_type = std::chrono::steady_clock;

using subrequest_return_type =
    std::tuple<size_t /* HTTP Response Code */, std::string /* Content */,
               clock_type::duration /* Response time  - for metrics */
               > /* Response */;

using subrequest_arg =
    std::tuple<std::string /* URL */, std::string /* method */,
               std::string /* Content */,
               std::map<std::string, std::string> /* headers */,
               clock_type::time_point /* Request time - for metrics */
               >;

using work_queue_type =
    BatchWorkStealingQueue<3, 25, subrequest_return_type, subrequest_arg>;

class BatchWorkQueueFixture : public work_queue_type, public ::testing::Test {
 public:
  boost::synchronized_value<std::vector<size_t>> batch_sizes;

  constexpr static const std::chrono::milliseconds round_trip_cost{50};
  constexpr static const std::chrono::milliseconds cost_per_subrequest{1};

  BatchWorkQueueFixture()
      : work_queue_type(
            // This function accepts a batch of inputs and an output iterator to
            // the corrasponding output promises.
            [&](auto batch_begin, const auto& batch_end, auto output_promises) {
              const auto batch_length = std::distance(batch_begin, batch_end);
              batch_sizes->emplace_back(batch_length);
              // Assemble the batch of subrequests into a single big request
              std::vector<subrequest_arg> big_request;
              big_request.reserve(batch_length);
              for (; batch_begin != batch_end; ++batch_begin) {
                big_request.emplace_back(*batch_begin);
              }

              // Do an actual HTTP request with the big, assembled, request.
              std::this_thread::sleep_for(round_trip_cost +
                                          batch_length * cost_per_subrequest);

              // Disassemble the result and set the promise values for each
              // subrequest.
              const auto now = clock_type::now();
              for (const auto& subrequest : big_request) {
                (output_promises++)
                    ->set_value(std::make_tuple(404, "Blah",
                                                now - std::get<4>(subrequest)));
              }

              return output_promises;
            }) {}

  template <typename FuturesArrayType>
  auto get_results(FuturesArrayType& futures) {
    // Get all our results
    std::array<subrequest_return_type, std::tuple_size<FuturesArrayType>::value>
        values;
    std::transform(std::begin(futures), std::end(futures), std::begin(values),
                   [](auto& f) { return f.get(); });

    // We've returned from get()-ing our futures. All the work is done. Let's
    // see how we did.

    // Let's see how big our batches were
    {
      const auto& synched = batch_sizes.value();
      auto count =
          std::accumulate(std::begin(synched), std::end(synched), size_t{0},
                          [](auto lhs, const auto& rhs) { return lhs + rhs; });
      EXPECT_EQ(count, std::size(futures));
    }

    // Our average delay
    const auto average_delay =
        std::accumulate(
            std::begin(values), std::end(values), clock_type::duration{},
            [](auto lhs, const auto& rhs) { return lhs + std::get<2>(rhs); }) /
        std::size(values);

    // What we would have expected from the same number of workers, but no
    // batching
    constexpr const auto nonbatched_delay =
        (round_trip_cost + cost_per_subrequest) *
        std::tuple_size<FuturesArrayType>::value / work_queue_type::NUM_WORKERS;

    // How much better did we do?
    [[maybe_unused]] const auto speed_up = nonbatched_delay / average_delay;

    return values;
  }
};

TEST_F(BatchWorkQueueFixture, Demo) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(1, 50);

  // We will create some HTTP requests.
  constexpr const size_t request_count = 100;
  std::array<work_queue_type::future_type, request_count> futures;
  for (size_t i = 0; i < request_count; ++i) {
    futures[i] =
        this->emplace("http://example.com", "GET", "",
                      std::map<std::string, std::string>(), clock_type::now());
    // Random delays, mimicking calls from different locations, so we get
    // different batch sizes
    std::this_thread::sleep_for(std::chrono::microseconds(dis(gen)));
  }

  const auto values = get_results(futures);

  for (auto&& v : values) {
    EXPECT_EQ(std::get<0>(v), 404);
    EXPECT_EQ(std::get<1>(v), "Blah");
    EXPECT_GE(std::get<2>(v), clock_type::duration::zero());
  }
}

TEST_F(BatchWorkQueueFixture, Add) {
  // We will create some HTTP requests.
  constexpr const size_t request_count = 1'000;
  std::array<work_queue_type::future_type, request_count> futures;
  std::deque<subrequest_arg> arg(
      request_count,
      subrequest_arg("http://example.com", "GET", "",
                     std::map<std::string, std::string>(), clock_type::now()));

  EXPECT_EQ(this->add(std::begin(arg), std::end(arg), std::begin(futures)),
            std::end(futures));

  const auto values = get_results(futures);

  for (auto&& v : values) {
    EXPECT_EQ(std::get<0>(v), 404);
    EXPECT_EQ(std::get<1>(v), "Blah");
    EXPECT_GE(std::get<2>(v), clock_type::duration::zero());
  }
}

TEST_F(BatchWorkQueueFixture, Moves) {
  this->pro_stop();
  // We will create some HTTP requests.
  constexpr const size_t request_count = 1;
  std::array<work_queue_type::future_type, request_count> futures;

  {
    std::string content(1000, 'A');
    const auto& data_ptr = std::data(content);

    this->emplace("http://example.com", "GET", std::move(content),
                  std::map<std::string, std::string>(), clock_type::now());

    EXPECT_EQ(data_ptr,
              std::data(std::get<2>(this->task_queue[0].front().arg)));
  }
  // Reset the fixture
  {
    std::lock_guard lock(this->mutex);
    this->task_queue.clear();
  }
  {
    std::string content(1000, 'A');
    const auto& data_ptr = std::data(content);

    std::vector<subrequest_arg> arg;
    arg.emplace_back("http://example.com", "GET", std::move(content),
                     std::map<std::string, std::string>(), clock_type::now());

    EXPECT_EQ(
        this->add(std::make_move_iterator(std::begin(arg)),
                  std::make_move_iterator(std::end(arg)), std::begin(futures)),
        std::end(futures));

    EXPECT_EQ(data_ptr,
              std::data(std::get<2>(this->task_queue[0].front().arg)));
  }
}

TEST(BatchWorkQueue, Moves) {
  {
    std::string content(1000, 'A');
    const auto& data_ptr = std::data(content);
    work_queue_type q(
        [&](auto batch_begin, const auto& batch_end, auto output_promises) {
          auto& content = std::get<2>(*batch_begin);
          EXPECT_EQ(std::data(content), data_ptr);
          (output_promises++)
              ->set_value(std::make_tuple(404, std::move(content),
                                          clock_type::duration::zero()));
          return output_promises;
        });
    auto future =
        q.emplace("http://example.com", "GET", std::move(content),
                  std::map<std::string, std::string>(), clock_type::now());

    EXPECT_EQ(std::data(std::get<1>(future.get())), data_ptr);
  }
  {
    std::string content(1000, 'A');
    const auto& data_ptr = std::data(content);
    work_queue_type q(
        [&](auto batch_begin, const auto& batch_end, auto output_promises) {
          auto& content = std::get<2>(*batch_begin);
          EXPECT_EQ(std::data(content), data_ptr);
          (output_promises++)
              ->set_value(std::make_tuple(404, std::move(content),
                                          clock_type::duration::zero()));
          return output_promises;
        });

    std::vector<subrequest_arg> v;
    v.emplace_back("http://example.com", "GET", std::move(content),
                   std::map<std::string, std::string>(), clock_type::now());
    work_queue_type::future_type future;

    q.add(std::make_move_iterator(std::begin(v)),
          std::make_move_iterator(std::end(v)), &future);

    EXPECT_EQ(std::data(std::get<1>(future.get())), data_ptr);
  }
}

TEST(WorkQueue, WorkQueue) {
  WorkStealingQueue<10> q;

  std::atomic_uint count = 0;

  auto f = [&] {
    count++;
    return 1;
  };

  constexpr const size_t expected_count = 10'000;
  std::array<std::future<int>, expected_count> futures;
  for (size_t i = 0; i < expected_count; ++i) {
    futures[i] = q.add(f);
  }

  auto acc = std::accumulate(std::begin(futures), std::end(futures), 0,
                             [](const auto s, auto& f) { return s + f.get(); });
  EXPECT_EQ(acc, expected_count);

  EXPECT_EQ(count, expected_count);
  q.pro_stop();
  q.pro_stop();
}
