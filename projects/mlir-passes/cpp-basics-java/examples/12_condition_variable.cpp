#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

std::mutex mutex;
std::condition_variable cv;
std::queue<int> jobs;
bool done = false;

void producer() {
  for (int job = 1; job <= 3; ++job) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      jobs.push(job);
      std::cout << "producer: pushed job " << job << "\n";
    }

    cv.notify_one();
  }

  {
    std::lock_guard<std::mutex> lock(mutex);
    done = true;
  }
  cv.notify_one();
}

void consumer() {
  while (true) {
    std::unique_lock<std::mutex> lock(mutex);

    cv.wait(lock, [] {
      return !jobs.empty() || done;
    });

    if (jobs.empty() && done) {
      std::cout << "consumer: no more jobs\n";
      return;
    }

    int job = jobs.front();
    jobs.pop();
    lock.unlock();

    std::cout << "consumer: handling job " << job << "\n";
  }
}

int main() {
  std::thread worker(consumer);
  std::thread feeder(producer);

  feeder.join();
  worker.join();

  return 0;
}

