# 12 std::condition_variable

## What It Demonstrates

`std::condition_variable` lets one thread sleep until another thread changes
shared state and sends a notification.

The usual pattern is:

- protect shared state with `std::mutex`
- wait with `std::unique_lock<std::mutex>`
- use `cv.wait(lock, predicate)`
- notify with `notify_one()` or `notify_all()`

`std::lock_guard` is not enough for `wait`, because `wait` must temporarily
unlock the mutex while the thread is sleeping, then lock it again after wakeup.

## Example

See `examples/12_condition_variable.cpp`.

Build and run:

```bash
g++ -std=c++17 -pthread examples/12_condition_variable.cpp -o /tmp/cv_demo && /tmp/cv_demo
```

