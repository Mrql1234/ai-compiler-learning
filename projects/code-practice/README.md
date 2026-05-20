# code-practice

This directory contains small C++ standard library practice programs.

## Entry Files

- `test_vector.cpp`: basic `std::vector` operations.
- `test_string.cpp`: basic `std::string` operations.
- `test_queue.cpp`: basic `std::queue` operations.
- `test_dequeue.cpp`: basic `std::deque` operations and an example of invalid access after emptying the deque.
- `test_set.cpp`: basic `std::unordered_set` and `std::set` operations.
- `test_map.cpp`: basic `std::map` iteration and bound lookup.
- `test_unordered_map.cpp`: basic `std::unordered_map` lookup, update, erase, and iteration.

## Build And Run

Build one example:

```bash
g++ -std=c++17 test_vector.cpp -o /tmp/test_vector
/tmp/test_vector
```

Build all source files from the repository root:

```bash
for src in projects/code-practice/*.cpp; do
  out="/tmp/$(basename "${src%.cpp}")"
  g++ -std=c++17 "$src" -o "$out"
done
```

