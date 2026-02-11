# C++ Guidelines

## std::optional

- **Do not use `.has_value()`** — use `operator bool` instead, as it is available and more concise.

```cpp
// Preferred
if (opt) { ... }

// Avoid
if (opt.has_value()) { ... }
```

## File organization

- Function order in `.cpp` files must match the declaration order in the corresponding `.h` file.
- In both `.h` and `.cpp` files: type definitions (structs, classes) come first, then `detail` namespace, then all function declarations/definitions (including templates).
