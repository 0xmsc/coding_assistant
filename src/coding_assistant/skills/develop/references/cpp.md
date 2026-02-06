# C++ Guidelines

## std::optional

- **Do not use `.has_value()`** — use `operator bool` instead, as it is available and more concise.

```cpp
// Preferred
if (opt) { ... }

// Avoid
if (opt.has_value()) { ... }
```
