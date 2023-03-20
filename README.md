# currycompose
A crate providing a trait for performing currying (and non-currying) function-composition in rust.

A can be composed with B if A implements FnOnce and takes one or more argument, while B implements FnOnce and returns something of the same type as the first argument of A.

- For instance (A, B) -> Y can be composed with (C) -> A, yiedling (B, C) -> Y (currying).

- Composing (A, B) -> Y with (C) -> A then with (D) -> B yields (C, D) -> Y (currying twice).

- While of course (A) -> Y composed with (B) -> A, yields (B) -> Y as expected (non-currying composition).

Currying functions which implement FnMut or Fn will yield something also implementing FnMut/Fn if both operands do.