"""Backend implementations for the embodied runtime.

* ``procthor``    -- the real ProcTHOR / AI2-THOR backend (primary product path).
* ``deterministic`` -- CPU test harness only. It is **not** a training source and
  is never presented as a ProcTHOR environment; it exists solely to verify the
  runtime contracts, the closed-loop transition, and the Done verifier without a
  Unity binary.
"""
