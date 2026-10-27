# Development status

This software is released as open source on GitHub and will soon have external
consumers. Treat public APIs, configuration, file formats, CLI behavior, and
documented semantics as compatibility surfaces.

# Priorities

Decisions are made in this order:

1. **Speed.** Make things as fast as possible. When choosing between designs,
   pick the one that performs best, and do not accept performance regressions
   for the sake of convenience, cleanliness, or compatibility.
2. **Backward compatibility.** Preserve it whenever it can be done without
   sacrificing speed. Prefer compatible changes (additive APIs, deprecation
   periods, defaults that keep existing behavior).

# Architectural continuity

Extend the repository's established architecture instead of creating a
parallel implementation of responsibilities it already owns. Before adding a
build wrapper, dependency manager, packaging layer, configuration system, or
other orchestration:

- inspect the current implementation and the commits that established it;
- identify the concrete requirement the existing owner cannot satisfy; and
- prefer extending that owner when it can satisfy the requirement cleanly.

Separating lifecycle stages does not by itself justify separate tooling. For
example, independent dependency preparation, compilation, model export,
packaging, and qualification may remain distinct stages while CMake still owns
native dependency acquisition and compilation. Do not replace or bypass a
recent architectural decision without calling out the conflict and obtaining
explicit user agreement.

Tests can show that new machinery works, but not that it should exist. Judge a
new abstraction by whether it reduces the total concepts, code paths, and
ownership boundaries a maintainer must understand. If it duplicates an
established tool or merely moves complexity between files, remove it.
