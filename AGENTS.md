# Development status

This software is unreleased. Do not preserve backward compatibility unless the
user explicitly requests it. Prefer the cleanest current design, and remove
obsolete APIs, configuration, and implementation paths instead of adding
compatibility shims.

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
