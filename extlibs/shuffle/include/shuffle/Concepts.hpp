#pragma once

#ifdef __cpp_concepts_DISABLE
#define REQUIRES(X) requires(X)
#else
#define REQUIRES(X)
#endif
