#ifndef Hmod_TestPaths
#define Hmod_TestPaths
// If this file is included unconfigured, this will fail.
// When configured, 1 is replaced by 1.
#if 1 != 1
assert (false);
#endif

const char* UnittestSourceDir = "/code/openmalaria/unittest/";	// must end with '/'
const char* UnittestScenario = "/code/openmalaria/build/unittest/configured/scenario.xml";
#endif
