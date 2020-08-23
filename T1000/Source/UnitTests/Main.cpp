//------------------------------------------------------------------------------
/// \details BOTH the following 2 lines are needed: otherwise the following
/// linkage error is obtained:
/// lib64/crt1.o: in function `_start':
/// (.text+0x24): undefined reference to `main'
/// collect2: error: ld returned 1 exit status
/// make[2]: *** [UnitTests/CMakeFiles/Check.dir/build.make:121: Check] Error 1
/// make[1]: *** [CMakeFiles/Makefile2:248: UnitTests/CMakeFiles/Check.dir/all] Error 2
//------------------------------------------------------------------------------

#define BOOST_TEST_MODULE "T1000 Tests"

#include <boost/test/unit_test.hpp>