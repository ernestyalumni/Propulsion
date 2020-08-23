# CMake Integration

## Integrating CoolProp

cf. http://www.coolprop.org/coolprop/wrappers/StaticLibrary/index.html#static-library

Go to the line for `ADD_SUBDIRECTORY` for `CoolProp` in the `CMakeLists.txt`. You'll have to specify where CoolProp is on your filesystem.

*Example Compilation Usage:*


```
cmake -DCMAKE_C_FLAGS=gcc-8 -DCMAKE_CXX_FLAGS=g++-8 -DCMAKE_BUILD_TYPE=Release ../
```

