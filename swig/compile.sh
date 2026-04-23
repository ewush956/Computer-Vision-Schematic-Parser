#!/bin/bash
rm -f *.o *.so trace_skeleton_wrap.c

swig -python trace_skeleton.i

gcc -O3 -fPIC -c trace_skeleton.c trace_skeleton_wrap.c $(python3-config --includes)

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS Logic
    echo "Detected macOS - using -bundle and dynamic_lookup"
    gcc -bundle -undefined dynamic_lookup *.o -o _trace_skeleton.so $(python3-config --ldflags)
else
    # Linux / Codespaces Logic
    echo "Detected Linux - using -shared"
    gcc -shared *.o -o _trace_skeleton.so $(python3-config --ldflags)
fi

echo "Build Complete: _trace_skeleton.so created."