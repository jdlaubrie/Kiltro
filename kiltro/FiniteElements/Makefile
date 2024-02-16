PYTHON_VERSION = python3.8
PYTHON_LANG_LEVEL = 3
PYTHON_INCLUDE_PATH = /usr/include/python3.8/
PYTHON_LD_PATH = /usr/lib/
CYTHONFLAGS = -DPY_NO_DEPRECATED_API

CYTHON = cython --cplus -$(PYTHON_LANG_LEVEL)

REMOVE = rm

OPTFLAGS= -O3 -fno-strict-aliasing -DNDEBUG
CXXFLAGS = -fPIC -shared -pthread -Wall $(CYTHONFLAGS) $(OPTFLAGS)

build:
	$(CYTHON) ComputeSparsityPattern.pyx
	$(CXX) $(CXXFLAGS) ComputeSparsityPattern.cpp -o ComputeSparsityPattern.so -I. -I$(PYTHON_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION)

clean:
	$(REMOVE) ComputeSparsityPattern.cpp ComputeSparsityPattern.so
