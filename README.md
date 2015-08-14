# pyEvalData
Python Modul to evaluate SPEC data and Dectris Pilatus reciprocal space maps

evalData module provide class definitions to read, average, plot, and fit data 
from spec files. The base class "spec" supports only spec files. The spec class
allows for user-defined counters as predefinitions or even at run-time.
The spec file is converted into a hdf5 file.
All file I/O are handled by the xrayutility package.
The child-class pilatusXPP inherits from spec and has additional methods to read
Pilatus ((C) Dectris Ltd.) .tiff files and further convert them to q-space
using the xrayutility package.
