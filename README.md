# pyEvalData
Python Modul to evaluate SPEC data, area detector reciprocal space maps and scope traces

evalData module provide class definitions to read, average, plot, and fit data 
from spec files. The base class "spec" supports only spec files. The spec class
allows for user-defined counters as predefinitions or even at run-time.
The spec file is converted into a hdf5 file.
All file I/O are handled by the xrayutility package.
The child-class ara
The child-class areaDetector inherits from spec and has additional methods to read
images from area detectors such as the predifined Pilatus ((C) Dectris Ltd.) or
Prinction Instruments (C) CCD. The raw data is also stored in the same hdf5 file
as the spec data and can be easily vonverted to q- or HKL-space using the 
xrayutility package.
The child-class scopeTraces inherits from spec and allows to read time traces from a
scope (here a LeCroy (C) scope) which are stored as txt-files on the harddrive.
The read data is stored in the same hdf5 file as the spec data. 
