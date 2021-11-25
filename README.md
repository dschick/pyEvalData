# Welcome to pyEvalData

This is a Python module to read and evaluate experimental data.
It can handle raw data from different sources such as
[spec](https://certif.com/content/spec/),
[hdf5](https://www.hdfgroup.org/solutions/hdf5/),
[NeXus](https://www.nexusformat.org/) files which are common data formats at
synchrotrons, FELs, as well as in a growing number of laboratories.
The evaluation provides common functionalities such as binning,
error calculation, and advanced data manipulation via algebraic expressions as
well as pre- and post-data-filters.
Furthermore, advanced wrapper functions allow for plotting and fitting
sequences of one or multiple scans in dependence of an external parameter.

A minimal code example would look like this:

```python
import pyEvalData as ped
# define your data source
spec = ped.io.Spec(file_name='data.spec')
# initialize the evaluation
ev = ped.Evaluation(spec)
# define the x- and y-data
ev.xcol = 'motor1'
ev.clist = ['ct1', 'ct2', 'ct1/ct2']
# create a plot for scans 1-3
ev.plot_scans([1, 2, 3])
```

Please follow the [user guide](https://pyevaldata.readthedocs.io/en/latest/user_guide.html) and [examples](https://pyevaldata.readthedocs.io/en/latest/examples.html) for your first steps with `pyEvalData`.

## Features

- reading of several pre-defined raw data formats
  * [spec](https://certif.com/content/spec/)
  * [hdf5](https://www.hdfgroup.org/solutions/hdf5/)
  * [NeXus](https://www.nexusformat.org/)
  * user-defined text files
  * camera images (Dectris Pilatus, Princeton MTE, Greateyes, ...)
  * composite sources
- easy implementation of new raw data formats using an `interface class`
- common methods for plotting and fitting of experimental data, including:
  * data binning
  * error calculation
  * data manipulation via algebraic expressions
  * common data pre- and post-filters

## Installation

You can either install directly from
[pypi.org](https://www.pypi.org/project/pyEvalData) using the command

    $ pip install pyEvalData

or if you want to work on the latest develop release you can clone 
`pyEvalData` from the main git repository:

    $ git clone https://github.com/dschick/pyEvalData.git pyEvalData

To work in editable mode (source is only linked 
but not copied to the python site-packages), just do:

    $ pip install -e ./pyEvalData

Or to do a normal install with

    $ pip install ./pyEvalData

Optionally, you can also let pip install directly from the repository: 

    $ pip install git+https://github.com/dschick/pyEvalData.git

You can have the following optional installation to enable unit tests, as well
as building the documentation:

    $ pip install pyEvalData[testing]
    $ pip install pyEvalData[documentation]
    
## Contribute & Support

If you are having issues please let us know via the
[issue tracker](https://github.com/dschick/pyEvalData/issues).

You can contribute to the project via pull-requests following the
[GitHub flow concept](https://docs.github.com/en/get-started/quickstart/github-flow).

## License

The project is licensed under the MIT license.
