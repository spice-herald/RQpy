The below line of code was run in the command line to create the `upperlim.pyf` file.

```
f2py -m upperlim -h upperlim.pyf UpperLim.f
```

For testing purposes, the below line of code can be run to create a Python module.

```
f2py -c -m upperlim upperlim.pyf UpperLim.f CMaxinf.f y_vs_CLf.f Cinf.f ConfLev.f CERN_Stuff.f

```

See the `setup.py` file for how the code is compiled for the package.
