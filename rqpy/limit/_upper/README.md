The below line of code was run in the command line to create the `upper.pyf` file.

```
f2py -m upper -h upper.pyf Upper.f UpperLim.f
```

For testing purposes, the below line of code can be run to create a Python module.

```
f2py -c -m upper upper.pyf Upper.f UpperLim.f CMaxinf.f y_vs_CLf.f Cinf.f ConfLev.f CombConf.f C0.f CnMax.f CERN_Stuff.f

```

See the `setup.py` file for how the code is compiled for the package.
