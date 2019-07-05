

`python hhtseis.py -h `

usage: hhtseis.py [-h] [--xhdr XHDR] [--yhdr YHDR] [--xyscalerhdr XYSCALERHDR]

                  [--tracerange TRACERANGE TRACERANGE] [--numofimf NUMOFIMF]

                  [--ifreqsmoothwlen IFREQSMOOTHWLEN] [--plottrace PLOTTRACE]

                  [--outdir OUTDIR] [--hideplots]
                  segyfile

Hilbert Huang Transform: Empirical Mode Decomposition generating Intrinsic Mode Functions

positional arguments:
  segyfile              segy file name

optional arguments:
  -h, --help            show this help message and exit
  --xhdr XHDR           xcoord header.default=73
  --yhdr YHDR           ycoord header. default=77
  --xyscalerhdr XYSCALERHDR
                        hdr of xy scaler to divide by.default=71
  --tracerange TRACERANGE TRACERANGE
                        Start and end trace #s. default full range
  --numofimf NUMOFIMF   # of IMF to calculate. default= program decides, i.e.
                        variable
  --ifreqsmoothwlen IFREQSMOOTHWLEN
                        smooth ifreq window length. default = 21
  --plottrace PLOTTRACE
                        plot increment. default=50000
  --outdir OUTDIR       output directory,default= same dir as input
  --hideplots           Only save to pdf. default =show and save

  ##  General Concept

  The program expects a segy for input, on which Hilbert Huang Transform (Empirical Mode Decomposition) is applied to generate as many Intrinsic Mode Functions (oscillations, approximately mono frequencies) and then for each IMF computing various trace attributes, e.g. envelope, instantaneous phase, and instantaneous frequency.

  The residual after extracting all IMFs represents the background trend.

  All the computed IMFs and their corresponding attributes are saved into seperate segy files. A total of 15 segy files are generated in the process
