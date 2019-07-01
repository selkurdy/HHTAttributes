#  __HHTAttributes__
__Generate attributes from Hilbert Huang Transform (aka Empirical Mode Decomposition)__  

##  __GENERAL CONCEPT__

> **The Hilbert-Huang transform (HHT)** is NASA's designated name for the combination of the **Empirical Mode Decomposition (EMD)** and the **Hilbert Spectral Analysis (HSA)**.
> 
>  It is an adaptive data analysis method designed specifically for analyzing data from *__nonlinear__* and *__nonstationary__* processes. The key part of the HHT is the EMD method with which any complicated data set can be decomposed into a finite and often small number of components, called **Intrinsic Mode Functions (IMF)**. An IMF is defined as any function having the same (or differing at most by one) numbers of zero-crossing and extrema, and also having symmetric envelopes defined by the local maxima and minima, respectively. The definition of an IMF guarantees a well-behaved Hilbert transform of the IMF. This decomposition method operating in the time domain is adaptive, and, therefore, highly efficient. Since the decomposition is based on the local characteristic time scale of the data, it is applicable to nonlinear and non-stationary processes. With the Hilbert transform, the IMF's yield instantaneous frequencies as functions of time that give sharp identifications of imbedded structures. The final presentation of the results is an energy-frequency-time distribution, designated as the Hilbert spectrum.
> 
> HHT is an *empirical approach*, and has been tested and validated exhaustively but only empirically. In almost all the cases studied, HHT gives results much sharper than any of the traditional analysis methods in time-frequency-energy representation. Additionally, it reveals true physical meanings in many of the data examined. 
> The development of HHT is motivated by the need to describe nonlinear and nonstationary distorted waves in details.
[Wikipedia](http://www.scholarpedia.org/article/Hilbert-Huang_transform)

> Methods based on the Fourier transform are almost synonymous with frequency domain processing of signals.  However, its popularity and effectiveness have a downside. It has led to a very specific and limited view of frequency in the context of signal processing. Simply put, frequencies, in the context of Fourier methods, are just a collection of the individual frequencies of periodic signals that a given signal is composed of. A complementary method is the Hilbert spectral analysis (__HSA__).

> __HSA__ does not in any way replace Fourier, but that it provides an alternative interpretation of frequency, and an alternative view of __*nonlinear*__ and __*nonstationary*__ phenomena.

## __A note on time-frequency analysis__

> The energy spectrum is perfectly valid, but the Fourier transform is essentially an integral over time. Thus, we lose all information that varies with time. We can only identify amplitudes at specific frequencies, but we cannot identify at which time these frequencies occur. 
> 
>A powerful variant of the Fourier transform is the **wavelet transform**. By using finite-support basis functions, wavelets are able to approximate even nonstationary data. These basis functions possess most of the desirable properties required for linear decomposition (like orthogonality, completeness , etc) and they can be drawn from a large dictionary of wavelets. This makes the wavelet transform a versatile tool for analysis of nonstationary data. But the wavelet transform is still a linear decomposition and hence suffers from related problems like the uncertainty principle. Moreover, like Fourier, the wavelet transform too is non-adaptive. The basis functions are selected a priori and consequently make the wavelet decomposition prone to spurious harmonics and ultimately incorrect interpretations of the data.

> The next approach is time-frequency analysis, which is a field that deals with signal processing in both time *__and__* frequency domain. A tradeoff is made between time and frequency resolution of a signal, depending on what makes more sense for a particular application. __HHT__ too is a tool for time-frequency analysis. 

## __Time Frequency Representations__
> A popular choice to represent both time and frequency characteristics is the *__short-time Fourier transform (STFT)__*, which, simply put, transforms contiguous chunks of the input and aggregates the result in a 2 dimensional form, where one axis represents frequency and the other represents time. 

> Unfortunately, none of these methods are fully data driven, in that they rely very strongly on a *parametric model* of the data, and the representation is only as good as the model. A major drawback of time frequency distributions that depend on Fourier or wavelet models is that they don't allow for an *__"unsupervised"__* or *__data driven__* approach to time series analysis.
[jaidevd](https://github.com/jaidevd/pyhht/blob/dev/docs/examples/notebooks/fourier_limitations.ipynb)

## **Instantaneous Frequency**
An alternative to Fourier Transform and Wavelet Transform  is *instantaneous frequency* which is defined for analytic signals.

## __Instantaneous Frequencies from HHT__
> The real innovation of the HHT is an iterative algorithm called the **Empirical Mode Decomposition (EMD)** which breaks a signal down into  **Intrinsic Mode Functions (IMFs)** which are characterized by being narrowband, nearly monocomponent and having a large time-bandwidth product. This allows the IMFs to have well-defined Hilbert transforms and consequently, *__physically meaningful instantaneous frequencies__*. 

##  **__Intrinsic Mode Functions__**

> A function is called an intrinsic mode function when:
> 
> + The number of its extrema and zero-crossings differ at most by unity.
> + The mean of the local envelopes defined by its local maxima and that defined by its local minima should be zero at all times.
> 
> Condition 1 ensures that there are no localized oscillations in the signal and it crosses the X-axis at least once before it goes from one extremum to another, which makes it adaptive. 
> 
> Condition 2 ensures meaningful instantaneous frequencies

## **Empirical Mode Decomposition**
> The EMD is an iterative algorithm which breaks a signal down into IMFs. The process is performed as follows:

> + Find all local extrema in the signal.
> + Join all the local maxima with a cubic spline, creating an upper envelope. 
> + Repeat for local minima and create a lower envelope.
> + Calculate the mean of the envelopes.
> + Subtract mean from original signals.
> + Repeat steps 1-4 until result is an IMF.
> + Subtract this IMF from the original signal.
> + Repeat steps 1-6 till there are no more IMFs left in the signal.

## **__Properties of Intrinsic Mode Functions__**
> By virtue of the EMD algorithm, the decomposition is complete, meaning the sum of the IMFs and the residuals subtracted from the input signal leaves behind only a negligible residual. The decomposition is almost orthogonal. 

> The greatest advantage of the IMFs are well-behaved Hilbert transforms, enabling the extraction of physically meaningful instantaneous frequencies.

> IMFs have large time-bandwidth products, which indicates that they tend to move away from the lower bound of the Heisenberg-Gabor inequality, thereby avoiding the limitations of the Uncertainty principle

## **__Seismic Application of Hilbert Huang Transform (HHT):__**

> The module comprises three command line programs:
> 
>  + **hhtseis.py**:
> 
> > >The program reads in a segy, 2D or 3D, and transforms each trace sequentially to 15 attributes. Each attribute is written to a seperate segy file. Each input trace is HHT transformed by Empirical Mode Decomposition to Intrinsic Mode Functions (IMF). The number of IMFs is dependent on the data.
> 
> > > The first 4 functions are considered to be reliable data, beyond which IMFs are viewed as artifacts. The last IMF is the residual represent the trend. 

> > > For instructions to run the program please read the file `hhtseis.md` 
> + **mlhht.py**:
> 
> > > The program expects a segy file and performs the EMD (HHT) to generate the IMFs and for each IMF three attributes are generated, viz. the envelope, instantaneous phase, and instantaneous frequency. 
> 
> > >  This results in 21 attributes that are saved as a csv file and a binary npy file. The logic is that this csv can be later used as input to any Machine Learning model building.
> 
> > > For instructions to run the program please read the file `mlhht.md` 
> 
>  + **clustering2segy.py**:

> > > The program expects the csv file generated by `mlhht.py` and the same segy file used to generate the csv file.

>>> The data is scaled and submitted to **UMAP (Uniform Manifold Approximation Projection)** for dimensionality reduction to 3 features only. The resulting 3 features are submitted to KMeans clustering to 5 clusters.

>>> The clustered results are written back to a new segy file

> > > For instructions to run the program please read the file `clustering2segy.md` 


## **__Seismic Interpretation Considerations__**
+ **Uses for the Trend volume**:
+ **Multi volume interpretation of IMFs**:
+ **Identification of multiples generating geologic intervals**:
+ 