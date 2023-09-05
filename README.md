# IWT_dedup
The image fuzzy deduplication scheme based on IWT algorithm.

## Dependency
### Language
Python: version 3.11.5

### Library
1. numpy: version 1.25.2
2. PyWavelets: version 1.4.1
3. opencv-python: version 4.8.0.76
4. pycryptodome: version 3.17

### Dataset
1. Bill images dataset.  
*see detail in directory: ./bills/*

## Source Code
- iwt.py: implement Integer Wavelet Transform (IWT) algorithm.
- scheme.py: implement the proposed scheme.
- liu_fuzzy.py: implement scheme [1].
- performance.py: experiment in performance.
- security.py: experiment in security.

## Experiment
> **Performance**  
> 1. Time  
> - Upload: IWT (divide into base and diff) + base check + base encryption  
> - Download: base decrytion + IWT reverse?  
> - Diff Deduplication Check: compare with [1], which is based on   
> 
> 2. Deduplication rate  
> - 

> **Security**
> 

## Output
Base dictionary: **./result**
- Performance Experiment: **$(Base dictionary)/perf/**
- Security Experiment: **$(Base dictionary)/secu/**

## Reference
[1] Liu, X., Tang, X., Jin, L., Chen, X., Zhou, Z., Zhang, S. (2022). Secure Cross-User Fuzzy Deduplication for Images in Cloud Storage. In: Tan, Y., Shi, Y. (eds) Data Mining and Big Data. DMBD 2022. Communications in Computer and Information Science, vol 1745. Springer, Singapore. https://doi.org/10.1007/978-981-19-8991-9_20