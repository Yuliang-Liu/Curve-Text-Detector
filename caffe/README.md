# Caffe branch for R-FCN

This is a branch of Caffe supporting [**R-FCN**](https://github.com/daijifeng001/R-FCN), which has been tested under Windows (Windows 7, 8, Server 2012 R2) and Linux (Ubuntu 14.04).

## Linux Setup

### Pre-Build Steps

Copy `Makefile.config.example` to `Makefile.config`

We need to modify Makefile.config to specify some software PATHS, you may view my [Makefile.config](https://1drv.ms/u/s!Am-5JzdW2XHzhc43M2A9CwV-4O6c8w) for reference.

### CUDA
Download `CUDA Toolkit 7.5` [from nVidia website](https://developer.nvidia.com/cuda-toolkit).

### Matlab
Uncomment ```MATLAB_DIR``` and set ```MATLAB_DIR``` accordingly to build Caffe Matlab wrapper. Matlab 2014a and later versions are supported.

### cuDNN (optional)
For cuDNN acceleration using NVIDIA’s proprietary cuDNN software, uncomment the ```USE_CUDNN := 1``` switch in Makefile.config. cuDNN is sometimes but not always faster than Caffe’s GPU acceleration.

Download `cuDNN v3` or `cuDNN v4` [from nVidia website](https://developer.nvidia.com/cudnn). And unpack downloaded zip to $CUDA_PATH (It typically would be /usr/local/cuda/include and /usr/local/cuda/lib64).

### Build

Simply type
```
make -j8 && make matcaffe
```

## Windows Setup
**Requirements**: Visual Studio 2013

### Pre-Build Steps
Copy `.\windows\CommonSettings.props.example` to `.\windows\CommonSettings.props`

3rd party dependencies required by Caffe are automatically resolved via NuGet.

### CUDA
Download `CUDA Toolkit 7.5` [from nVidia website](https://developer.nvidia.com/cuda-toolkit).

### Matlab
Set `MatlabSupport` to `true` and `MatlabDir` to the root of your Matlab installation in `.\windows\CommonSettings.props` to build Caffe Matlab wrapper. Matlab 2014a and later versions are supported.

### cuDNN (optional)
Download `cuDNN v3` or `cuDNN v4` [from nVidia website](https://developer.nvidia.com/cudnn).
Unpack downloaded zip to %CUDA_PATH% (environment variable set by CUDA installer).
Alternatively, you can unpack zip to any location and set `CuDnnPath` to point to this location in `.\windows\CommonSettings.props`.
`CuDnnPath` defined in `.\windows\CommonSettings.props`.
By default, cuDNN is not enabled. You can enable cuDNN by setting `UseCuDNN` to `true` in the property file.

### Build
Now, you should be able to build `.\windows\Caffe.sln`

#### Remark
After you have built solution with Matlab support, copy all files in .\Build\x64\Release to R-FCN\external\caffe\matlab\caffe_rfcn.

## Further Details

Refer to the BVLC/caffe master branch README for all other details such as license, citation, and so on.

