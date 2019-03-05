# README


## Profile

### 1. Usage
Usage(main driver):

> In `dl_framework-dl_framework-intel_caffe2_tools` folder

```
python3 ./tools/profile_tool/profile.py [cpu_log_fiile] [-g gpu_log_file] [-o output_folder] [-b pbtxt] [benchdnn_path]
```
where:
- `cpu_log_file` is cpu's log file(without short options)
- `gpu_log_file` is gpu's log file
- `output_folder` is output path(it should be a folder)
- `pbtxt` is Caffe2's pb file(If this is set, the path to benchdnn should be added at the end)

The output statistical file's name is the input log file name + "result.csv"
The cpu and gpu comparison file's name is "compared\_result.csv"



### 2. Command parameters
#### 2.1 log info
The first parameter is CPU'log file, then:
`-g`:GPU's log file
`-o`:output path(It should be a folder)

#### 2.2 benchdnn info:
`-b`:Caffe2 pbtxt file(if you don't want the benchdnn info,you don't need it.
                        Otherwise you need to set the benchdnn path)
**Note:**
If you enter this, you can set the environment variable before the program runs. If not set, the default environment variable will be used. The default environment variable will prompt when the program is running.

#### 2.3 summary info:
`-s`:if you don't want Comparison of cpu and gpu,you can set `-s 0`, default `"1"`

#### 2.4 kernel info:
`-k`:if you don't want kernel info(include kernel type and shape), set `-k 0`, default `"1"`



## roofline

### Usage

> In `dl_framework-dl_framework-intel_caffe2_tools` folder

```
python3 ./tools/profile_tool/roofline.py -i <input_file> -o <output_folder>
```
where:
- `input_file`: input log file
- `output_folder`: Output file storage directory(It should be a folder)

The output roofline file's name is the input log file name + "roofline.csv"
