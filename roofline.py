import re
import pdb
import sys
import getopt

iteration = 5000
mode = 'all'
# filename = '../log/CPU-log_faster_fp32_skx_newmkldnn_mkl2019_profile_epoch2'
# # filename = '../log/GPU-profile_faster_rcnn_X-101_epoch2'
# output_file = '../result/Roofline.csv'

try:
  opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
except getopt.GetoptError:
  print('use -h for help')
  sys.exit(2)
for opt, arg in opts:
  if opt == '-h':
    print('python profile.py -i <inputfile> -o <outputfile> -m mode')
    print(' ')
    print('for example: convert2caffe2.py -i input -o output -m mode')
    print('input:  the input file')
    print('output: the output file')
    sys.exit()

  elif opt in ("-i", "--ifile"):
    filename = arg
  elif opt in ("-o", "--ofile"):
    output_file = arg

batch_size = 32
Gops = 5.495  # /ms
MBms = 115 * 1024 / 1000
acc_mem = 0

p1 = re.compile('.*Running\s+operator\s+(.*)\((\S+)\).*')
p2 = re.compile('.*This\s+operator\s+iteration\s+took\s+(\S+)\s+ms\s+to\s+complete.*')
p3 = re.compile('.*Exit\s+after\s+running\s+(\S+)\s+iterations.*')
p4 = re.compile('.*Output\s+shape:.*,\s+computing\s+in.*seconds,\s+processing.*')
p5 = re.compile('.*INFO\s+test_engine.py:\s+.*\s+(\S+)\s+:\s+(\S+)')
p6 = re.compile('.*INFO\s+infer_simple.py:\s+.*\s+\|\s+(\S+):\s+(\S+)s')
p7 = re.compile('.*INFO\s+infer_simple.py:\s+.*\s+(\S+\s+\S+):\s+(\S+)s')
# p8 = re.compile('.*mkldnn_verbose,exec,+(\S+),jit:uni+.*num:+(\S+),+(\S+),+(\S+)')
# p9 = re.compile('.*mkldnn_verbose,exec,+(\S+),+(\S+),+(\S+),+.*,+(\S+),+(\S+)')
p8 = re.compile('.*mkldnn_verbose,exec,+(\S+),+(\S+),+(\S+),+(.*?),+(.*?),+(\S+),+(\S+)')

cs9 = re.compile('.*mb(\d+)_g(\d+)ic(\d+)oc(\d+)_ih(\d+)oh(\d+)kh(\d+)sh+.*iw(\d+)ow(\d+)kw(\d+)+sw+.*')
ps9 = re.compile('.*mb(\d+)ic(\d+)_ih(\d+)oh(\d+)kh(\d+)sh+.*iw(\d+)ow(\d+)kw(\d+)sw+.*')
fs9 = re.compile('.*mb(\d+)ic(\d+)oc(\d+)')

sameop = False
eopnames = []
operators = {}
roofline = {}
roofline_list = []
oneiter = -1
with open(filename, "r") as f_log:
    for line in f_log.readlines():
        if 'Running net' in line and 'mask_net' not in line:
            oneiter += 1
            if oneiter == 1:
                break
            eopnames = []
        else:
            y1 = p1.match(line)
            if y1 != None and not sameop:
                sameop = True
                if y1.group(1):
                    npos = y1.group(1).index(':')
                    if npos >= 0:
                        extraname = y1.group(1)[0:npos - 1]
                    else:
                        extraname = y1.group(1)
                        opname = extraname + y1.group(2)
                        eopname = extraname + y1.group(2)
                else:
                    opname = y1.group(2)
                    eopname = y1.group(2)
                if not eopnames:
                    eopnames.append("start")
                for j in range(len(eopnames)):
                    if eopname in eopnames:
                        eopname = opname + str(j + 2)
                    else:
                        eopnames.append(eopname)
                        break

            elif y1 != None:
                print('duplicate op line ', line)
            else:
                pass

            y8 = p8.match(line)
            if y8 != None and sameop:
                type = y8.group(1)
                time = float(y8.group(7))
                if mode == 'all':
                    operator = y8.group(2)
                    operator_time = time
                    kernel_shape = y8.group(6)
                    ycs9 = cs9.match(kernel_shape)
                    if ycs9 != None:
                        roofline_list.append(eopname)
                        mb, ic, oc, ih, oh, kh, iw, ow, kw = ycs9.group(1), ycs9.group(3), ycs9.group(4), \
                                                             ycs9.group(5), ycs9.group(6), ycs9.group(7), \
                                                             ycs9.group(8), ycs9.group(9), ycs9.group(10)
                        if eopname not in roofline:
                            # roofline[eopname] = [ic, ih, iw, oc, oh, ow, kh, kw]
                            roofline[eopname] = [float(ic), float(ih), float(iw), float(oc),
                                                 float(oh), float(ow), float(kh), float(kw), float(mb)]
                        last_eopname = eopname
                    yps9 = ps9.match(kernel_shape)
                    if yps9 != None:
                        roofline_list.append(eopname)
                        mb, ic, ih, oh, kh, iw, ow, kw = yps9.group(1), yps9.group(2), yps9.group(3), \
                                                         yps9.group(4), yps9.group(5), yps9.group(6), \
                                                         yps9.group(7), yps9.group(8)
                        oc = ic
                        if eopname not in roofline:
                            roofline[eopname] = [float(ic), float(ih), float(iw), float(oc),
                                                 float(oh), float(ow), float(kh), float(kw), float(mb)]
                        last_eopname = eopname
                    yfs9 = fs9.match(kernel_shape)
                    if yfs9 != None:
                        roofline_list.append(eopname)
                        mb, ic, oc = yfs9.group(1), yfs9.group(2), yfs9.group(3)

                        ih, iw, oh, ow = roofline[last_eopname][1], roofline[last_eopname][2], \
                                         roofline[last_eopname][4], roofline[last_eopname][5]
                        kh, kw = 0, 0
                        if eopname not in roofline:
                            roofline[eopname] = [float(ic), float(ih), float(iw), float(oc),
                                                 float(oh), float(ow), float(kh), float(kw), float(mb)]
                        last_eopname = eopname


            y2 = p2.match(line)
            if y2 != None and sameop:
                sameop = False
                optime = float(y2.group(1))

            elif y2 != None:
                print('can not find op before line ', line)
            else:
                pass

            y3 = p3.match(line)
            if y3 != None:
                iteration = float(y3.group(1))
                break
            else:
                continue
# print(roofline_list)
with open(output_file, "w") as f_roofline:
    f_roofline.write("{0} , {1} , {2} , {3} , {4} , {5} , {6} , {7} , {8} , {9} , "
                     "{10} , {11} , {12} , {13} , {14} , {15} , {16} , {17} , {18} , "
                     "{19} , {20} , {21}, {22}\n".format("op name", "batch_size", "ic", "ih", "iw", "oc",
                                                         "oh", "ow", "kh", "kw", "Gops/ms", "MAC(Gops)",
                                                         "computation(ms)","mem_in(MB)", "mem_out(MB)",
                                                         "wei_in(MB)", "MB/ms", "src_mem(ms)", "acc_mem(ms)",
                                                         "dst_mem(ms)", "wei_mem(ms)", "time(ms)",
                                                         "ratio(comp/memtime"))
    # for key, value in roofline.items():
    for key in roofline_list:
        value = roofline[key]
        batch_size = value[8]
        MAC = batch_size * value[0] * value[3] * value[4] * value[5] * value[6] * value[7] * 2 / 10**9
        computation = MAC / Gops
        mem_in = batch_size * value[0] * value[1] * value[2] / 1024 / 1024
        mem_out = batch_size * value[3] * value[4] * value[5] / 1024 / 1024
        wei_in = value[0] * value[3] * value[6] * value[7] / 1024 / 1024
        src_mem = mem_in / MBms
        acc_mem = 0
        dst_mem = mem_out / MBms
        wei_mem = wei_in / MBms
        time = src_mem + acc_mem + dst_mem + wei_mem
        ratio = computation / time
        if value[6] == 0 or value[7] == 0:
            f_roofline.write("{0} , {1} , {2} , {3} , {4} , {5} , {6} , {7} , {8} , {9} , "
                             "{10} , {11} , {12} , {13} , {14} , {15} , {16} , {17} , {18} , "
                             "{19} , {20} , {21}, {22}\n".format(key, batch_size, value[0], value[1], value[2],
                                                                 value[3], value[4], value[5], "", "",
                                                                 Gops, MAC,computation, mem_in, mem_out, wei_in,
                                                                 MBms, src_mem, acc_mem, dst_mem, wei_mem, time,
                                                                 ratio))
        else:
            f_roofline.write("{0} , {1} , {2} , {3} , {4} , {5} , {6} , {7} , {8} , {9} , "
                             "{10} , {11} , {12} , {13} , {14} , {15} , {16} , {17} , {18} , "
                             "{19} , {20} , {21}, {22}\n".format(key, batch_size, value[0], value[1], value[2],
                                                                 value[3],value[4],value[5], value[6], value[7],
                                                                 Gops, MAC, computation, mem_in, mem_out,wei_in,
                                                                 MBms, src_mem, acc_mem, dst_mem, wei_mem, time,ratio))
    print(output_file, " has been created !")

