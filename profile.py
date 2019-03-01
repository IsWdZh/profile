import re
import pdb
import sys
import getopt
import os
import copy
import collections
# from subprocess import check_output, check_call, CalledProcessError
import subprocess
# from progressbar import *


def pbtxt(pbtxt_file):
    '''
    Get Fusion_type from .pbtxt file.
    :param pbtxt_file:
    :return:Fusion_type for each op. If it does not exist, it is ""
    '''
    name = re.compile('.*name:\s"(\S+)".*')
    type = re.compile('.*type:\s"(\S+)".*')

    layer_count = 0
    getename, gettyname, fusion_exit = False, False, False
    cname = collections.OrderedDict()
    gname = collections.OrderedDict()
    with open(pbtxt_file, "r") as f:
        for line in f.readlines():
            if fusion_exit:
                fusion_type = re.findall(r'.*i:\s(\d+)', line)[0]
                fusion_exit = False

            if "op {" in line:
                layer_count += 1
                # print(layer_count)
                if layer_count == 1:
                    cname["start"], gname["start"] = 0, 0
                # if layer_count > 1 and fusion_type != "":
                if layer_count > 1:
                    for i in range(len(cname)):
                        if tyname in cname:
                            tyname = tyname_bak + str(i + 2)
                        else:
                            cname[tyname] = fusion_type
                            break
                    for j in range(len(gname)):
                        if ename in gname:
                            ename = ename_bak + str(j + 2)
                        else:
                            gname[ename] = fusion_type
                            break
                fusion_type = ""
                getename, gettyname = False, False
                continue

            if not getename and name.match(line) != None:
                ename = re.sub("\d", "", name.findall(line)[0])
                ename_bak = re.sub("\d", "", name.findall(line)[0])
                getename = True
            if not gettyname and type.match(line) != None:
                tyname = type.findall(line)[0]
                tyname_bak = type.findall(line)[0]
                gettyname = True
            if "fusion_type" in line:
                fusion_exit = True

        if layer_count > 1:
            for i in range(len(cname)):
                if tyname in cname:
                    tyname = tyname_bak + str(i + 2)
                else:
                    cname[tyname] = fusion_type
            for j in range(len(gname)):
                if ename in gname:
                    ename = ename_bak + str(j + 2)
                else:
                    gname[ename] = fusion_type
    del cname["start"]
    del gname["start"]
    # print("cname = {}\n".format(cname))
    # print("gname = ", gname)

    return cname, gname

def benchdnn(eopnames, pbtxt, graph, op, kernels):
    '''
    Call the "benchdnn" program to get the time when different kernel run under benchdnn
    :param eopnames:
    :param pbtxt:
    :param graph:
    :return:
    '''
    benchdnn_graph = graph
    benchop = op
    benchkernels = kernels

    env = {"MKLDNN_VERBOSE": "1",
           "LD_PRELOAD": "/home/wendizho/wdzh/lib/libiomp5.so",
           "OMP_NUM_THREADS": "28",
           "KMP_AFFINITY": "proclist=[0-27]",
           "granularity": "thread,explicit"
           }
    for key, value in env.items():
        os.environ[key] = value

    attr = {"1":"--attr=\"post_ops='relu'\"",
            "2":"--attr=\"post_ops='sum'\"",
            "3":"--attr=\"post_ops='sum'\""}

    for each in eopnames[1:]:
        if "Conv" not in each and "conv" not in each:
            benchdnn_graph[each]["benchdnn"] = {"min time": "", "avg time": ""}
            continue
        cmd = ["~/mkl-dnn/build/tests/benchdnn/benchdnn",
               "--conv", "--mode=pc", "--cfg=f32"]
        if each not in pbtxt:
            pbtxt[each] = ""
        if pbtxt[each] != "":
            cmd.append(attr[pbtxt[each]])

        kernel_type = graph[each]["kernel_type"]
        kernel_shape = graph[each]["kernel_shape"]
        if kernel_shape != "":
            cmd.append(kernel_shape)
            try:
                # output = check_output(cmd)
                comd = subprocess.Popen(" ".join(cmd), shell=True, stdout=subprocess.PIPE)
                std = comd.communicate()
                time = re.findall(r'.*min\(ms\):(\d+\.\d+)\savg\(ms\):(\d+\.\d+).*', std[0].decode(encoding='utf-8'))
                if time == []:
                    benchdnn_graph[each]["benchdnn"] = {"min time": "", "avg time": ""}
                    print(each)
                    print(" ".join(cmd))
                    continue
                min_time = float(time[0][0])
                avg_time = float(time[0][1])
                # min_time = 2
                # avg_time = 2
                print(each,"     benchdnn: min time = ", min_time, "   avg time = ", avg_time)
                benchdnn_graph[each]["benchdnn"] = {"min time": min_time, "avg time": avg_time}
                eop = re.split(r'\d+$', each)[0]
                if "benchdnn" in benchop[eop]:
                    benchop[eop]["benchdnn"] = benchop[eop]["benchdnn"] + avg_time
                else:
                    benchop[eop]["benchdnn"] = avg_time

                # benchkernels[kernel_type[0]][kernel_type[1]]["kernel_time"] = avg_time
                # print("benchkernels = ", benchkernels)
                # print("kernel_type[0] = ", kernel_type[0])
                # print("kernel_type[1] = ", kernel_type[1])
                if "benchdnn" in benchkernels[kernel_type[0]][kernel_type[2]]:
                    benchkernels[kernel_type[0]][kernel_type[2]]["benchdnn"] += avg_time
                else:
                    benchkernels[kernel_type[0]][kernel_type[2]]["benchdnn"] = avg_time

            except FileNotFoundError:
                print(cmd[0], " can't find benchdnn, please check it!")
                print(cmd)
        else:
            benchdnn_graph[each]["benchdnn"] = {"min time": "", "avg time": ""}

        for key in benchop.keys():
            if "benchdnn" not in benchop[key]:
                benchop[key]["benchdnn"] = 0.0

        # for key in benchdnn_graph.keys():
        #     if benchdnn_graph[key]["kernel_type"][0] =

        # for key1 in benchkernels.keys():
        #     for key2 in benchkernels[key1].keys():
        #         key = key1 + "({})".format(key2)
        #         if key in


    return benchdnn_graph, benchop, benchkernels

def getinfo(filename):
    p1 = re.compile('.*Running\s+operator\s+(.*)\((\S+)\).*')
    p2 = re.compile('.*This\s+operator\s+iteration\s+took\s+(\S+)\s+ms\s+to\s+complete.*')
    p3 = re.compile('.*Exit\s+after\s+running\s+(\S+)\s+iterations.*')
    p4 = re.compile('.*Output\s+shape:.*,\s+computing\s+in.*seconds,\s+processing.*')
    p5 = re.compile('.*INFO\s+test_engine.py:\s+.*\s+(\S+)\s+:\s+(\S+)')
    p6 = re.compile('.*INFO\s+infer_simple.py:\s+.*\s+\|\s+(\S+):\s+(\S+)s')
    p7 = re.compile('.*INFO\s+infer_simple.py:\s+.*\s+(\S+\s+\S+):\s+(\S+)s')
    p8 = re.compile('.*mkldnn_verbose,exec,+(\S+),+(\S+),+(\S+),+(.*?),+(.*?),+(\S+),+(\S+)')

    graph = {}  # each graph datails
    eopnames = [] # each layer' order in op
    part = {}  # Statistics of the time of each graph
    op = collections.OrderedDict()  # sum of kinds of op time list
    kernels = {}  # each type of kernel's time
    eachop = {}  # each op datails

    count, iter = 0, 0
    sameop = False
    kernel_flag = False

    # widgets = ['Extract log info: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
    #            ' ', ETA()]

    with open(filename, "r") as f_log:
        totalline = f_log.readlines()
        # pbar = ProgressBar(widgets=widgets, maxval=len(totalline)).start()
        for line in totalline:
            # pbar.update(count+1)
            if "begin to run benchmark" in line:
                graph, op, part, kernels, eachop = {}, {}, {}, {}, {}
                iter = 0
                sameop, kernel_flag = False, False
                continue
            if 'Running net' in line and 'mask_net' not in line:
                iter += 1
                eopnames = []
            else:
                y5 = p5.match(line)
                if y5 != None:
                    partname = y5.group(1)
                    partvalue = y5.group(2)
                    if partname in part:
                        part[partname] += float(partvalue)
                    else:
                        part[partname] = float(partvalue)
                    continue

                y6 = p6.match(line)
                if y6 != None:
                    partname = y6.group(1)
                    partvalue = y6.group(2)
                    if partname in part:
                        part[partname] += float(partvalue)
                    else:
                        part[partname] = float(partvalue)
                else:
                    y7 = p7.match(line)
                    if y7 != None:
                        partname = y7.group(1)
                        partvalue = y7.group(2)
                        if partname in part:
                            part[partname] += float(partvalue)
                        else:
                            part[partname] = float(partvalue)

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
                    if opname not in op:
                        op[opname] = {}

                    if not eopnames:
                        eopnames.append("start")
                    for j in range(len(eopnames)):
                        if eopname in eopnames:
                            eopname = opname + str(j + 2)
                        else:
                            eopnames.append(eopname)
                            # if eopname not in graph:
                            #     graph[eopname] = {}
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
                        if type == "reorder":
                            reorder_time = time
                            if "reorder_time" in eachop:
                                eachop["reorder_time"] = eachop["reorder_time"] + reorder_time
                            else:
                                eachop["reorder_time"] = reorder_time

                            # total of the ConvFusion/MaxPool's time
                            if opname in op and "reorder_time" in op[opname]:
                                op[opname]["reorder_time"] = op[opname]["reorder_time"] + reorder_time
                            else:
                                op[opname]["reorder_time"] = reorder_time
                        else:
                            kernel_type = y8.group(2)
                            kernel_shape = y8.group(6)
                            kernel_time = time
                            # print("fwei = ", re.findall(r'.*fwei:(\S+)\s.*', y8.group(4)))
                            try:
                                fwei = re.findall(r'.*fwei:(\S+)\s.*', y8.group(4))[0]
                            except IndexError:
                                fwei = ""

                            if type == "convolution":
                                kernel_flag = True
                                if kernel_type in kernels.keys():
                                    # kernels[kernel_type]["kernel_time"] += kernel_time
                                    if fwei in kernels[kernel_type].keys():
                                        kernels[kernel_type][fwei]["kernel_time"] += kernel_time
                                    else:
                                        kernels[kernel_type][fwei] = {"kernel_time": kernel_time}
                                else:
                                    kernels[kernel_type] = {fwei: {"kernel_time": kernel_time}}

                            if "kernel_time" in eachop:
                                eachop["kernel_time"] = eachop["kernel_time"] + kernel_time
                            else:
                                eachop["kernel_time"] = kernel_time

                            if "kernel_type" not in eachop or "kernel_shape" not in eachop:
                                if fwei == "":
                                    fur_kernel_type = kernel_type
                                else:
                                    fur_kernel_type = kernel_type + "({})".format(fwei)
                                eachop["kernel_type"] = [kernel_type, fur_kernel_type, fwei]
                                eachop["kernel_shape"] = kernel_shape

                            if opname in op and "kernel_time" in op[opname]:
                                op[opname]["kernel_time"] = op[opname]["kernel_time"] + kernel_time
                            else:
                                op[opname]["kernel_time"] = kernel_time

                y2 = p2.match(line)
                if y2 != None and sameop:
                    sameop = False
                    optime = float(y2.group(1))
                    if mode == 'all':
                        eachop["optime"] = optime
                        if "reorder_time" not in eachop:
                            eachop["reorder_time"] = 0.0
                        if "kernel_time" not in eachop:
                            eachop["kernel_time"] = 0.0
                        eachop["framework"] = optime - eachop["reorder_time"] - eachop["kernel_time"]

                    if "optime" in op[opname]:
                        op[opname]["optime"] = op[opname]["optime"] + optime
                    else:
                        op[opname]["optime"] = optime
                    #op[opname]["framework"] = op[opname]["optime"] - op[opname]["reorder_time"] - op[opname]["kernel_time"]

                    if kernel_flag:
                        kernel_flag = False
                        if "optime" in kernels[kernel_type][fwei]:
                            kernels[kernel_type][fwei]["optime"] += optime
                        else:
                            kernels[kernel_type][fwei]["optime"] = optime
                        if "reorder_time" in eachop:
                            if "reorder_time" in kernels[kernel_type][fwei]:
                                kernels[kernel_type][fwei]["reorder_time"] += eachop["reorder_time"]
                            else:
                                kernels[kernel_type][fwei]["reorder_time"] = eachop["reorder_time"]

                    if "kernel_type" not in eachop or "kernel_shape" not in eachop:
                        eachop["kernel_type"] = ["", "", ""]      # [kernel_type, kernel_type+fwei]
                        eachop["kernel_shape"] = ""

                    if eopname not in graph:
                        graph[eopname] = eachop
                    else:
                        graph[eopname]["optime"] += eachop["optime"]
                        graph[eopname]["reorder_time"] += eachop["reorder_time"]
                        graph[eopname]["kernel_time"] += eachop["kernel_time"]
                        graph[eopname]["framework"] += eachop["framework"]
                    #print("eachop = ", eachop)
                    eachop = {}

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

        # pbar.finish()

    for key in op.keys():
        if "reorder_time" not in op[key].keys():
            op[key]["reorder_time"] = 0.0
        if "kernel_time" not in op[key].keys():
            op[key]["kernel_time"] = 0.0
        op[key]["framework"] = op[key]["optime"] - op[key]["reorder_time"] - op[key]["kernel_time"]

    for key1 in kernels:
        for key2 in kernels[key1]:
            if "reorder_time" not in kernels[key1][key2]:
                kernels[key1][key2]["reorder_time"] = 0.0
            if "kernel_time" not in kernels[key1][key2]:
                kernels[key1][key2]["kernel_time"] = 0.0
            kernels[key1][key2]["framework"] = kernels[key1][key2]["optime"] - \
                                               kernels[key1][key2]["reorder_time"] - \
                                               kernels[key1][key2]["kernel_time"]


    # print("graph[\"ConvFusion\"] = ", graph["ConvFusion"])
    # print("graph[\"Conv20\"] = ", graph["Conv20"])
    # print("op = ", op)
    # print("kernels = ", kernels)

    return graph, eopnames, part, op, kernels, iter

def output(output_file, graph, eopnames, part, op, kernels, iteration, pbtxt={}):
    exist = True
    for key in graph:
        if graph[key]["kernel_time"] != 0.0 or graph[key]["reorder_time"] != 0.0:
            exist = True
            break
        else:
            exist = False
    if pbtxt != {}:
        fusiontype = collections.OrderedDict()
        for key, value in pbtxt.items():
            fusiontype[key.title()] = value

    order = {}
    with open(output_file, "w") as f_out:
        f_out.write("each op time list below" + "\n")
        if exist:
            if kernel == "1":
                if pbtxt == {}:
                    f_out.write("{0} , {1} , {2} , {3} , {4} , {5} , {6}\n".format("op name", "optime(ms)",
                                                                                   "reorder(ms)","kernel(ms)",
                                                                                   "framework_time(ms)",
                                                                                   "kernel_type",
                                                                                   "kernel_shape"))
                    titleorder = ["op name", "optime(ms)", "reorder(ms)","kernel(ms)",
                                  "framework_time(ms)", "kernel_type", "kernel_shape"]
                    for each in eopnames[1:]:
                        f_out.write(
                            '{0} , {1} , {2} , {3} , {4} , {5} , {6}\n'.format(each, graph[each]["optime"]/iteration,
                                                                               graph[each]["reorder_time"]/iteration,
                                                                               graph[each]["kernel_time"]/iteration,
                                                                               graph[each]["framework"]/iteration,
                                                                               graph[each]["kernel_type"][1],
                                                                               graph[each]["kernel_shape"]))
                else:
                    f_out.write("{0} , {1} , {2} , {3} , {4} , {5} , {6} , {7} , {8} , {9}\n".format("op name",
                                                                                             "optime(ms)",
                                                                                             "reorder(ms)",
                                                                                             "kernel(ms)",
                                                                                             "framework_time(ms)",
                                                                                             "kernel_type",
                                                                                             "kernel_shape",
                                                                                             "fusion_type",
                                                                                             "benchdnn(ms)",
                                                                                             "ratio(kernel/benchdnn)"))
                    titleorder = ["op name", "optime(ms)", "reorder(ms)", "kernel(ms)",
                                  "framework_time(ms)", "kernel_type", "kernel_shape",
                                  "fusion_type", "benchdnn(ms)", "ratio(kernel/benchdnn)"]
                    for each in eopnames[1:]:
                        if each.title() not in fusiontype:
                            fusiontype[each.title()] = ""
                        ratio = ""
                        if graph[each]["benchdnn"]["avg time"] != "":
                            ratio = (graph[each]["kernel_time"]/iteration)/graph[each]["benchdnn"]["avg time"]
                        f_out.write('{0} , {1} , {2} , {3} , {4} , {5} , {6} , {7} , {8} , {9}\n'.format(each,
                                                                               graph[each]["optime"]/iteration,
                                                                               graph[each]["reorder_time"]/iteration,
                                                                               graph[each]["kernel_time"]/iteration,
                                                                               graph[each]["framework"]/iteration,
                                                                               graph[each]["kernel_type"][1],
                                                                               graph[each]["kernel_shape"],
                                                                               fusiontype[each.title()],
                                                                               graph[each]["benchdnn"]["avg time"],
                                                                                                      ratio))

            else:      # kernel != 1
                if pbtxt == {}:
                    f_out.write("{0} , {1} , {2} , {3} , {4}\n".format("op name", "optime(ms)", "reorder(ms)",
                                                                       "kernel(ms)", "framework_time(ms)"))
                    titleorder = ["op name", "optime(ms)", "reorder(ms)",
                                  "kernel(ms)", "framework_time(ms)"]
                    for each in eopnames[1:]:
                        f_out.write('{0} , {1} , {2} , {3} , {4}\n'.format(each, graph[each]["optime"] / iteration,
                                                                           graph[each]["reorder_time"] / iteration,
                                                                           graph[each]["kernel_time"] / iteration,
                                                                           graph[each]["framework"] / iteration))
                else:
                    f_out.write("{0} , {1} , {2} , {3} , {4} , {5}\n".format("op name", "optime(ms)",
                                                                            "reorder(ms)", "kernel(ms)",
                                                                            "framework_time(ms)", "fusion_type"))
                    titleorder = ["op name", "optime(ms)", "reorder(ms)",
                                  "kernel(ms)", "framework_time(ms)", "fusion_type"]
                    for each in eopnames[1:]:
                        if each.title() not in fusiontype:
                            fusiontype[each.title()] = ""
                        f_out.write('{0} , {1} , {2} , {3} , {4} , {5}\n'.format(each,
                                                                                graph[each]["optime"]/iteration,
                                                                                graph[each]["reorder_time"]/iteration,
                                                                                graph[each]["kernel_time"]/iteration,
                                                                                graph[each]["framework"]/iteration,
                                                                                fusiontype[each.title()]))

        else:       #Exit == False
            if kernel == "1":
                if pbtxt == {}:
                    f_out.write("{0} , {1} , {2} , {3}\n".format("op name", "optime(ms)",
                                                                 "kernel_type", "kernel_shape"))
                    titleorder = ["op name", "optime(ms)", "kernel_type", "kernel_shape"]
                    for each in eopnames[1:]:
                        f_out.write('{0} , {1} , {2} , {3}\n'.format(each, graph[each]["optime"] / iteration,
                                                                     graph[each]["kernel_type"][1],
                                                                     graph[each]["kernel_shape"]))
                else:
                    f_out.write("{0} , {1} , {2} , {3} , {4}\n".format("op name", "optime(ms)",
                                                                      "kernel_type", "kernel_shape",
                                                                      "fusion_type"))
                    titleorder = ["op name", "optime(ms)", "kernel_type", "kernel_shape", "fusion_type"]
                    for each in eopnames[1:]:
                        if each.title() not in fusiontype:
                            fusiontype[each.title()] = ""
                        f_out.write('{0} , {1} , {2} , {3} , {4}\n'.format(each, graph[each]["optime"]/iteration,
                                                                          graph[each]["kernel_type"][1],
                                                                          graph[each]["kernel_shape"],
                                                                          fusiontype[each.title()]))

            else:
                if pbtxt == {}:
                    f_out.write("{0} , {1}\n".format("op name", "optime(ms)"))
                    titleorder = ["op name", "optime(ms)"]
                    for each in eopnames[1:]:
                        f_out.write('{0} , {1}\n'.format(each, graph[each]["optime"] / iteration))
                else:
                    f_out.write("{0} , {1} , {2}\n".format("op name", "optime(ms)", "fusion_type"))
                    titleorder = ["op name", "optime(ms)", "fusion_type"]
                    for each in eopnames[1:]:
                        if each.title() not in fusiontype:
                            fusiontype[each.title()] = ""
                        f_out.write('{0} , {1} , {2}\n'.format(each, graph[each]["optime"] / iteration,
                                                               fusiontype[each.title()]))

        # titlepos = {"total_optime":"optime(ms)", "total_reorder":"reorder(ms)", "total_kernel":"kernel(ms)",
        #             "total_framework":"framework_time(ms)", "total_benchdnn":"benchdnn(ms)"}

        sum_optime_order = []
        f_out.write("\nsum of kinds of op time list below" + "\n")
        total_optime, total_reorder, total_kernel, total_framework, total_benchdnn = 0, 0, 0, 0, 0
        titlepos = {"optime(ms)": ["optime",total_optime], "reorder(ms)": ["reorder_time",total_reorder],
                    "kernel(ms)": ["kernel_time",total_kernel],"framework_time(ms)": ["framework",total_framework],
                    "benchdnn(ms)": ["benchdnn",total_benchdnn]}
        # total_dict = {"total_optime":0, "total_reorder":0, "total_kernel":0,
        #               "total_framework":0, "total_benchdnn":0}
        # oitems = op.items()
        # sorted(oitems)
        # for key, value in oitems:
        for key, value in op.items():
            row = [key]
            for col in titleorder[1:]:
                if not titlepos.get(col):
                    row.append("")
                    continue
                if titlepos.get(col)[0] == "benchdnn":
                    row.append(value[titlepos.get(col)[0]])
                    titlepos.get(col)[1] += value[titlepos.get(col)[0]]
                else:
                    row.append(value[titlepos.get(col)[0]]/iteration)
                    titlepos.get(col)[1] += value[titlepos.get(col)[0]]/iteration
            row = " , ".join('%s' %time for time in row)
            f_out.write(row + "\n")
            sum_optime_order.append(key)
        row = ["total time"]
        for col in titleorder[1:]:
            if not titlepos.get(col):
                row.append("")
                continue
            row.append(titlepos.get(col)[1])
        row = " , ".join('%s' %time for time in row)
        f_out.write(row + "\n")


        f_out.write("\npart time list below" + "\n")
        f_out.write('{0} , {1}\n'.format("preprocessing", part["data1"]*1000/float(iteration)))
        f_out.write('{0} , {1}\n'.format("run", part["run"]*1000/float(iteration)))
        f_out.write('{0} , {1}\n'.format("postprocessing", part["misc_bbox"]*1000/float(iteration)))


        conv_comp_order = []
        f_out.write("\nconv primitive and Comp time\n")
        total_optime, total_reorder, total_kernel, total_framework, total_benchdnn = 0, 0, 0, 0, 0
        titlepos = {"optime(ms)": ["optime", total_optime], "reorder(ms)": ["reorder_time", total_reorder],
                    "kernel(ms)": ["kernel_time", total_kernel], "framework_time(ms)": ["framework", total_framework],
                    "benchdnn(ms)": ["benchdnn", total_benchdnn]}
        for key1 in kernels:
            for key2, value in kernels[key1].items():
                key = key1 + "({})".format(key2)
                row = [key]
                conv_comp_order.append(key)
                for col in titleorder[1:]:
                    if not titlepos.get(col):
                        row.append("")
                        continue
                    if titlepos.get(col)[0] == "benchdnn":
                        row.append(value[titlepos.get(col)[0]])
                        titlepos.get(col)[1] += value[titlepos.get(col)[0]]
                    else:
                        row.append(value[titlepos.get(col)[0]] / iteration)
                        titlepos.get(col)[1] += value[titlepos.get(col)[0]] / iteration
                row = " , ".join('%s' % time for time in row)
                f_out.write(row + "\n")
            sum_optime_order.append(key)
        row = ["total time"]
        for col in titleorder[1:]:
            if not titlepos.get(col):
                row.append("")
                continue
            row.append(titlepos.get(col)[1])
        row = " , ".join('%s' % time for time in row)
        f_out.write(row + "\n")

        #
        # if exist:
        #     # for key, value in kernels.items():
        #     if pb:
        #         for key1 in kernels:
        #             for key2, value in kernels[key1].items():
        #                 key = key1 + "({})".format(key2)
        #                 f_out.write('{0} , {1} , {2} , {3} , {4} , {5}\n'.format(key, value["optime"] / iteration,
        #                                                                    value["reorder_time"] / iteration,
        #                                                                    value["kernel_time"] / iteration,
        #                                                                    value["framework"] / iteration,
        #                                                                          value["benchdnn"]))
        #                 total_optime += value["optime"] / iteration
        #                 total_reorder += value["reorder_time"] / iteration
        #                 total_kernel += value["kernel_time"] / iteration
        #                 total_framework += value["framework"] / iteration
        #                 total_benchdnn += value["benchdnn"]
        #                 conv_comp_order.append(key)
        #
        #         f_out.write("total conv , {0} , {1} , {2} , {3} , {4}".format(total_optime, total_reorder,
        #                                                                 total_kernel, total_framework, total_benchdnn))
        #     else:
        #         for key1 in kernels:
        #             for key2, value in kernels[key1].items():
        #                 key = key1 + "({})".format(key2)
        #                 f_out.write('{0} , {1} , {2} , {3} , {4}\n'.format(key, value["optime"] / iteration,
        #                                                                          value["reorder_time"] / iteration,
        #                                                                          value["kernel_time"] / iteration,
        #                                                                          value["framework"] / iteration))
        #                 total_optime += value["optime"] / iteration
        #                 total_reorder += value["reorder_time"] / iteration
        #                 total_kernel += value["kernel_time"] / iteration
        #                 total_framework += value["framework"] / iteration
        #                 total_benchdnn += value["benchdnn"]
        #                 conv_comp_order.append(key)
        #
        #         f_out.write("total conv , {0} , {1} , {2} , {3}".format(total_optime, total_reorder,
        #                                                                       total_kernel, total_framework))
        # else:
        #     # for key, value in kernels.items():
        #     for key1 in kernels:
        #         for key2, value in kernels[key1].items():
        #             key = key1 + "({})".format(key2)
        #             f_out.write('{0} , {1}\n'.format(key, value["optime"] / iteration))
        #             total_optime += value["optime"] / iteration
        #             conv_comp_order.append(key)
        #     f_out.write("total conv , {0}".format(total_optime))


        order["sum_optime_order"] = sum_optime_order
        order["conv_comp_order"] = conv_comp_order

    print(os.path.split(output_file)[-1], " has been created !")
    return order, output_file

def summary(cfilename, gfilename):
    cpu = {}
    gpu = {}
    cpu_key_list = []
    gpu_key_list = []
    # cpu = collections.OrderedDict()
    # gpu = collections.OrderedDict()
    with open(cfilename, "r") as fc, open(gfilename, "r") as fg:
        for line in fc.readlines():
            line = line.split(" , ")
            if line[-1] == "\n" or line[-1][-1] == "\n":
                line[-1] = line[-1][:-1]
            # print(line)
            try:
                if kernel == "1":
                    if len(line) > 1 and isinstance(float(line[1]), float):
                        cpu_key_list.append(line[0])
                        if pb:
                            # line[8] maybe "" ,such as Maxpool
                            cpu[line[0]] = {"optime": float(line[1]), "reorder_time": float(line[2]),
                                            "kernel_time": float(line[3]), "framework": float(line[4]),
                                            "kernel_type": line[5], "kernel_shape": line[6],
                                            "fusion_type": line[7], "benchdnn":line[8]}
                        # try:
                        # except IndexError:
                        else:
                            cpu[line[0]] = {"optime":float(line[1]), "reorder_time":float(line[2]),
                                            "kernel_time":float(line[3]), "framework":float(line[4]),
                                            "kernel_type":line[5], "kernel_shape":line[6]}

                    else:
                        if "sum of kinds of op time" in line[0]:
                            break
                        continue
                else:
                    if len(line) > 1 and isinstance(float(line[1]), float):
                        cpu_key_list.append(line[0])
                        # cpu[line[0]] = [float(line[1]), float(line[2]), float(line[3]),
                        #                 float(line[4])]
                        cpu[line[0]] = {"optime": float(line[1]), "reorder_time": float(line[2]),
                                        "kernel_time": float(line[3]), "framework": float(line[4])}
                    else:
                        if "sum of kinds of op time" in line[0]:
                            break
                        continue
            except ValueError:
                continue

            else:
                if "sum of kinds of op time" in line[0]:
                    break
                continue


        for line in fg.readlines():
            line = line.split(" , ")
            if line[-1] == "\n" or line[-1][-1] == "\n":
                line[-1] = line[-1][:-1]
            try:
                if kernel == "1":
                    if len(line) > 1 and isinstance(float(line[1]), float):
                        gpu_key_list.append(line[0])
                        # gpu[line[0]] = float(line[1])
                        # gpu[line[0]] = [float(line[1]), line[2], line[3]]
                        gpu[line[0]] = {"optime":float(line[1]), "kernel_type":line[2], "kernel_shape":line[3]}
                    else:
                        if "sum of kinds of op time" in line[0]:
                            break
                        continue
                else:
                    if len(line) > 1 and isinstance(float(line[1]), float):
                        gpu_key_list.append(line[0])
                        gpu[line[0]] = {"optime":float(line[1])}
                    else:
                        if "sum of kinds of op time" in line[0]:
                            break
                        continue
            except ValueError:
                continue

    pos = 0
    new_op = {}
    new_gpu = collections.OrderedDict()
    kernels_to_gpu = {}


    for i in range(len(cpu_key_list) - 1):
        try:
            if "ConvFusion" in cpu_key_list[i] and "Conv" in cpu_key_list[i + 1]:
                if "Conv" not in gpu_key_list[pos]:
                    exit("please check the program in {}".format(gpu_key_list[pos]))
                # kernels_to_gpu[gpu_key_list[pos]] = [cpu[cpu_key_list[i]]["kernel_type"],
                #                                      cpu[cpu_key_list[i]]["kernel_shape"]]
                if pb:
                    kernels_to_gpu[gpu_key_list[pos]] = {"kernel_type": cpu[cpu_key_list[i]]["kernel_type"],
                                                         "kernel_shape": cpu[cpu_key_list[i]]["kernel_shape"],
                                                         "benchdnn": cpu[cpu_key_list[i]]["benchdnn"]}
                else:
                    kernels_to_gpu[gpu_key_list[pos]] = {"kernel_type":cpu[cpu_key_list[i]]["kernel_type"],
                                                         "kernel_shape":cpu[cpu_key_list[i]]["kernel_shape"]}

                convname = gpu_key_list[pos]
                convtime = gpu[convname]["optime"]
                newtime = gpu[gpu_key_list[pos]]["optime"]   # op time
                pos += 1
                while "Conv" not in gpu_key_list[pos]:
                    newtime += gpu[gpu_key_list[pos]]["optime"]
                    pos += 1
                # newtime:th sum of Merged layer time    convname:gpu layer name  convtime: only conv layer's time
                new_gpu[cpu_key_list[i]] = [newtime, convname, convtime]
            elif "ConvFusion" in cpu_key_list[i] and "Conv" not in cpu_key_list[i + 1]:
                if "Conv" not in gpu_key_list[pos]:
                    exit("please check the program in {}".format(gpu_key_list[pos]))
                # kernels_to_gpu[gpu_key_list[pos]] = [cpu[cpu_key_list[i]]["kernel_type"],
                #                                      cpu[cpu_key_list[i]]["kernel_shape"]]
                if pb:
                    kernels_to_gpu[gpu_key_list[pos]] = {"kernel_type": cpu[cpu_key_list[i]]["kernel_type"],
                                                         "kernel_shape": cpu[cpu_key_list[i]]["kernel_shape"],
                                                         "benchdnn": cpu[cpu_key_list[i]]["benchdnn"]}
                else:
                    kernels_to_gpu[gpu_key_list[pos]] = {"kernel_type":cpu[cpu_key_list[i]]["kernel_type"],
                                                         "kernel_shape":cpu[cpu_key_list[i]]["kernel_shape"]}

                convname = gpu_key_list[pos]
                convtime = gpu[convname]["optime"]
                newtime = gpu[gpu_key_list[pos]]["optime"]
                pos += 1
                while gpu_key_list[pos] != cpu_key_list[i + 1]:
                    newtime += gpu[gpu_key_list[pos]]["optime"]
                    pos += 1
                new_gpu[cpu_key_list[i]] = [newtime, convname, convtime]
            elif "Conv" in cpu_key_list[i] and "ConvFusion" not in cpu_key_list[i]:
                if "Conv" not in gpu_key_list[pos]:
                    exit("please check the program in {}".format(gpu_key_list[pos]))
                # kernels_to_gpu[gpu_key_list[pos]] = [cpu[cpu_key_list[i]]["kernel_type"],
                #                                      cpu[cpu_key_list[i]]["kernel_shape"]]
                if pb:
                    kernels_to_gpu[gpu_key_list[pos]] = {"kernel_type": cpu[cpu_key_list[i]]["kernel_type"],
                                                         "kernel_shape": cpu[cpu_key_list[i]]["kernel_shape"],
                                                         "benchdnn": cpu[cpu_key_list[i]]["benchdnn"]}
                else:
                    kernels_to_gpu[gpu_key_list[pos]] = {"kernel_type":cpu[cpu_key_list[i]]["kernel_type"],
                                                         "kernel_shape":cpu[cpu_key_list[i]]["kernel_shape"]}

                convname = gpu_key_list[pos]
                convtime = gpu[convname]["optime"]
                newtime = gpu[gpu_key_list[pos]]["optime"]
                pos += 1
                if "AffineChannel" in gpu_key_list[pos]:
                    newtime += gpu[gpu_key_list[pos]]["optime"]
                    pos += 1
                new_gpu[cpu_key_list[i]] = [newtime, convname, convtime]
            else:
                if pb:
                    if "benchdnn" not in cpu[cpu_key_list[i]]:
                        cpu[cpu_key_list[i]]["benchdnn"] = ""
                    kernels_to_gpu[gpu_key_list[pos]] = {"kernel_type": cpu[cpu_key_list[i]]["kernel_type"],
                                                         "kernel_shape": cpu[cpu_key_list[i]]["kernel_shape"],
                                                         "benchdnn": cpu[cpu_key_list[i]]["benchdnn"]}
                else:
                    kernels_to_gpu[gpu_key_list[pos]] = {"kernel_type":cpu[cpu_key_list[i]]["kernel_type"],
                                                         "kernel_shape":cpu[cpu_key_list[i]]["kernel_shape"]}

                newtime = gpu[cpu_key_list[i]]["optime"]
                pos += 1
                new_gpu[cpu_key_list[i]] = [newtime, newtime, newtime]
        except IndexError:
            new_gpu[cpu_key_list[i]] = [newtime, newtime, newtime]
            break

    if pb:
        if "benchdnn" not in cpu[cpu_key_list[i]]:
            cpu[cpu_key_list[i]]["benchdnn"] = ""
        kernels_to_gpu[gpu_key_list[-1]] = {"kernel_type": cpu[cpu_key_list[-1]]["kernel_type"],
                                             "kernel_shape": cpu[cpu_key_list[-1]]["kernel_shape"],
                                             "benchdnn": cpu[cpu_key_list[-1]]["benchdnn"]}
    else:
        kernels_to_gpu[gpu_key_list[-1]] = {"kernel_type": cpu[cpu_key_list[-1]]["kernel_type"],
                                             "kernel_shape": cpu[cpu_key_list[-1]]["kernel_shape"]}




    new_gpu[cpu_key_list[-1]] = [gpu[gpu_key_list[-1]]["optime"],
                                 gpu[gpu_key_list[-1]]["optime"],
                                 gpu[gpu_key_list[-1]]["optime"]]
    for key in new_gpu:
        opn = re.split('\d+$', key)[0]
        if opn in new_op:
            new_op[opn] += new_gpu[key][0]
        else:
            new_op[opn] = new_gpu[key][0]

    return cpu, new_gpu, kernels_to_gpu, new_op

def sumfile(cpu, sum_file, new_gpu):
    with open(sum_file, "w") as outfile:
        outfile.write("GPU compared with CPU\n")
        outfile.write("{0} , {1} , {2} , {3} , {4}".format("op_name", "cpu_optime", "gpu_conv_fuse",
                                                     "conv_only", "Ratio(gpu/cpu)\n"))
        for key, value in new_gpu.items():
            outfile.write("{0} , {1} , {2} , {3} , {4}\n".format(key, cpu[key]["optime"], value[0],
                                                           value[2], round(value[0] / cpu[key]["optime"], 4)))

def newgpu(oldgpu, new_gpu_file, kernels_to_gpu, order, gpu_op, gpu_part):
    conv_breakdown = {}
    end = False
    new_gpu_op = gpu_op
    with open(new_gpu_file, "w") as newoutfile, open(oldgpu, "r") as fg:
        for line in fg.readlines():
            if "sum of kinds of op" in line:
                # end = True
                newoutfile.write(line)
                break

            if "sum of kinds of op" in line:
                # end = True
                newoutfile.write(line)
                break
            else:
                spline = line.split(" , ")
                if spline[-1] == "\n" or spline[-1][-1] == "\n":
                    spline[-1] = spline[-1][:-1]
                if spline[0] == "op name":
                    if pb:
                        spline.append("benchdnn(ms)")
                    line = " , ".join(spline)
                    newoutfile.write(line + "\n")
                    titleorder = spline
                    continue

                if pb:
                    if spline[0] in kernels_to_gpu:
                        print(spline)
                        if "benchdnn" in kernels_to_gpu[spline[0]]:
                            sumopname = re.split(r'\d+$', spline[0])[0]
                            if kernels_to_gpu[spline[0]]["benchdnn"] != "":
                                if "benchdnn" in new_gpu_op[sumopname].keys():
                                    new_gpu_op[sumopname]["benchdnn"] += float(kernels_to_gpu[spline[0]]["benchdnn"])
                                else:
                                    new_gpu_op[sumopname]["benchdnn"] = float(kernels_to_gpu[spline[0]]["benchdnn"])
                        spline[-3] = kernels_to_gpu[spline[0]]["kernel_type"]
                        spline[-2] = kernels_to_gpu[spline[0]]["kernel_shape"]
                        if spline[-3] != "":
                            if spline[-3] not in conv_breakdown.keys():
                                conv_breakdown[spline[-3]] = {"optime": float(spline[-4])}
                                # conv_breakdown[spline[-3]] = float(spline[-4])
                            else:
                                conv_breakdown[spline[-3]]["optime"] += float(spline[-4])
                                # conv_breakdown[spline[-3]] += float(spline[-4])

                            if kernels_to_gpu[spline[0]]["benchdnn"] != "":
                                if "benchdnn" in conv_breakdown[spline[-3]].keys():
                                    conv_breakdown[spline[-3]]["benchdnn"] += float(kernels_to_gpu[spline[0]]["benchdnn"])
                                else:
                                    conv_breakdown[spline[-3]]["benchdnn"] = float(kernels_to_gpu[spline[0]]["benchdnn"])

                        spline.append(kernels_to_gpu[spline[0]]["benchdnn"])
                        line = " , ".join("%s"%i for i in spline)
                        newoutfile.write(line + "\n")
                    else:
                        newoutfile.write(line)
                else:
                    if spline[0] in kernels_to_gpu:
                        spline[-2] = kernels_to_gpu[spline[0]]["kernel_type"]
                        spline[-1] = kernels_to_gpu[spline[0]]["kernel_shape"]
                        if spline[-2] != "":
                            if spline[-2] not in conv_breakdown.keys():
                                conv_breakdown[spline[-2]] = {"optime":float(spline[-3])}
                            else:
                                conv_breakdown[spline[-2]]["optime"] += float(spline[-3])
                        line = " , ".join(spline)
                        newoutfile.write(line + "\n")
                    else:
                        newoutfile.write(line)

        total_optime, total_reorder, total_kernel, total_framework, total_benchdnn = 0, 0, 0, 0, 0
        titlepos = {"optime(ms)": ["optime", total_optime],
                    "reorder(ms)": ["reorder_time", total_reorder],
                    "kernel(ms)": ["kernel_time", total_kernel],
                    "framework_time(ms)": ["framework", total_framework],
                    "benchdnn(ms)": ["benchdnn", total_benchdnn]}


        for key in new_gpu_op.keys():
            if "benchdnn" not in new_gpu_op[key]:
                new_gpu_op[key]["benchdnn"] = 0.0

        print("new_gpu_op = ", new_gpu_op)
        for key, value in new_gpu_op.items():
            row = [key]
            for col in titleorder[1:]:
                if not titlepos.get(col):
                    row.append("")
                    continue
                if not value.get(titlepos.get(col)[0]):
                    row.append("")
                    continue
                if value[titlepos.get(col)[0]] == "":
                    value[titlepos.get(col)[0]] = 0.0

                # print("titlepos.get(col)[0] = ", titlepos.get(col)[0])
                # print("titlepos.get(col)[1] = ", titlepos.get(col)[1])
                # print("value[titlepos.get(col)[0]] = ", value[titlepos.get(col)[0]])

                if titlepos.get(col)[0] == "benchdnn":
                    row.append(value[titlepos.get(col)[0]])
                    titlepos.get(col)[1] += value[titlepos.get(col)[0]]
                else:
                    row.append(value[titlepos.get(col)[0]]/iteration)
                    titlepos.get(col)[1] += value[titlepos.get(col)[0]]/iteration
            row = " , ".join('%s' % time for time in row)
            newoutfile.write(row + "\n")
            continue
        row = ["total time"]
        for col in titleorder[1:]:
            if not titlepos.get(col):
                row.append("")
                continue
            row.append(titlepos.get(col)[1])
        row = " , ".join('%s' % time for time in row)
        newoutfile.write(row + "\n")



        newoutfile.write("\npart time list below" + "\n")
        newoutfile.write('{0} , {1}\n'.format("preprocessing", gpu_part["data1"] * 1000 / float(iteration)))
        newoutfile.write('{0} , {1}\n'.format("run", gpu_part["run"] * 1000 / float(iteration)))
        newoutfile.write('{0} , {1}\n'.format("postprocessing", gpu_part["misc_bbox"] * 1000 / float(iteration)))

        total_optime, total_reorder, total_kernel, total_framework, total_benchdnn = 0, 0, 0, 0, 0
        titlepos = {"optime(ms)": ["optime", total_optime],
                    "reorder(ms)": ["reorder_time", total_reorder],
                    "kernel(ms)": ["kernel_time", total_kernel],
                    "framework_time(ms)": ["framework", total_framework],
                    "benchdnn(ms)": ["benchdnn", total_benchdnn]}
        newoutfile.write("\nconv primitive and Comp time\n")
        for kname in order["conv_comp_order"]:
            row = [kname]
            for col in titleorder[1:]:
                if not titlepos.get(col):
                    row.append("")
                    continue
                if not conv_breakdown[kname].get(titlepos.get(col)[0]):
                    row.append("")
                    continue
                row.append(conv_breakdown[kname][titlepos.get(col)[0]])
                titlepos.get(col)[1] += conv_breakdown[kname][titlepos.get(col)[0]]

            row = " , ".join('%s' % time for time in row)
            newoutfile.write(row + "\n")
            continue
        row = ["total time"]
        for col in titleorder[1:]:
            if not titlepos.get(col):
                row.append("")
                continue
            row.append(titlepos.get(col)[1])
        row = " , ".join('%s' % time for time in row)
        newoutfile.write(row + "\n")



# def progressbar():
#     widgets = ['Extract log info: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
#                ' ', ETA()]
#     pbar = ProgressBar(widgets=widgets, maxval=total).start()
#     pbar.update(10 + 1)
#     pbar.finish()


if __name__ == '__main__':
    mode = 'all'
    kernel, sum = "1", "1"
    pb = False

    try:
        # opts, args = getopt.getopt(sys.argv[1:],"hc:g:i:p:q:o:s:n:m:k:", ["cpu=","gpu=","ifile=","cpuout=","gpuout=",
        #                             "ofile=","sum=","iter=","mode=","kernel="])
        opts, args = getopt.getopt(sys.argv[2:], "hg:i:p:o:s:k:",
                                   ["gpu=", "ifile=", "pbtxt=", "ofolder=", "sum=", "kernel="])
    except getopt.GetoptError:
        print('use -h for help')
        sys.exit(2)
    cpu_file = sys.argv[1]
    if not os.path.isfile(cpu_file):
        exit("The first parameter should be the log file running on the CPU!")
    for opt, arg in opts:
        if opt == '-h':
            print('python profile.py <cpu_log> -g <gpu_log> -o <output> -p <pbtxt>')
            print(' ')
            print('for example: profile.py CPU_log -g GPU_log -o output_folder -p pbtxt')
            print('If you don\'t want kernel info, you can set -k <kernel> to 0' )
            print('If you don\'t want compared info, you can set -s <sum> to 0')
            print('The first parameter is GPU\'log, don\'t need options')
            print('output: the output folder')
            print('sum: gpu compared with cpu')
            print('kernel: 1 or 0')
            sys.exit()

        elif opt in ("-g", "--gpu"):
            gpu_file = arg
        elif opt in ("-i", "--ifile"):
            filename = arg
        elif opt in ("-p", "--pbtxt"):
            pbtxt_file = arg
        elif opt in ("-o", "--ofolder"):
            output_folder = arg
        elif opt in ("-s", "--sum"):
            sum = arg
        elif opt in ("-k", "--kernel"):
            kernel = arg

    if not os.path.isdir(output_folder):
        exit("The output path is wrong, it shoule be a folder!")

    if "pbtxt_file" in locals().keys():
        cpu_fusion_type , gpu_fusion_type = pbtxt(pbtxt_file)
        pb = True

    if "gpu_file" in locals().keys():
        cpu_out = os.path.join(output_folder, os.path.split(cpu_file)[-1] + "_result.csv")
        gpu_out = os.path.join(output_folder, os.path.split(gpu_file)[-1]+"_result.csv")
        gpu_out_bak = copy.deepcopy(gpu_out)

        graph, eopnames, part, op, kernels, iteration = getinfo(cpu_file)
        if pb:
            if kernel == "1":
                graph, op, kernels = benchdnn(eopnames, cpu_fusion_type, graph, op, kernels)
            cpu_order, cpu_out_file = output(cpu_out, graph, eopnames, part, op, kernels, iteration, pbtxt=cpu_fusion_type)
        else:
            cpu_order, cpu_out_file = output(cpu_out, graph, eopnames, part, op, kernels, iteration)

        # cpu_out_file = cpu_out

        graph, eopnames, gpu_part, gpu_op, kernels, iteration = getinfo(gpu_file)
        if pb:
            # if kernel == "1":
            #     graph = benchdnn(eopnames, gpu_fusion_type, graph)
            gpu_order, gpu_out_file = output(gpu_out, graph, eopnames, gpu_part, gpu_op,
                                             kernels, iteration, pbtxt=gpu_fusion_type)
        else:
            gpu_order, gpu_out_file = output(gpu_out, graph, eopnames, gpu_part, gpu_op, kernels, iteration)
        cpu, new_gpu, kernels_to_gpu, new_op = summary(cpu_out_file, gpu_out_file)
        # if "sum_file" in locals().keys():
        if sum == "1":
            sum_file = os.path.join(output_folder, "compared_result.csv")
            sumfile(cpu, sum_file, new_gpu)
        if kernel == "1":
            path = os.path.split(gpu_out_file)[0]
            suffix = os.path.splitext(gpu_out_file)[-1]
            oldgpu = os.path.join(path,"oldgputemp"+suffix)
            os.rename(gpu_out_file, oldgpu)
            newgpu(oldgpu, gpu_out_bak, kernels_to_gpu, cpu_order, gpu_op, gpu_part)
            os.remove(oldgpu)
        else:
            exit()

    elif "output_folder" in locals().keys():
        cpu_out = os.path.join(output_folder, os.path.split(cpu_file)[-1] + "_result.csv")
        graph, eopnames, part, op, kernels, iteration = getinfo(cpu_file)
        if pb:
            if kernel == "1":
                graph = benchdnn(eopnames, cpu_fusion_type, graph)
            cpu_out_file = output(cpu_out, graph, eopnames, part, op, kernels, iteration, pbtxt=cpu_fusion_type)
        else:
            cpu_out_file = output(cpu_out, graph, eopnames, part, op, kernels, iteration)
        exit()
    else:
        exit("Wrong command Line, please use -h for help.")



