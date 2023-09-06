from scheme import iwt_dedup
from liu_fuzzy import fuzzy_dedup

import os
import sys

ds_path = './bill/'
out_path = './result/perf/'
# hanmming distance threshold
ts = [1, 2, 3]
# file count
fcnt = 500

# read all files from path
# return files' path
def readFiles(path : str, cnt : int):
    files = []
    for f in os.listdir(path):
        # ignore hidden files
        if f.startswith('.'):
            continue
        fp = os.path.join(path, f)
        if os.path.isfile(fp):
            files.append(fp)
        if len(files) == cnt:
            break
    return files

def test_time_scheme(fps : list):
    serv = iwt_dedup()
    for fp in fps:
        req_hash = serv.upload(fp)
        serv.download(req_hash)
    with open(out_path + 'time_scheme.csv', 'w') as f:
        f.write('up_time,down_time,diff_dedup_time\n')
        f.write('%f,%f,%f\n' % (serv.up_time, serv.down_time, serv.diff_dedup_time))

def test_time_liu(fps : list):
    serv = fuzzy_dedup()
    for fp in fps:
        serv.upload(fp)
        serv.download(fp)
    with open(out_path + 'time_liu.csv', 'w') as f:
        f.write('up_time,down_time,diff_dedup_time\n')
        f.write('%f,%f,%f\n' % (serv.up_time, serv.down_time, serv.diff_dedup_time))

def test_dedup_scheme(fps : list):
    serv = iwt_dedup()
    up_cnt = 0
    f = open(out_path + 'dedup_scheme.csv', 'w')
    f.write('fnum, storage overhead,\n')
    for fp in fps:
        req_hash = serv.upload(fp)
        serv.download(req_hash)
        up_cnt += 1
        if up_cnt >= 100 and up_cnt % 10 == 0:
            strg_oh = 0
            for k in serv.base_dic:
                strg_oh += len(serv.base_dic[k][0])
            for k in serv.diff_dic:
                strg_oh += len(serv.diff_dic[k].tobytes())
            f.write('%d,%d\n' % (up_cnt, strg_oh))
    print(len(serv.base_dic))
    print(len(serv.diff_dic))

def test_dedup_liu(fps : list):
    serv = fuzzy_dedup()
    up_cnt = 0
    f = open(out_path + 'dedup_liu.csv', 'w')
    f.write('fnum, storage overhead,\n')
    for fp in fps:
        serv.upload(fp)
        serv.download(fp)
        up_cnt += 1
        if up_cnt >= 100 and up_cnt % 10 == 0:
            f.write('%d,%d\n' % (up_cnt, serv.base_strg_oh + serv.diff_strg_oh))
    f.close()

if __name__ == '__main__':
    fps = readFiles(ds_path, fcnt)
    # test time
    if sys.argv[1] == 'time' or sys.argv[1] == 'all':
        if sys.argv[2] == 'scheme' or sys.argv[2] == 'all':
            test_time_scheme(fps)
        if sys.argv[2] == 'liu' or sys.argv[2] == 'all':
            test_time_liu(fps)

    # test deduplication rate
    if sys.argv[1] == 'dedup' or sys.argv[1] == 'all':
        if sys.argv[2] == 'scheme' or sys.argv[2] == 'all':
            test_dedup_scheme(fps)
        if sys.argv[2] == 'liu' or sys.argv[2] == 'all':
            test_dedup_liu(fps)
        for t in ts:
            pass