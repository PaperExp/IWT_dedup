from scheme import iwt_dedup
from iwt import image_to_iwt, iwt_to_image
from phash import phash

import hashlib

ds_path = './cmp_img/'
out_path = './result/secu/'

def get_phash(img_path : str):
    approximation, (horizontal, vertical, diagonal) = image_to_iwt(img_path)
    return phash(approximation)

def get_hash(img_path : str):
    approximation, (horizontal, vertical, diagonal) = image_to_iwt(img_path)
    return hashlib.sha256(approximation).digest()

def write_phash(phash : int, out_f : str):
    print(phash)
    cnt = 0
    with open(out_path + out_f, 'w') as f:
        while cnt < 8:
            f.write('%d,\n' % (phash & 0xff))
            phash >>= 8
            cnt += 1

def write_hash(hash : bytes, out_f : str):
    with open(out_path + out_f, 'w') as f:
        for byte in hash:
            f.write('%d,\n' % byte)

def test_security():
    ph = get_hash(ds_path + 'medical-record-org1.jpeg')
    write_hash(ph, 'img1_base.csv')

    ph = get_hash(ds_path + 'medical-record-org2.jpeg')
    write_hash(ph, 'img2_base.csv')

if __name__ == '__main__':
    test_security()