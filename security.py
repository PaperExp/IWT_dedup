from scheme import iwt_dedup
from iwt import image_to_iwt, iwt_to_image
from phash import phash

ds_path = './cmp_img/'
out_path = './result/secu/'

def get_phash(img_path : str):
    approximation, (horizontal, vertical, diagonal) = image_to_iwt(img_path)
    return phash(approximation)

def write_phash(phash : int, out_f : str):
    print(phash)
    cnt = 0
    with open(out_path + out_f, 'w') as f:
        while cnt < 8:
            f.write('%d,\n' % (phash & 0xff))
            phash >>= 8
            cnt += 1

def test_security():
    ph = get_phash(ds_path + 'medical-record-org1.jpeg')
    write_phash(ph, 'img1_base.csv')

    ph = get_phash(ds_path + 'medical-record-org2.jpeg')
    write_phash(ph, 'img2_base.csv')

if __name__ == '__main__':
    test_security()