import sox
import os
from glob import glob
from pathlib import Path
def upsample_wav(file, rate):
    tfm = sox.Transformer()
    tfm.rate(rate)
    out_path = file.split('.wav')[0] + "a.wav"
    tfm.build(file, out_path)
    return out_path
mp3_path = Path('/home/sibo/桌面/da/littleow')
files = sorted(list(glob(str(mp3_path / "*/*.wav"))))
for x in files:
    upsample_wav(x,16000)
    #newname = x.split('a.wav')[0]
    #newname = newname + '.wav'
    #os.rename(x, newname)
    os.remove(x)