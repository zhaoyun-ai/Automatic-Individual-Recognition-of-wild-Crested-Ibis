from byol_a.common import *
from byol_a.augmentations import PrecomputedNorm
from byol_a.models import AudioNTT2020
import glob
import umap
device = torch.device('cuda')
cfg = load_yaml_config('config.yaml')
print(cfg)

# Mean and standard deviation of the log-mel spectrogram of input audio samples, pre-computed.
# See calc_norm_stats in evaluate.py for your reference.
stats = [-5.4919195,  5.0389895]

# Preprocessor and normalizer.
to_melspec = torchaudio.transforms.MelSpectrogram(
    sample_rate=cfg.sample_rate,
    n_fft=cfg.n_fft,
    win_length=cfg.win_length,
    hop_length=cfg.hop_length,
    n_mels=cfg.n_mels,
    f_min=cfg.f_min,
    f_max=cfg.f_max,
)
normalizer = PrecomputedNorm(stats)

# Load pretrained weights.
model = AudioNTT2020(d=cfg.feature_d)
model.load_weight('/home/sibo/桌面/1/byol-a/checkpoints/BYOLA-NTT2020d3072s64x96-2212241609-e500-bs128-lr0003-rs42.pth', device)
mp3_path = Path('/home/sibo/桌面/da/zhuhuan/')
files = sorted(Path(mp3_path).glob('*/*.wav'))
totle = 1
flag = 0
for x in files:
    # Load your audio file.
    wav, sr = torchaudio.load(x) # a sample from SPCV2 for now
    assert sr == cfg.sample_rate, "Let's convert the audio sampling rate in advance, or do it here online."
    print(x)
    # Convert to a log-mel spectrogram, then normalize.
    lms = normalizer((to_melspec(wav) + torch.finfo(torch.float).eps).log())
    # Now, convert the audio to the representation.
    features = model(lms.unsqueeze(0))
    features = features.tolist()
    x = str(x)
    label = x.split("/")[-2]
    label = list(label)
    print(label)
    if (flag == 0):
        totle = features
        labels = label
        flag = 1
    else:
        totle = np.vstack((totle, features))
        labels = labels + label
        labels = list(map(int, labels))
        print(labels)
print(totle.shape)
l = np.array(labels)
print(l.shape)
np.save("BYOLA-3072-test-feature.npy",totle)
np.save("BYOLA-3072-test-label.npy",labels)

