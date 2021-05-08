# The following snippet has been borrowed (and modified
# to support vectorized operations) from:
#     https://gist.github.com/johnmeade/d8d2c67b87cda95cd253f55c21387e75
#
# Credits to the original author: John Meade

import numpy as np
import torch


def wada_snr(wav):
    # Direct blind estimation of the SNR of a speech signal.
    #
    # Paper on WADA SNR:
    #   http://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf
    #
    # This function was adapted from this matlab code:
    #   https://labrosa.ee.columbia.edu/projects/snreval/#9
    #
    # MIT license, John Meade, 2020

    # init
    eps = torch.tensor(1e-10)
    # next 2 lines define a fancy curve derived from a gamma distribution -- see paper
    db_vals = torch.arange(-20, 101, dtype=torch.float)
    g_vals = torch.tensor(
        [0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006, 0.41027138, 0.41052627,
         0.41101024, 0.41143264, 0.41231718, 0.41337272, 0.41526426, 0.4178192, 0.42077252, 0.42452799,
         0.42918886, 0.43510373, 0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236,
         0.52353709, 0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496, 0.66750818, 0.69583724,
         0.72454762, 0.75414799, 0.78323148, 0.81240985, 0.84219775, 0.87166406, 0.90030504, 0.92880418,
         0.95655449, 0.9835349, 1.01047155, 1.0362095, 1.06136425, 1.08579312, 1.1094819, 1.13277995,
         1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898, 1.25364313, 1.27103891, 1.28718029,
         1.30302865, 1.31839527, 1.33294817, 1.34700935, 1.3605727, 1.37345513, 1.38577122, 1.39733504,
         1.40856397, 1.41959619, 1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568, 1.47486938,
         1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915, 1.5229097, 1.528578,
         1.53389835, 1.5391211, 1.5439065, 1.54858517, 1.55310776, 1.55744391, 1.56164927, 1.56566348,
         1.56938671, 1.57307767, 1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477,
         1.5941969, 1.59693155, 1.599446, 1.60185011, 1.60408668, 1.60627134, 1.60826199, 1.61004547,
         1.61192472, 1.61369656, 1.61534074, 1.61688905, 1.61838916, 1.61985374, 1.62135878, 1.62268119,
         1.62390423, 1.62513143, 1.62632463, 1.6274027, 1.62842767, 1.62945532, 1.6303307, 1.63128026,
         1.63204102])

    # peak normalize, get magnitude, clip lower bound

    wav = wav / torch.abs(wav + eps).max(dim=-1)[0].unsqueeze(1)
    abs_wav = abs(wav)
    abs_wav[abs_wav < eps] = eps

    # calcuate statistics
    # E[|z|]
    v1 = torch.maximum(eps, abs_wav.mean(dim=-1))
    # E[log|z|]
    v2 = torch.log(abs_wav).mean(dim=-1)
    # log(E[|z|]) - E[log(|z|)]
    v3 = torch.log(v1) - v2

    # table interpolation
    wav_snr_idx = None
    if torch.any(g_vals < v3.reshape(-1, 1)):
        wav_snr_idx = torch.where(g_vals < v3.reshape(-1, 1), 1, 0)
        wav_snr_idx = torch.flip(wav_snr_idx, dims=[-1])
        wav_snr_idx = wav_snr_idx.shape[-1] - wav_snr_idx.argmax(dim=-1) - 1

    wav_snr = torch.zeros(wav.shape[0])

    # handle edge cases or interpolate
    if wav_snr_idx is None:
        wav_snr += db_vals[0]
    else:
        mask_indices = torch.where(wav_snr_idx == len(db_vals) - 1, True, False)
        unmask_indices = torch.where(wav_snr_idx != len(db_vals)-1, True, False)

        filtered_db_vals = torch.gather(db_vals.repeat(wav_snr_idx.shape[0], 1), dim=-1, index=wav_snr_idx[unmask_indices].unsqueeze(1)).squeeze(1)
        filtered_db_vals2 = torch.gather(db_vals.repeat(wav_snr_idx.shape[0], 1), dim=-1, index=wav_snr_idx[unmask_indices].unsqueeze(1)+1).squeeze(1)
        filtered_g_vals = torch.gather(g_vals.repeat(wav_snr_idx.shape[0], 1), dim=-1, index=wav_snr_idx[unmask_indices].unsqueeze(1)).squeeze(1)
        filtered_g_vals2 = torch.gather(g_vals.repeat(wav_snr_idx.shape[0], 1), dim=-1, index=wav_snr_idx[unmask_indices].unsqueeze(1)+1).squeeze(1)

        wav_snr[mask_indices] = db_vals[-1]
        wav_snr[unmask_indices] = filtered_db_vals + (v3[unmask_indices] - filtered_g_vals) / (filtered_g_vals2 - filtered_g_vals) * \
                (filtered_db_vals2 - filtered_db_vals)

    # Calculate SNR
    dEng = torch.sum(wav ** 2, dim=-1)
    dFactor = 10 ** (wav_snr / 10)
    dNoiseEng = dEng / (1 + dFactor)  # Noise energy
    dSigEng = dEng * dFactor / (1 + dFactor)  # Signal energy

    # Add a very small scalar to avoid division errors or logarithm errors
    dSigEng += 1e-10
    dNoiseEng += 1e-10

    snr = 10 * torch.log10(dSigEng / dNoiseEng)

    return snr
