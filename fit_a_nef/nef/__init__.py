from fit_a_nef.nef.mfn import FourierNet, FourierNet_key, GaborNet, GaborNet_key
from fit_a_nef.nef.mlp import MLP, MLP_key
from fit_a_nef.nef.rffnet import RFFNet, RFFNet_key
from fit_a_nef.nef.siren import SIREN, SIREN_key

param_key_dict = {
    "FourierNet": FourierNet_key,
    "GaborNet": GaborNet_key,
    "MLP": MLP_key,
    "RFFNet": RFFNet_key,
    "SIREN": SIREN_key,
}
