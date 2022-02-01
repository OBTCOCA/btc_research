##
import numpy as np
from skimage.util import view_as_windows
import sys
##
def roll_sum_ein(x, window):
    X = view_as_windows(x.reshape(-1, 1), (window, 1))[..., 0]
    out = np.full(x.reshape(-1, 1).shape, np.nan).reshape(-1)
    out[window - 1:] = np.einsum('ijk->ij', X).reshape(-1)
    return out
##
def roll_mean_ein(x, window):
    X = view_as_windows(x.reshape(-1, 1), (window, 1))[..., 0]
    out = np.full(x.reshape(-1, 1).shape, np.nan).reshape(-1)
    out[window - 1:] = np.einsum('ijk->ij', X).reshape(-1) / window
    return out
##
def roll_var_ein(x, window):
    X = view_as_windows(x.reshape(-1, 1), (window, 1))[..., 0]
    X_mX = X - X.mean(-1, keepdims=True)
    out = np.full(x.reshape(-1, 1).shape, np.nan).reshape(-1)
    out[window - 1:] = np.einsum('ijk,ijk->ij', X_mX, X_mX).reshape(-1) / (window - 1)
    return out
##
def roll_cov_ein(x, y, window):
    X = view_as_windows(x.reshape(-1, 1), (window, 1))[..., 0]
    Y = view_as_windows(y.reshape(-1, 1), (window, 1))[..., 0]

    X_mX = X - X.mean(-1, keepdims=True)
    Y_mY = Y - Y.mean(-1, keepdims=True)

    out = np.full(x.reshape(-1, 1).shape, np.nan).reshape(-1)
    out[window - 1:] = np.einsum('ijk,ijk->ij', X_mX, Y_mY).reshape(-1) / (window - 1)
    return out
##
def roll_beta_ein(x, y, window):
    x += np.random.randn(len(x)) * sys.float_info.epsilon
    y += np.random.randn(len(x)) * sys.float_info.epsilon

    cov_xy = roll_cov_ein(x, y, window)
    var_x = roll_var_ein(x, window)
    beta = cov_xy / var_x
    return beta
##
def roll_corr_ein(x, y, window):
    cov_xy = roll_cov_ein(x, y, window)
    var_x = roll_var_ein(x, window)
    var_y = roll_var_ein(y, window)
    rho = cov_xy / np.sqrt(var_x * var_y)
    return rho
##
def linearize_dt_ein(p, dt, window):
    DT = view_as_windows(dt.reshape(-1, 1), (window, 1))[..., 0]
    P = view_as_windows(p.reshape(-1, 1), (window, 1))[..., 0]
    CDT = DT.cumsum(axis=2)

    P_mP = P - P.mean(-1, keepdims=True)
    CDT_mCDT = CDT - CDT.mean(-1, keepdims=True)

    cov_p_cdt = np.einsum('ijk,ijk->ij', P_mP, CDT_mCDT).reshape(-1)
    var_cdt = np.einsum('ijk,ijk->ij', CDT_mCDT, CDT_mCDT).reshape(-1)

    out = np.full(p.reshape(-1, 1).shape, np.nan).reshape(-1)
    out[window - 1:] = (cov_p_cdt / var_cdt).reshape(-1)
    return out
##
def linearize_dt_sigma_ein(p, dt, window):
    DT = view_as_windows(dt.reshape(-1, 1), (window, 1))[..., 0]
    P = view_as_windows(p.reshape(-1, 1), (window, 1))[..., 0]
    CDT = DT.cumsum(axis=0)

    P_mP = P - P.mean(-1, keepdims=True)
    CDT_mCDT = CDT - CDT.mean(-1, keepdims=True)

    cov_p_cdt = np.einsum('ijk,ijk->ij', P_mP, CDT_mCDT).reshape(-1)
    var_cdt = np.einsum('ijk,ijk->ij', CDT_mCDT, CDT_mCDT).reshape(-1)

    beta = np.full(p.reshape(-1, 1).shape, np.nan).reshape(-1)
    beta[window - 1:] = (cov_p_cdt / var_cdt).reshape(-1)

    BETA = view_as_windows(beta.reshape(-1, 1), (window, 1))[..., 0]
    ALPHA = (P - BETA * CDT).mean(-1, keepdims=True)
    E = (P - ALPHA - BETA * CDT).std(-1, keepdims=True).reshape(1, -1)[0]
    E1 = np.full(p.reshape(-1, 1).shape, np.nan).reshape(-1)
    E1[(window - 1):] = E
    return E1
##
def cum_ofi_bps_ein(ofi, bps, window):
    OFI = view_as_windows(ofi.reshape(-1, 1), (window, 1))[..., 0]
    BPS = view_as_windows(bps.reshape(-1, 1), (window, 1))[..., 0]

    COFI = OFI.cumsum(axis=0)
    CBPS = BPS.cumsum(axis=0)

    COFI_mCOFI = COFI - COFI.mean(-1, keepdims=True)
    CBPS_mCBPS = CBPS - CBPS.mean(-1, keepdims=True)

    cov_cofi_cbps = np.einsum('ijk,ijk->ij', COFI_mCOFI, CBPS_mCBPS).reshape(-1)
    var_cofi = np.einsum('ijk,ijk->ij', COFI_mCOFI, COFI_mCOFI).reshape(-1)

    out = np.full(bps.reshape(-1, 1).shape, np.nan).reshape(-1)
    out[window - 1:] = (cov_cofi_cbps / var_cofi).reshape(-1)
    return out
##
def ein_semi_vol(bps, sign, window):
    x_ = bps.copy()
    x_[x_ * sign < 0] = 0

    X_ = view_as_windows(x_.reshape(-1, 1), (window, 1))[..., 0]
    X__mX_ = X_ - X_.mean(-1, keepdims=True)

    out = np.full(x_.reshape(-1, 1).shape, np.nan).reshape(-1)
    out[window - 1:] = np.einsum('ijk,ijk->ij', X__mX_, X__mX_).reshape(-1)
    return out
##
def ein_pnlr(bps, dt, window):
    DT = view_as_windows(dt.reshape(-1, 1), (window, 1))[..., 0]
    BPS = view_as_windows(bps.reshape(-1, 1), (window, 1))[..., 0]
    sDT = DT.sum(-1, keepdims=True).reshape(-1)
    sBPS = BPS.sum(-1, keepdims=True).reshape(-1)
    out = np.full(bps.reshape(-1, 1).shape, np.nan).reshape(-1)
    out[window - 1:] = sBPS / sDT
    return out
##
def ein_roll_min(x, window):
    X = view_as_windows(x.reshape(-1, 1), (window, 1))[..., 0]
    Xmin = X.min(-1, keepdims=True)
    out = np.full(x.reshape(-1, 1).shape, np.nan).reshape(-1)
    out[window - 1:] = Xmin.reshape(-1)
    return out
##
def ein_roll_max(x, window):
    X = view_as_windows(x.reshape(-1, 1), (window, 1))[..., 0]
    Xmax = X.max(-1, keepdims=True)
    out = np.full(x.reshape(-1, 1).shape, np.nan).reshape(-1)
    out[window - 1:] = Xmax.reshape(-1)
    return out
##


