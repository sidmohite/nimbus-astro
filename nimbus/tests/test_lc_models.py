from pytest import approx
from nimbus import nimbus

def lc_model_linear(M0, alpha, t_0, t):
    return M0 + alpha*(t-t_0)

def nullevent_mlim_pdf(mlow,mhigh):
    return 1./(mhigh-mlow)

def test_lc_model_linear():
    M0 = -20
    alpha = -1
    t_0 = 2458598.85
    t = 2458599.55
    kne_inf = nimbus.Kilonova_Inference(
        lc_model_funcs = [lc_model_linear, lc_model_linear,
                          lc_model_linear],
        nullevent_mlim_pdf = nullevent_mlim_pdf)
    assert kne_inf.lc_model_linear(M0,alpha,t_0,t) == approx(-20.7)
    
def test_lc_model_powerlaw():
    M0 = -20
    gamma = -3
    t_0 = 2458598
    t = 2458599
    kne_inf = nimbus.Kilonova_Inference(
        lc_model_funcs = [lc_model_linear, lc_model_linear,
                          lc_model_linear],
        nullevent_mlim_pdf = nullevent_mlim_pdf)
    assert kne_inf.lc_model_powerlaw(M0,gamma,t_0,t) == approx(-20.00002)
