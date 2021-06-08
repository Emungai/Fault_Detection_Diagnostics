import numpy as np

def angularMomentumPredictor(logger,L_st_toe_init):
    g=9.81
    m=46.2104
    t=logger.time  
    T_total=0.2
    T_left=0.2
    p_com=logger.p_com[:,-1]
    p_LeftFoot=logger.p_LeftFoot[:,-1]
    p_RightFoot=logger.p_RightFoot[:,-1]
    L_st_toe=logger.L_AverageFeet[:,-1]
    L_st_toe_init=logger.L_AverageFeet[:,0]
    p_com_LF=p_com-p_LeftFoot
    p_com_RF=p_com-p_RightFoot
    H=p_com[2]
    rp_sT=0.5*(p_com_LF+p_com_RF)
    l=np.sqrt(g/p_com[2])

    Lx_now=rp_sT[0]*H*l*np.sinh(l*T_left)*rp_sT[0]+np.cosh(l*T_left)*L_st_toe
    Lx_des=rp_sT[0]*H*l*np.sinh(l*T_left)*rp_sT[0]+np.cosh(l*T_total)*L_st_toe_init
    px_des=(Lx_des-np.cosh(l*T_left)*Lx_now)/(m*H*l*np.sinh(l*T_left))
    return px_des

# def inverseKin(px_des):

