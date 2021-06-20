import numpy as np
# import matlab.engine
# # eng.addpath('/home/exo/Documents/eva/Digit_Controller/version/release_2021.02.11/model/FROST/Kinematics_Dynamics_Generation_Eva/gen/kin_2021_04_06/m',nargout=0)

# def predictFeatures(q,dq):
#    LG.append(eng.LG_all_f(q))
#    L_lFoot.append(getDigitAngularMomentum(p_LF(:,i),[q;dq]))
#    L_rFoot.append(eng.L_LeftFoot(q))
#    rp_CoMFoot.append(eng.L_LeftFoot(q)) 

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

