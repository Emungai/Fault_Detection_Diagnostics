import numpy as np
import scipy.io as spio

from predictor import angularMomentumPredictor

def getDigitData(feat_info):
    #get parameters
    diff_sampleRate=feat_info['diff_sampleRate']
    forceAxis=feat_info['forceAxis']
    file_name=feat_info['file_name']
    allfeat=feat_info['all_feat']

    #loading data
    #see https://docs.scipy.org/doc/scipy/reference/tutorial/io.html for reference
    dir='/home/exo/Documents/eva/Fault_Detection_Diagnostics/FDD_Analysis/data/digit/Biped_Controller/'
    mat=spio.loadmat(dir+forceAxis+'_force_act_moment/'+file_name, struct_as_record=False, squeeze_me=True)
    # if forceAxis == "y":
    #     mat=spio.loadmat('/home/exo/Documents/eva/Fault_Detection_Diagnostics/FDD_Analysis/data/digit/Biped_Controller/y_force_act_moment/100N_4-10-21.mat', struct_as_record=False, squeeze_me=True)
    # elif forceAxis == "x":
    #     mat=spio.loadmat('/home/exo/Documents/eva/Fault_Detection_Diagnostics/FDD_Analysis/data/digit/Biped_Controller/x_force_act_moment/100N_4-10-21.mat', struct_as_record=False, squeeze_me=True)

    #get features
    # region
    logger=mat['logger']
    q_all=logger.q_all
    dq_all=logger.dq_all
    ua_all=logger.ua_all
    ud_all=logger.ud_all
    LG=logger.LG
    L_LeftFoot=logger.L_LeftFoot
    L_RightFoot=logger.L_RightFoot
    rp_COMFoot=logger.rp_COMFoot
    task=logger.task
    time_data=logger.time
    feat=logger.feat_names
    p_LeftFoot=logger.p_LeftFoot
    rpy_LeftFoot=logger.rpy_LeftFoot
    p_RightFoot=logger.p_RightFoot
    rpy_RightFoot=logger.rpy_RightFoot
    p_com=logger.p_com
    p_links=logger.p_links
    if not allfeat:
        if forceAxis == "y":
            q_idx=np.array([2,3,6,7,11,13,14,18,22,24,25])-1
            u_idx=np.array([1,5,6,7,11,12,13,17])-1
            L_idx=0
            p_idx=np.array([2,3])-1

        elif forceAxis== "x":
            q_idx=np.array([1,3,5,9,10,11,12,15,17,20,21,22,23,26,28])-1
            u_idx=np.array([3,4,5,6,9,10,11,10,14,15,18,20])-1
            L_idx=1
            p_idx=np.array([1,3])-1  
    else:
        q_idx=np.arange(len(q_all))
        u_idx=np.arange(len(ua_all))
        L_idx=np.arange(len(L_LeftFoot))
        p_idx=np.arange(len(p_com))      


    q_all_f=np.array(q_all[q_idx,])
    dq_all_f=np.array(dq_all[q_idx,])
    ua_all_f=np.array(ua_all[u_idx,])
    ud_all_f=np.array(ud_all[u_idx,])
    LG_all_f=np.array(LG[L_idx,])
    L_LeftFoot_f=np.array(L_LeftFoot[L_idx,])
    L_RightFoot_f=np.array(L_RightFoot[L_idx,])
    rp_COMFoot_f=np.array(rp_COMFoot[p_idx,])
    p_LeftFoot_f=np.array(p_LeftFoot[p_idx,])
    rpy_LeftFoot_f=np.rad2deg(np.array(rpy_LeftFoot[L_idx,]))
    p_RightFoot_f=np.array(p_RightFoot[p_idx,])
    rpy_RightFoot_f=np.rad2deg(np.array(rpy_RightFoot[L_idx,]))
    p_com=np.array(p_com)
    p_links=np.array(p_links)

    logger_dict={}
    logger_dict['q_all_f']=q_all_f
    logger_dict['dq_all_f']=dq_all_f
    logger_dict['ua_all_f']=ua_all_f
    logger_dict['ud_all_f']=ud_all_f
    logger_dict['LG_all_f']=LG_all_f
    logger_dict['L_LeftFoot_f']=L_LeftFoot_f
    logger_dict['L_RightFoot_f']=L_RightFoot_f
    logger_dict['rp_COMFoot_f']=rp_COMFoot_f
    logger_dict['p_LeftFoot_f']=p_LeftFoot_f
    logger_dict['rpy_LeftFoot_f']=rpy_LeftFoot_f
    logger_dict['p_RightFoot_f']=p_RightFoot_f
    logger_dict['rpy_RightFoot_f']=rpy_RightFoot_f
    logger_dict['p_com']=p_com
    logger_dict['p_links']=p_links
    logger_dict['task']=task
    logger_dict['time_data']=time_data
    return logger_dict




def getDigitFeatures(feat_info):
    diff_sampleRate=feat_info['diff_sampleRate']
    forceAxis=feat_info['forceAxis']
    file_name=feat_info['file_name']
    allfeat=feat_info['all_feat']

    logger_dict=getDigitData(feat_info)

    q_all_f=logger_dict['q_all_f']
    dq_all_f=logger_dict['dq_all_f']
    ua_all_f=logger_dict['ua_all_f']
    ud_all_f=logger_dict['ud_all_f']
    LG_all_f=logger_dict['LG_all_f']
    L_LeftFoot_f=logger_dict['L_LeftFoot_f']
    L_RightFoot_f=logger_dict['L_RightFoot_f']
    rp_COMFoot_f=logger_dict['rp_COMFoot_f']
    p_LeftFoot_f=logger_dict['p_LeftFoot_f']
    rpy_LeftFoot_f=logger_dict['rpy_LeftFoot_f']
    p_RightFoot_f=logger_dict['p_RightFoot_f']
    rpy_RightFoot_f=logger_dict['rpy_RightFoot_f']
    p_com=logger_dict['p_com']
    p_links=logger_dict['p_links']
    task=logger_dict['task']
    time_data=logger_dict['time_data']

    # angularMomentumPredictor(logger,logger.L_AverageFeet[:,0])

    # FDD = np.row_stack((q_all_f, dq_all_f,ua_all_f-ud_all_f,LG_all_f,L_LeftFoot_f,L_RightFoot_f,rp_COMFoot_f))
    FDD = np.row_stack((q_all_f, dq_all_f,LG_all_f,L_LeftFoot_f,L_RightFoot_f,rp_COMFoot_f))
    FDD=np.transpose(FDD)
    if not allfeat:
        feet_info=np.transpose(np.row_stack((p_LeftFoot_f,rpy_LeftFoot_f,p_RightFoot_f,rpy_RightFoot_f)))
    else:
        if forceAxis=="y":
            p_feet_idx=np.array([2,3])-1
            r_feet_idx=1           
        elif forceAxis=="x":
            p_feet_idx=np.array([1,3])-1  
            r_feet_idx=1
        feet_info=np.transpose(np.row_stack((p_LeftFoot_f[p_feet_idx,],rpy_LeftFoot_f[r_feet_idx,],p_RightFoot_f[p_feet_idx,],rpy_RightFoot_f[r_feet_idx,])))


    p_com=np.transpose(p_com)
    p_links=np.transpose(p_links)
    #endregion 

    #*****cutting off the part where AR controller's was running***
    # region
    t=np.array(time_data)-3
    idx_bc=np.where(t<0)
    idx_end=len(t)
    FDD_bc=FDD[idx_bc[0][-1]:idx_end,]
    time_data=time_data[idx_bc[0][-1]:idx_end]
    feet_info=feet_info[idx_bc[0][-1]:idx_end]
    p_links=p_links[idx_bc[0][-1]:idx_end]
    p_com=p_com[idx_bc[0][-1]:idx_end]

    # endregion


    # get different sample rate
    if diff_sampleRate:

        FDD_bc=FDD_bc[0::12,:] #to grab everyother column Y[:,0::2]
        time_data=time_data[0::12]
        feet_info=feet_info[0::12]
        p_links=p_links[0::12]
        p_com=p_com[0::12]


    #figure out when robot has started to fall and fallen, and create label for classification
    #region
    label=np.zeros((len(FDD_bc),1))
    #robot starting to fall
    feet_info_r=np.column_stack((feet_info[:,2],feet_info[:,5]))
    feet_info_rb=30- np.abs(feet_info_r)
    #first figuring out where feet_info_rb is negaitve since that's where the feet angles are greater than zero
    #we do this using feet_info_rb<0 which returns a Boolean (note it'll return true when the feet angles are greater than 30) 
    #we then sum the rows of this Boolean, if the feet angles are greater than
    #zero, the summation will be greater than 0
    r_falling=np.where(np.sum(feet_info_rb<0,axis=1)>0)
    if len(r_falling[0])>0:
        idx_falling=r_falling[0][0]
        label[idx_falling:len(label):,]=1
    #robot has fallen
    p_linksSansToes=np.delete(p_links,[6,7,17,18],1)
    p_feet=np.concatenate(([feet_info[0,1]],[feet_info[0,4]],p_links[0,[6,7,17,18]])) #initial z positions for: left foot, right foot,left toe pitch link, left toe roll link, right toe pitch link, right toe roll link
    p_feet_zmax=np.amax(p_feet)
    p_linksSansToesb=p_linksSansToes-p_feet_zmax-0.1 #trying to find where the other limbs are atleast 0.1m away from floor
    r_fallen=np.where(np.sum(p_linksSansToesb<0,axis=1)>0)
    if len(r_fallen[0])>0:
        idx_fallen=r_fallen[0][0]
        label[idx_fallen:len(label):,]=2
    #endregion

    data={}
    data['time_data']=time_data
    data['FDD_bc']=FDD_bc
    data['label']=label
    return data

def getDigitState(feat_info):
    logger_dict=getDigitData(feat_info)
    #get parameters
    diff_sampleRate=feat_info['diff_sampleRate']
    forceAxis=feat_info['forceAxis']
    file_name=feat_info['file_name']
    allfeat=feat_info['all_feat']

    q_all_f=logger_dict['q_all_f']
    dq_all_f=logger_dict['dq_all_f']
    ua_all_f=logger_dict['ua_all_f']
    ud_all_f=logger_dict['ud_all_f']
    LG_all_f=logger_dict['LG_all_f']
    L_LeftFoot_f=logger_dict['L_LeftFoot_f']
    L_RightFoot_f=logger_dict['L_RightFoot_f']
    rp_COMFoot_f=logger_dict['rp_COMFoot_f']
    p_LeftFoot_f=logger_dict['p_LeftFoot_f']
    rpy_LeftFoot_f=logger_dict['rpy_LeftFoot_f']
    p_RightFoot_f=logger_dict['p_RightFoot_f']
    rpy_RightFoot_f=logger_dict['rpy_RightFoot_f']
    p_com=logger_dict['p_com']
    p_links=logger_dict['p_links']
    task=logger_dict['task']
    time_data=logger_dict['time_data']

    q_all_f=np.transpose(q_all_f)
    dq_all_f=np.transpose(dq_all_f)
    ua_all_f=np.transpose(ua_all_f)
    #*****cutting off the part where AR controller's was running***
    # region
    t=np.array(time_data)-3
    idx_bc=np.where(t<0)
    idx_end=len(t)
    q_all_f_bc=q_all_f[idx_bc[0][-1]:idx_end,]
    dq_all_f_bc=dq_all_f[idx_bc[0][-1]:idx_end,]
    ua_all_f_bc=ua_all_f[idx_bc[0][-1]:idx_end,]
    time_data_bc=time_data[idx_bc[0][-1]:idx_end]

    #*****cutting off the part after 10 seconds, robot has fallen
    # region 
    t=np.array(time_data_bc)-10
    idx_be=np.where(t>0)
    q_all_f_be=q_all_f_bc[0:idx_be[0][0],]
    dq_all_f_be=dq_all_f_bc[0:idx_be[0][0],]
    ua_all_f_be=ua_all_f_bc[0:idx_be[0][0],]
    time_data_be=time_data_bc[0:idx_be[0][0],]  

    # get different sample rate
    if diff_sampleRate:
        time_data_fin=time_data_be[0::12]
        q_all_f_fin=q_all_f_be[0::12]
        dq_all_f_fin=dq_all_f_be[0::12]
        ua_f_fin=ua_all_f_be[0::12]

    diff_t=np.diff(time_data)
    # q_all_f_fin=q_all_f_fin[0:196,]
    # dq_all_f_fin=dq_all_f_fin[0:196,]
    X=np.concatenate((q_all_f_fin, dq_all_f_fin), axis=1)
    return X, time_data_fin, ua_f_fin



    