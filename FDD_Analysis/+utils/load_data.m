function [data_info]= load_data(info)
fivelink=info.fivelink;
digit=info.digit;
noise=info.noise;
walk=info.walk;
stand=info.stand;
saveData=info.saveData
name_save=info.name_save;
forceAxis=info.forceAxis;
if digit
    allFeat=info.allFeat;
    compileFDD_Data=info.compileFDD_Data;
end
%% fivelink
cur=pwd;
if fivelink
    
    data_dir='..\FiveLinkWalker_Yukai\Simulation';
    if walk
        if ~noise
            data_name='swingKnee_-3000N_24-Mar-2021.mat';
            load_dir = fullfile(cur, data_dir,'data\x_externalForceDisturbance\varyingApplicationSites',data_name);
        else
            %noisy data
            data_name='stepKneeExtForce_0N_-3000N_30-Mar-2021.mat';
            load_dir = fullfile(cur, data_dir, 'data\noisy_data',data_name);
            
        end
        load(fullfile(load_dir));
        Data=dataInfo.Data;
        %FDD_info consists of:
        %lCoM=angular momentum about the center of mass
        %lstance=angular momentum about the stance foot
        %v_sw=swing leg velocity
        %p_sw=swing leg position
        %torso_angle=torso angle
        %CoM_height=relative height of center of mass with respect to stance foot (p_com-p_st)
        %p_st=stance leg z position
        %p_relCoMLegs = CoM x position subtracted from the average x
        %position of the legs
        %stanceFootxMove= distance stance foot moves in x axis during a single step
        %task= records when the external force applied on a robot changes
        %e.g. at beginning robot has no ext force so task is 0 but when ext
        %force is applied task is 1
        % Data.step.Data(end)-Data.step.Data(1,:)'= escape time of robot in
        % terms of steps robot takes before it falls
        %Time = time of the simulation
        %f_ext = external force applied to robot
        FDD_info=[Data.lCoM.Data(1,:)',Data.lstance.Data(1,:)',Data.v_sw.Data(1:2,:)',...
            Data.p_sw.Data(2,:)',Data.torso_angle.Data(1,:)',...
            Data.CoM_height.Data(1,:)',Data.p_st.Data(2,:)',Data.p_relCoMLegs.Data(1,:)',...%Data.stepDuration.Data(1,:)',...
            Data.y.Data(1:4,:)',Data.dy.Data(1:4,:)',...% ...
            Data.stanceFootxMove.Data(1,:)',...
            Data.task.Data(1,:)',...
            Data.step.Data(end)-Data.step.Data(1,:)',... %Data.st_GRF.Data(1:2,:)',Data.sw_GRF.Data(1:2,:)',... %don't need this
            Data.p_st.Time,Data.f_ext.Data(1,:)'];
        
        FDD_info_analyze=FDD_info(:,1:end-4);
        
        
    end
    
    %% digit
elseif digit
    data_dir='../../Digit_Controller/version/release_2021.02.11/fdd/log_ctrl/Biped_Controller';
    if stand
        if strcmp(forceAxis,'y')
            data_name=['log_ctrl_y_', name_save,'.txt'];
            save_dir=fullfile(cur,'/data/digit/Biped_Controller/y_force_act_moment');
            
        elseif strcmp(forceAxis,'x')
            data_name=['log_ctrl_x_', name_save,'.txt'];
            save_dir=fullfile(cur,'/data/digit/Biped_Controller/x_force_act_moment');
        end
        if compileFDD_Data
            
            load_dir = fullfile(cur, data_dir,data_name);
            [digitData,~]=utils.getDigitLoggedData(load_dir);
            [logger]=utils.getDigitFddData(digitData);
            if saveData
                
                
                file_name=[name_save,'.mat'];
                fprintf('Saving info %s\n',file_name);
                if ~exist(save_dir,'dir'), mkdir(save_dir);end
                
                save(fullfile(save_dir,file_name),'logger');
                fprintf('Saved');
            end
        else
            load(fullfile(save_dir,[name_save,'.mat']))
        end
        
        q_all=logger.q_all;
        dq_all=logger.dq_all;
        ua_all=logger.ua_all;
        ud_all=logger.ud_all;
        LG=logger.LG;
        L_LeftFoot=logger.L_LeftFoot;
        L_RightFoot=logger.L_RightFoot;
        rp_COMFoot=logger.rp_COMFoot;
        task=logger.task;
        time=logger.time;
        feat=logger.feat_names;
        feat_all=[];
        p_LF=logger.p_LeftFoot;
        p_RF=logger.p_RightFoot;
        rpy_LF=logger.rpy_LeftFoot;
        rpy_RF=logger.rpy_RightFoot;
        rpy_b_LF=logger.rpy_b_LeftFoot;
        rpy_b_RF=logger.rpy_b_RightFoot;
        p_com=logger.p_com;
        p_links=logger.p_links;
        
        for i=1:length(feat)
            
            feat_all=[feat_all,feat{i}];
        end
        
        if strcmp(forceAxis,'y') && ~allFeat
            q_idx=[2,3,6,7,11,13,14,18,22,24,25];
            u_idx=[1,5,6,7,11,12,13,17];
            L_idx=1;
            p_idx=[2,3];
            q_all_f=q_all(q_idx,:);
            dq_all_f=dq_all(q_idx,:);
            ua_all_f=ua_all(u_idx,:);
            ud_all_f=ud_all(u_idx,:);
            LG_all_f=LG(L_idx,:);
            L_LeftFoot_f=L_LeftFoot(L_idx,:);
            L_RightFoot_f=L_RightFoot(L_idx,:);
            rp_COMFoot_f=rp_COMFoot(p_idx,:);
            p_LF_f=p_LF(p_idx,:);
            p_RF_f=p_RF(p_idx,:);
            rpy_LF_f=rpy_LF(L_idx,:);
            rpy_RF_f=rpy_RF(L_idx,:);
            
            
            
        elseif strcmp(forceAxis,'x') && ~allFeat
            q_idx=[1,3,5,9,10,11,12,15,17,20,21,22,23,26,28];
            u_idx=[3,4,5,6,9,10,11,12,14,16,18,20];
            L_idx=2;
            p_idx=[1,3];
            q_all_f=q_all(q_idx,:);
            dq_all_f=dq_all(q_idx,:);
            ua_all_f=ua_all(u_idx,:);
            ud_all_f=ud_all(u_idx,:);
            LG_all_f=LG(L_idx,:);
            L_LeftFoot_f=L_LeftFoot(L_idx,:);
            L_RightFoot_f=L_RightFoot(L_idx,:);
            rp_COMFoot_f=rp_COMFoot(p_idx,:);
            p_LF_f=p_LF(p_idx,:);
            p_RF_f=p_RF(p_idx,:);
            rpy_LF_f=rpy_LF(L_idx,:);
            rpy_RF_f=rpy_RF(L_idx,:);
            
        else
            q_idx=length(feat{1});
            u_idx=length(feat{3});
            L_idx=3; p_idx=L_idx;
            q_all_f=q_all;
            dq_all_f=dq_all;
            ua_all_f=ua_all;
            ud_all_f=ud_all;
            LG_all_f=LG;
            L_LeftFoot_f=L_LeftFoot;
            L_RightFoot_f=L_RightFoot;
            rp_COMFoot_f=rp_COMFoot;
            p_LF_f=p_LF;
            p_RF_f=p_RF;
            rpy_LF_f=rpy_LF;
            rpy_RF_f=rpy_RF;
            
            
        end
%                 FDD_info=[q_all_f',dq_all_f',(ua_all_f-ud_all_f)',LG',L_LeftFoot',L_RightFoot',rp_COMFoot',task',time'];
        FDD_info=[q_all_f',dq_all_f',LG_all_f',L_LeftFoot_f',L_RightFoot_f',rp_COMFoot_f',task',time'];
        
        FDD_info_analyze=FDD_info(:,1:end-2);
        feat_f=[feat{1}(q_idx), feat{2}(q_idx), feat{3}(u_idx),feat{4}(L_idx),feat{5}(L_idx),feat{6}(L_idx),feat{7}(p_idx),feat{8},feat{9}];
        feet_info=[p_LF_f',p_RF_f',rpy_LF_f',rpy_RF_f'];
        data_info.FDD_info=FDD_info;
        data_info.FDD_info_analyze=FDD_info_analyze;
        data_info.feat_f=feat_f;
        data_info.feet_info=feet_info;
        data_info.p_com=p_com;
        data_info.p_links=p_links;
        
    end
else
    fprintf('Unable to Load Data')
    
end
end