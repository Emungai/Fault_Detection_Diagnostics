function [FDD_info,FDD_info_analyze]= load_data(info)
fivelink=info.fivelink;
digit=info.digit;
noise=info.noise;
walk=info.walk;
stand=info.stand;
saveData=info.saveData
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
    data_dir='../../Digit_Controller/version/release_2021.02.11/fdd/log_ctrl';
    if stand
                data_name='log_ctrl_y_50N_4-4-21.txt';
%                 data_name='log_ctrl_y_40N_4-4-21.txt';
%         data_name='log_ctrl_y_100N_4-4-21.txt';
        load_dir = fullfile(cur, data_dir,data_name);
        [digitData,~]=utils.getDigitLoggedData(load_dir);
        [FDD_info,FDD_info_analyze]=utils.getDigitFddData(digitData);
        if saveData
            
            data_save.FDD_info=FDD_info;
            data_save.FDD_info_analyze=FDD_info_analyze;
            name_save='50N_4-4-21';
            save_dir=fullfile(cur,'/data/digit/y_force');
            file_name=[name_save,'.mat'];
            fprintf('Saving info %s\n',file_name);
            if ~exist(save_dir,'dir'), mkdir(save_dir);end
            
            save(fullfile(save_dir,file_name),'data_save');
            fprintf('Saved');
        end
        
        
        
    end
else
    fprintf('Unable to Load Data')
    
end
end