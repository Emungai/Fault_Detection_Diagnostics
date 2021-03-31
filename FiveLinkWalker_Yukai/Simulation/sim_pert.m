%% RunSim
%%
%covariance values are from Yukai's code
cov_q=1e-6;
cov_dq=0.0485;
% cov_q=0.05;
% cov_dq=0.0485;
info.torso=0;
info.knee=1;
info.hip=0;
info_bus_info = Simulink.Bus.createObject(info);
info_bus = evalin('base', info_bus_info.busName);

externalForce=-3000;
Simulation_Time=10;
% sim('FLWSim',Simulation_Time)
sim('FLWSimNoise',Simulation_Time)
% 
X_states=dataInfo.X_states;
tout=dataInfo.tout;
FA = FLWAnimation;
FA.Initializaztion(X_states',tout');
PCA_info_full=[];
%%
PCA_info=[Data.lCoM.Data(1,:)',Data.lstance.Data(1,:)',Data.v_sw.Data(1:2,:)',...
    Data.p_sw.Data(2,:)',Data.torso_angle.Data(1,:)',...
    Data.CoM_height.Data(1,:)',Data.p_st.Data(2,:)',Data.p_relCoMLegs.Data(1,:)',...%Data.stepDuration.Data(1,:)',...
    Data.y.Data(1:4,:)',Data.dy.Data(1:4,:)', Data.task.Data(1,:)'...
    Data.step.Data(end)-Data.step.Data(1,:)',... %Data.st_GRF.Data(1:2,:)',Data.sw_GRF.Data(1:2,:)',... %don't need this
    Data.p_st.Time,Data.f_ext.Data(1,:)'];
PCA_info_full=[];
PCA_info_full=[PCA_info_full;PCA_info];

%%
dataInfo.Data=Data;
dataInfo.X_states=X_states;
dataInfo.tout=tout;
data_name = char(datetime('now','TimeZone','local','Format','d-MMM-y'));%'local/longer_double_support_wider_step_dummy';
% name_save = [num2str(externalForce(1)), 'N_', data_name];
if ~externalForce
    site='';
elseif info.knee
    site='swingKnee_';
elseif info.hip
    site='hip_';
elseif info.torso
    site='torso_';
end
% name_save = [site,num2str(externalForce),'N_', data_name];
% save_dir = fullfile(cur, 'data\x_externalForceDisturbance\varyingApplicationSites');
name_save =['stepKneeExtForce_0N_-3000N_', data_name];
save_dir = fullfile(cur, 'data\noisy_data');
 if ~exist(save_dir,'dir'), mkdir(save_dir); end
 file_name = [name_save, '.mat'];
fprintf('Saving info %s\n', file_name);
        
save(fullfile(save_dir, file_name), 'dataInfo');


%% Get Data
%% setting up variables
PCA_info=[];
PCA=1;
%% loading Data if necessary
cur=pwd;
data_name='newFeatures_stepKneeExtForce_0N_-3000N_22-Mar-2021.mat';
load_dir = fullfile(cur, 'data\x_multi_extForceDisturbance',data_name);
%noisy data
data_name='stepKneeExtForce_0N_-3000N_23-Mar-2021.mat';
load_dir = fullfile(cur, 'data\noisy_data',data_name);
load(fullfile(load_dir));
PCA_info=[];

Data=dataInfo.Data;
PCA_info=[Data.lCoM.Data(1,:)',Data.lstance.Data(1,:)',Data.v_sw.Data(1:2,:)',...
    Data.p_sw.Data(2,:)',Data.torso_angle.Data(1,:)',...
    Data.CoM_height.Data(1,:)',Data.p_st.Data(2,:)',Data.p_relCoMLegs.Data(1,:)',...%Data.stepDuration.Data(1,:)',...
    Data.y.Data(1:4,:)',Data.dy.Data(1:4,:)',...% ...
    Data.stanceFootxMove.Data(1,:)',...
    Data.task.Data(1,:)',...
    Data.step.Data(end)-Data.step.Data(1,:)',... %Data.st_GRF.Data(1:2,:)',Data.sw_GRF.Data(1:2,:)',... %don't need this
    Data.p_st.Time,Data.f_ext.Data(1,:)'];
% PCA_info_full=[];
PCA_info_full=[PCA_info_full;PCA_info];

X=normalize(PCA_info_full(:,1:end-4));

%% RPCA, PCA
if PCA
    [L_O,S_O]=RPCA(X);
    [V,Score,lmd]=pca(L_O);
    
    var_PCA=[];
    for j=1:length(lmd)
        if j>1
            var_PCA(j)=lmd(j)+var_PCA(j-1);
        else
            var_PCA(j)=lmd(j);
        end
    end
    var_PCA=var_PCA./sum(lmd);
    

   fprintf('PCA calc done\n starting plot') ;
   
   %plotting variables
   plotInfo.X=S_O;
   plotInfo.PCA_info_full=PCA_info_full;
   plotInfo.escapeTime=1;
   plotInfo.ramp=0;
   if plotInfo.ramp
       colorNum=9;
       axisVec=[0 round(PCA_info_full(end,end-1))];
   elseif plotInfo.escapeTime
       colorNum=PCA_info_full(1,end-2)+1;
       axisVec=[-PCA_info_full(1,end-2) 0];
   else
       colorNum=7;
       axisVec=[1 k];
   end
   plotInfo.colorNum=colorNum;
   plotInfo.V=V;
   plotInfo.axisVec=axisVec;
   plotInfo.titlePlot='All RPCA-PCA(S)-S';
   
   FullData=plotPCA(plotInfo);
   
end






