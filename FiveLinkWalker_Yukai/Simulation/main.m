clear all; clc;
start_up;
%% setting up variables
Save_Solution=0;
noise=0; %set to 1 to run sim w/ noise
animate=0; %set to 1 to visualize the results of sim
FDD_info=[];

%covariance values are from Yukai's code
cov_q=1e-6;
cov_dq=0.0485;
% cov_q=0.05;
% cov_dq=0.0485;

%set where the external force should be applied
info.torso=0;
info.knee=1;
info.hip=0;
info_bus_info = Simulink.Bus.createObject(info);
info_bus = evalin('base', info_bus_info.busName);
externalForce=-3000; %magnitude & direction of external force
Simulation_Time=10; %sim duration
%% RunSim
%%
if ~noise
sim('FLWSim',Simulation_Time)
else
sim('FLWSimNoise',Simulation_Time)
end
% 
X_states=dataInfo.X_states;
tout=dataInfo.tout;
if animate
FA = FLWAnimation;
FA.Initializaztion(X_states',tout');
end


%% saving the data
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
if Save_Solution
% name_save = [site,num2str(externalForce),'N_', data_name];
% save_dir = fullfile(cur, 'data\x_externalForceDisturbance\varyingApplicationSites');
name_save =['stepKneeExtForce_0N_-3000N_', data_name];
save_dir = fullfile(cur, 'data\noisy_data');
 if ~exist(save_dir,'dir'), mkdir(save_dir); end
 file_name = [name_save, '.mat'];
fprintf('Saving info %s\n', file_name);
        
save(fullfile(save_dir, file_name), 'dataInfo');

end







