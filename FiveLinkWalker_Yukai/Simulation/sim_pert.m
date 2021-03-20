info.torso=0;
info.knee=1;
info.hip=0;
info_bus_info = Simulink.Bus.createObject(info);
info_bus = evalin('base', info_bus_info.busName);

externalForce=1100;
Simulation_Time=10;
sim('FLWSim',Simulation_Time)
% 
FA = FLWAnimation;
FA.Initializaztion(X_states',tout');
PCA_info_full=[];
%%
dataInfo.Data=Data;
dataInfo.X_states=X_states;
dataInfo.tout=tout;
data_name = char(datetime('now','TimeZone','local','Format','d-MMM-y'));%'local/longer_double_support_wider_step_dummy';
% name_save = [num2str(externalForce(1)), 'N_', data_name];
name_save = ['stepKneeExtForce_0N_1100N_', data_name];
save_dir = fullfile(cur, 'data\x_multi_extForceDisturbance');
 if ~exist(save_dir,'dir'), mkdir(save_dir); end
 file_name = [name_save, '.mat'];
fprintf('Saving info %s\n', file_name);
        
save(fullfile(save_dir, file_name), 'dataInfo');

