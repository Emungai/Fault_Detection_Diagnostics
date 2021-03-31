clear all; clc;
start_up;
PCA_info_full=[];
PCA_info_full_shr=[];
cur=pwd;
Save_Solution=0;
ramp=0;
info.torso=0;
info.knee=0;
info.hip=1;
info_bus_info = Simulink.Bus.createObject(info);
info_bus = evalin('base', info_bus_info.busName);
%%
k=1;
if ramp
    ext=1;
else
    ext=[0,1600,10000];
end
%%
for i=ext
%%
externalForce=i;
if ramp
    Simulation_Time=10;
else
    Simulation_Time=2;
end


sim('FLWSim',Simulation_Time)
%% saving data
%lCoM=angular momentum about the center of mass 
%lstance=angular momentum about the stance foot 
%v_sw=swing leg velocity
%p_sw=swing leg position 
%p_dw=swing leg desired position- it is the desired relative position of COM to stance foot swing foot in the beginning of next step,(at this step it is still swing foot) so that COM velocity can be V at time T
%torso_angle=torso angle
%CoM_height=relative height of center of mass with respect to stance foot (p_com-p_st)
%p_st=stance leg z position

Simulation_Time=2.5;
externalForce=-500;

info.torso=0;
info.knee=0;
info.hip=1;
info_bus_info = Simulink.Bus.createObject(info);
info_bus = evalin('base', info_bus_info.busName);

sim('FLWSim',Simulation_Time)

% PCA_info=[Data.lCoM.Data(1,:)',Data.lstance.Data(1,:)',Data.v_sw.Data(1:2,:)',...
%     Data.p_sw.Data(1:2,:)',Data.p_dsw.Data(1,:)',Data.torso_angle.Data(1,:)',...
%     Data.CoM_height.Data(1,:)',Data.p_st.Data(2,:)',Data.p_st.Time,...
%     Data.f_ext.Data(1,:)']; %lCoM,lstance,v_sw,p_sw,p_dw,torso_angle,CoM_height,p_st,time,external force along x
PCA_info=[Data.lCoM.Data(1,:)',Data.lstance.Data(1,:)',Data.v_sw.Data(1:2,:)',...
    Data.p_sw.Data(2,:)',Data.p_dsw.Data(1,:)',Data.torso_angle.Data(1,:)',...
    Data.CoM_height.Data(1,:)',Data.p_st.Data(2,:)',Data.p_relCoMLegs.Data(1,:)',...
    Data.p_st.Time,Data.f_ext.Data(1,:)']; 
CCA_info=[Data.st_GRF.Data(1:2,:)',Data.sw_GRF.Data(1:2,:)'];
dataInfo.Data=Data;
dataInfo.PCA_info=PCA_info;
dataInfo.CCA_info=CCA_info;
if ramp
PCA_info_full=[PCA_info_full;PCA_info];
else
PCA_info_full=[PCA_info_full;PCA_info];
PCA_info_str{k}=PCA_info;
PCA_info_full_shr=[PCA_info_full_shr;PCA_info(1200:53:end,:)];
PCA_info_str_shr{k}=PCA_info(1200:53:end,:);
end
dataInfo.PCA_info_full=PCA_info_full;
dataInfo.PCA_info_str=PCA_info_str;
dataInfo.PCA_info_full_shr=PCA_info_full_shr;
dataInfo.PCA_info_str_shr=PCA_info_str_shr;
if Save_Solution
data_name = char(datetime('now','TimeZone','local','Format','d-MMM-y'));%'local/longer_double_support_wider_step_dummy';
% name_save = [num2str(externalForce(1)), 'N_', data_name];
name_save = ['EscapeTime_windowsTest_stepKneeExtForce_0N_-3000N', data_name];
save_dir = fullfile(cur, 'data\x_multi_extForceDisturbance');
 if ~exist(save_dir,'dir'), mkdir(save_dir); end
 file_name = [name_save, '.mat'];
fprintf('Saving info %s\n', file_name);
        
save(fullfile(save_dir, file_name), 'dataInfo');
end

% PCA_info_full=[PCA_info_full;[normr(PCA_info(:,1:12)),PCA_info(:,13:end)]];

k=k+1;
end
 %% try
info.torso=1;
info.knee=0;
info.hip=0;
info_bus_info = Simulink.Bus.createObject(info);
info_bus = evalin('base', info_bus_info.busName);

externalForce=1500;
Simulation_Time=10;
sim('FLWSim',Simulation_Time)
% 
FA = FLWAnimation;
FA.Initializaztion(X_states',tout');
PCA_info_full=[];
% PCA_info=[Data.lCoM.Data(1,:)',Data.lstance.Data(1,:)',Data.v_sw.Data(1:2,:)',...
%     Data.p_sw.Data(1:2,:)',Data.p_dsw.Data(1,:)',Data.torso_angle.Data(1,:)',...
%     Data.CoM_height.Data(1,:)',Data.p_st.Data(2,:)',Data.p_st.Time,...
%     Data.f_ext.Data(1,:)']; 
PCA_info=[Data.lCoM.Data(1,:)',Data.lstance.Data(1,:)',Data.v_sw.Data(1:2,:)',...
    Data.p_sw.Data(2,:)',Data.p_dsw.Data(1,:)',Data.torso_angle.Data(1,:)',...
    Data.CoM_height.Data(1,:)',Data.p_st.Data(2,:)',Data.p_relCoMLegs.Data(1,:)',...%Data.stepDuration.Data(1,:)',...
    Data.step.Data(end)-Data.step.Data(1,:)',...
    Data.p_st.Time,Data.f_ext.Data(1,:)']; 
CCA_info=[Data.st_GRF.Data(1:2,:)',Data.sw_GRF.Data(1:2,:)'];
% PCA_info_full=[PCA_info_full;PCA_info(35:53:end,:)];
% PCA_info_full=[PCA_info_full;PCA_info];
%% RPCA
% PCA_info_full_norm=normr(PCA_info_full(1:12,:));
% [L,S]=RPCA(PCA_info_full_norm);
% 
% PCA_info_full_normT=normr(PCA_info_full);
% [L_T,S_T]=RPCA(PCA_info_full_normT);
% 
% [L_F,S_F]=RPCA(PCA_info_full);

X=normc(PCA_info_full(:,1:end-2));
X=normalize(PCA_info_full(:,1:end-3));
% X=normalize(X);
[L_O,S_O]=RPCA(X);
%[PCA_info,PCA_info]
Y=normalize(CCA_info);
[L_C,S_C]=RPCA(Y);

[A,B,r,U,V,stats] = canoncorr(X,Y);
%%

[r,c]=size(X);
T=[];
% rnk=rank(L_O);
proj=[];
for i=1:c
%     [rnk,rank([L_O,X(:,i)])];
    proj=[proj;[dot(X(:,i),L_O(:,i)),dot(X(:,i),S_O(:,i))]];
%     if rank([L_O,X(:,i)]) == rnk
%         T(i)=1;
%     else
%         T(i)=0;
    end
%% PCA
Lavg = mean(L_O,2);                       % Compute mean
B = L_O - Lavg*ones(1,size(L_O,2));           % Mean-subtracted Data
[U,S,V_svd] = svd(B/sqrt(length(L_O)),'econ');  % Find principal components (SVD)
lmd_svd=diag(S);
[V,Score,lmd]=pca(L_O);
% [V,Score,lmd]=pca(X);
var_PCA=[];
% V=V_svd;
% lmd=lmd_svd;
for j=1:length(lmd)
    if j>1
    var_PCA(j)=lmd(j)+var_PCA(j-1);
    else
         var_PCA(j)=lmd(j);
    end
end
var_PCA=var_PCA./sum(lmd);
%  plot(time,pitchAccel,'Color', jetcustom(j+1,:), 'LineWidth',2);
 Y=V'*L_O';
%% plotting on Principal Axes


%%
feat={'lcom_y','lstance_y','v_sw_x','v_sw_z','p_sw_z','p_dsw_x','torso_angle','com_height','p_st_z','CoM_rel_p_legs'};

prev=PCA_info_full(1,end);
jetcustom = jet(7);
k=1;
figure
for i=1:10
    subplot(2,5,i)
    if ramp
        t=Data.p_com.Time;
        plot(t,L_O(:,i)-S_O(:,i));
        hold on
        plot(t,zeros(size(L_O(:,i))),'k','LineWidth',2);
        
    else
        
        %         fst=length(PCA_info_str{1});
        %         snd=length(PCA_info_str{2});
        %         thd=length(PCA_info_str{3});
        fst=0;
        if window
            t_act=5509;
            jetcustom = jet(2);
            plot([t(1:t_act)],L_O(1:t_act,i)-S_O(1:t_act,i),'Color',jetcustom(1,:));
            hold on
            plot([t(t_act+1:end)],L_O(t_act+1:end,i)-S_O(t_act+1:end,i),'Color',jetcustom(2,:));
            hold on
            plot([t],zeros(size(L_O(:,i))),'k','LineWidth',2);
            
        else
            for j=1:length(PCA_info_str)
                
                snd=length(PCA_info_str{j});
                plot([fst+1:fst+snd],L_O(fst+1:snd+fst,i)-S_O(fst+1:snd+fst,i),'Color',jetcustom(j,:));
                hold on
                fst=fst+snd;
                 plot(zeros(size(L_O(:,i))),'k','LineWidth',2);
            end
        end
        %         hold on
        %         plot([fst+1:fst+snd],L_O(fst+1:snd+fst,i)-S_O(fst+1:snd+fst,i),'Color',jetcustom(2,:));
        %         hold on
        %         plot([1+snd+fst:fst+snd+thd],L_O(fst+snd+1:end,i)-S_O(fst+snd+1:end,1),'Color',jetcustom(3,:));
        %         hold on
       
    end
    
    title(feat{i})
end
sgtitle('L-S')
legend('nominal','-3000 knee')
% legend('nominal','500 torso','-40 torso', '30 knee', '-500 knee', '200 hip','-500 hip')