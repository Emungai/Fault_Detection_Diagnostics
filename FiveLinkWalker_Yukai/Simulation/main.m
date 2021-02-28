clear all; clc;
start_up;
PCA_info_full=[];
cur=pwd;
Save_Solution=0;
ramp=1;
%%
k=1;
if ramp
    ext=1;
else
    ext=[0,1600,10000];
end
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
%p_dw=swing leg desired position
%torso_angle=torso angle
%CoM_height=relative height of center of mass with respect to stance foot (p_com-p_st)
%p_st=stance leg position
PCA_info=[Data.lCoM.Data(1,:)',Data.lstance.Data(1,:)',Data.v_sw.Data(1:2,:)',...
    Data.p_sw.Data(1:2,:)',Data.p_dsw.Data(1,:)',Data.torso_angle.Data(1,:)',...
    Data.CoM_height.Data(1,:)',Data.p_st.Data(2,:)',Data.p_st.Time,...
    Data.f_ext.Data(1,:)']; %lCoM,lstance,v_sw,p_sw,p_dw,torso_angle,CoM_height,p_st,time,external force along x
info.Data=Data;
info.PCA_info=PCA_info;

if Save_Solution
data_name = char(datetime('now','TimeZone','local','Format','d-MMM-y'));%'local/longer_double_support_wider_step_dummy';
name_save = [num2str(externalForce(1)), 'N_', data_name];
save_dir = fullfile(cur, 'data\x_externalForceDisturbance');
 if ~exist(save_dir,'dir'), mkdir(save_dir); end
 file_name = [name_save, '.mat'];
fprintf('Saving info %s\n', file_name);
        
save(fullfile(save_dir, file_name), 'info');
end
if ramp
PCA_info_full=[PCA_info_full;PCA_info];
else
PCA_info_full=[PCA_info_full;PCA_info(300:end,:)];
PCA_info_str{k}=PCA_info(300:end,:);
end
% PCA_info_full=[PCA_info_full;[normr(PCA_info(:,1:12)),PCA_info(:,13:end)]];

k=k+1;
end
 %% try
externalForce=1600;
Simulation_Time=10;
sim('FLWSim',Simulation_Time)
% 
FA = FLWAnimation;
FA.Initializaztion(X_states',tout');
PCA_info_full=[];
PCA_info=[Data.lCoM.Data(1,:)',Data.lstance.Data(1,:)',Data.v_sw.Data(1:2,:)',...
    Data.p_sw.Data(1:2,:)',Data.p_dsw.Data(1,:)',Data.torso_angle.Data(1,:)',...
    Data.CoM_height.Data(1,:)',Data.p_st.Data(2,:)',Data.p_st.Time,...
    Data.f_ext.Data(1,:)']; 

PCA_info_full=[PCA_info_full;PCA_info];
%% RPCA
% PCA_info_full_norm=normr(PCA_info_full(1:12,:));
% [L,S]=RPCA(PCA_info_full_norm);
% 
% PCA_info_full_normT=normr(PCA_info_full);
% [L_T,S_T]=RPCA(PCA_info_full_normT);
% 
% [L_F,S_F]=RPCA(PCA_info_full);

X=normc(PCA_info_full(:,1:10));
[L_O,S_O]=RPCA(X);
%[PCA_info,PCA_info]
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
% [V,Score,lmd]=pca(L_O);
[V,Score,lmd]=pca(X);
var_PCA=[];

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

figure, hold on
obs=X;
% V=V_svd;
% for j=1: length(PCA_info_str)
%     obs=PCA_info_str{j}';
k=1;
if ramp
    prev=round(PCA_info_full(1,end-1));
    jetcustom = jet(9);
else
prev=PCA_info_full(1,end);
jetcustom = jet(3);
end

for i=1:size(obs,1)
    x = V(:,1)'*obs(i,:)';
    y = V(:,2)'*obs(i,:)';
    z = V(:,3)'*obs(i,:)';
    if ramp
        comp=round(PCA_info_full(i,end-1));
    else
        comp=PCA_info_full(i,end);
    end
    if  comp~= prev
        prev=comp;
        k=k+1;
    end
    
        plot3(x,y,z,'x','Color', jetcustom(k,:),'LineWidth',2);
%         plot3(x,y,z,'rx','LineWidth',2);
   % plot(x,y,'x','Color', jetcustom(k,:),'LineWidth',2)
      
    
end
% end
colormap(jetcustom); 
cb = colorbar; 
if ramp
caxis([0 8]) 
ylabel(cb,'time') 
else
 caxis([0 3]) 
 ylabel(cb,'force') 
end
view(85,25), grid on, set(gca,'FontSize',13)
xlabel('V1')
ylabel('V2')
zlabel('V3')

% title('RPCA-PCA(L)-X')
title('PCA(X)-X')
%%
feat={'lcom_y','lstance_y','v_sw_x','v_sw_z','p_sw_x','p_sw_z','torso_angle','com_height','p_st_x','p_st_z'};
t=Data.p_com.Time;
prev=PCA_info_full(1,end);
jetcustom = jet(3);
k=1;
figure
for i=1:10
    subplot(2,5,i)
    if ramp
    plot(t,L_O(:,i)-S_O(:,i));
    hold on
    plot(t,zeros(size(L_O(:,i))),'k','LineWidth',2);
    
    else
      
        fst=length(PCA_info_str{1});
        snd=length(PCA_info_str{2});
        thd=length(PCA_info_str{3});
        plot([1:fst],L_O(1:fst,i)-S_O(1:fst,i),'Color',jetcustom(1,:));
        hold on
        plot([fst+1:fst+snd],L_O(fst+1:snd+fst,i)-S_O(fst+1:snd+fst,i),'Color',jetcustom(2,:));
        hold on
        plot([1+snd+fst:fst+snd+thd],L_O(fst+snd+1:end,i)-S_O(fst+snd+1:end,1),'Color',jetcustom(3,:));
        hold on
        plot(zeros(size(L_O(:,i))),'k','LineWidth',2);
    end
 
    title(feat{i})
end
legend('0','1600','10000')