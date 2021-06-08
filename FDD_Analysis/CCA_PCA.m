start_up;
%% get data

data_name='swingKnee_-3000N_24-Mar-2021.mat';
load_dir = fullfile(cur, 'data\x_externalForceDisturbance\varyingApplicationSites',data_name);
load(fullfile(load_dir));
Data=dataInfo.Data;
CCA_info=[Data.lCoM.Data(1,:)',Data.lstance.Data(1,:)',Data.v_sw.Data(1:2,:)',...
    Data.p_sw.Data(2,:)',Data.torso_angle.Data(1,:)',...
    Data.CoM_height.Data(1,:)',Data.p_st.Data(2,:)',Data.p_relCoMLegs.Data(1,:)',...%Data.stepDuration.Data(1,:)',...
    Data.y.Data(1:4,:)',Data.dy.Data(1:4,:)',...% ...
    Data.stanceFootxMove.Data(1,:)',...
    Data.task.Data(1,:)',...
    Data.step.Data(end)-Data.step.Data(1,:)',... %Data.st_GRF.Data(1:2,:)',Data.sw_GRF.Data(1:2,:)',... %don't need this
    Data.p_st.Time,Data.f_ext.Data(1,:)'];

data_name='torso_-1000N_24-Mar-2021.mat';
load_dir = fullfile(cur, 'data\x_externalForceDisturbance\varyingApplicationSites',data_name);
load(fullfile(load_dir));
Data=dataInfo.Data;
PCA_info=[Data.lCoM.Data(1,:)',Data.lstance.Data(1,:)',Data.v_sw.Data(1:2,:)',...
    Data.p_sw.Data(2,:)',Data.torso_angle.Data(1,:)',...
    Data.CoM_height.Data(1,:)',Data.p_st.Data(2,:)',Data.p_relCoMLegs.Data(1,:)',...%Data.stepDuration.Data(1,:)',...
    Data.y.Data(1:4,:)',Data.dy.Data(1:4,:)',...% ...
    Data.stanceFootxMove.Data(1,:)',...
    Data.task.Data(1,:)',...
    Data.step.Data(end)-Data.step.Data(1,:)',... %Data.st_GRF.Data(1:2,:)',Data.sw_GRF.Data(1:2,:)',... %don't need this
    Data.p_st.Time,Data.f_ext.Data(1,:)'];

%% normalize data
X=normalize(PCA_info(:,1:end-4));

Y=normalize(CCA_info(:,1:end-4));




%% make Y and X same size
escape_time_fin=0; %# of steps before a fall
step_PCA=PCA_info(:,end-2);
step_CCA=CCA_info(:,end-2);
[LIA_PCA,LOC_PCA]=ismember(escape_time_fin,step_PCA);
[LIA_CCA,LOC_CCA]=ismember(escape_time_fin,step_CCA);
X=X(1:LOC_PCA,:);
Y=Y(1:LOC_PCA,:);
PCA_info=PCA_info(1:LOC_PCA,:);
CCA_info=CCA_info(1:LOC_PCA,:);

% escape_time_init=7; %# of steps before a fall
% step_PCA=PCA_info(:,end-2);
% step_CCA=CCA_info(:,end-2);
% [LIA_PCA,LOC_PCA_init]=ismember(escape_time_init,step_PCA);
% [LIA_CCA,LOC_CCA_init]=ismember(escape_time_init,step_CCA);
% if LOC_PCA_init >1
%     X=X(LOC_PCA_init:end,:);
% end
% if LOC_CCA_init>1
%     Y=Y(LOC_CCA_init:end,:);
% end

% PCA_info=PCA_info(LOC_PCA_init:LOC_PCA,:);
% CCA_info=CCA_info(LOC_CCA_init:LOC_PCA,:);
%% perform CCA
[A,B,r,U,V,stats] = canoncorr(X,Y);
%%
escape_time_fin=3; %# of steps before a fall
step_PCA=PCA_info(:,end-2);
step_CCA=CCA_info(:,end-2);
[LIA_PCA,LOC_PCA]=ismember(escape_time_fin,step_PCA);
[LIA_CCA,LOC_CCA]=ismember(escape_time_fin,step_CCA);
X=X(1:LOC_PCA,:);
Y=Y(1:LOC_PCA,:);
PCA_info=PCA_info(1:LOC_PCA,:);
CCA_info=CCA_info(1:LOC_PCA,:);
U=X*A(:,1)-mean(X*A(:,1));
V=Y*B(:,1)-mean(Y*B(:,1));

%% plot


figure

for j=1:1
    
    V_n=V(:,j);
    U_n=U(:,j);
    
    % subplot(1,4,j), hold on
    hold on
    
    
    
    
    
    
    
        
    colorNum=PCA_info(1,end-2)-PCA_info(end,end-2)+1;
  
        
    
    jetcustom = jet(colorNum);
    for i=1:size(V_n,1)
        x =U_n(i) ;
        y = V_n(i);
        
        
        comp=PCA_info(i,end-2)-PCA_info(end,end-2)+1;
        if comp ==0
            comp=comp+1
        end
        
        plot(x,y,'x','Color', jetcustom(comp,:),'LineWidth',2);
    end
    
    
end
% end
colormap(jetcustom);
cb = colorbar;

axisVec=[-escape_time_init -escape_time_fin];
caxis(axisVec)
ylabel(cb,'escape time')

%view(155,15), grid on, set(gca,'FontSize',13)
xlabel(['U',num2str(j)])
ylabel(['V',num2str(j)])
title(["CCA of -3000N swing knee, -1000N torso", "escape time for torso"])


