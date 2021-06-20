clear all; clc;
%% load pathsjjj
addpath(genpath(pwd));
addpath('C:\Users\mungam\Documents\School\TextBooks_Resources\Textboooks\Brunton_Data-Driven Science and Engineering\Code\CH03');
addpath('C:\Users\mungam\Documents\School\softwareTools\libsvm\libsvm-3.24\matlab');


%% setting up variables
%loading data
fivelink=0;
digit_rob=1;
noise=0; %use noisy data

%plot
plot_LminusS=1; %set to 1 if you want to plot L minus S

%% loading Data if necessary
load_info.fivelink=fivelink;
load_info.digit=digit_rob;
load_info.noise=noise
load_info.saveData=1;
force_all={'60N','80N','10N','45N','75N','65N','90N'};
force_all={'85N','95N'};
force_all={'20N','30N','35N','40N','55N','105N','110N','115N','120N','125N','130N'};
force_all={'125N','130N'};
for i=1:length(force_all)
if fivelink
    load_info.walk=1;
    load_info.stand=0;
elseif digit_rob
    load_info.walk=0;
    load_info.stand=1;
    load_info.allFeat=1;
    load_info.compileFDD_Data=1;
    load_info.forceAxis='x';
    date='_6-7-21';
    force=force_all{i};
    load_info.name_data=[force,date];
%     load_info.name_data='-100N_4-25-21';
%     load_info.name_data='100N_noise_5-6-21';
%      load_info.name_data='70N_5-6-21';
    load_info.name_save=load_info.name_data;
end

[data_info]=utils.load_data(load_info);
FDD_info=data_info.FDD_info;
FDD_info_analyze=data_info.FDD_info_analyze;
feat=data_info.feat_f;
feet_info=data_info.feet_info; %[p_LF_f',rpy_LF_f',p_RF_f,rpy_RF_f']
p_links=data_info.p_links';
end

%% only keep the part where the biped controller was running
a=FDD_info(:,end)-3;
[r,c]=find(a <0);
FDD_info=FDD_info(r(end)+1:length(FDD_info),:);
FDD_info_analyze=FDD_info_analyze(r(end)+1:length(FDD_info_analyze),:);
feet_info=feet_info(r(end)+1:length(feet_info),:);
p_links=p_links(r(end)+1:length(p_links),:);
%% get different sample rate
FDD_info=FDD_info(1:12:end,:); %change sampling rate
FDD_info_analyze=FDD_info_analyze(1:12:end,:); %change sampling rate
feet_info=feet_info(1:12:end,:); %change sampling rate
p_links=p_links(1:12:end,:); %change sampling rate
%% figuring out when robot has started to fall, and fallen and creating label for classification
label=zeros(length(FDD_info_analyze),1);

%robot starting to fall
feet_info_r=rad2deg(feet_info(:,[3,6])); %getting just the pitch or roll of the feet
feet_info_rb=30-abs(feet_info_r); %arbitrarily set 30 degs as the threshold for when robot is standing
%first figuring out where feet_info_rb is negaitve since that's where the feet angles are greater than zero
%we do this using feet_info_rb<0 which returns a Boolean (note it'll return true when the feet angles are greater than 30)
%we then sum the rows of this Boolean, if the feet angles are greater than
%zero, the summation will be greater than 0
[r_falling,c]=find(sum(feet_info_rb<0,2)>0);
label([r_falling(1):length(label)],:)=1;

%when robot has fallen
p_linksSansToes=p_links; p_linksSansToes(:,[7,8,18,19])=[];
%initial z positions for: left foot, right foot,left toe pitch link, left toe roll link, right toe pitch link, right toe roll link
p_feet=[feet_info(1,2),feet_info(1,5),p_links(1,7),p_links(1,8),p_links(1,18),p_links(1,19)];
p_feet_zmax=max(p_feet);
p_linksSansToesb=p_linksSansToes-p_feet_zmax-0.1;
[r_fallen,c]=find(sum(p_linksSansToesb<0,2)>0);


label([r_fallen(1):length(label)],:)=2;



%% RUNNING PCA
%% normalize data
%need to run this on matlab 2018 or recent
% X=normc(FDD_info_analyze);
X=normalize(FDD_info_analyze);
time=FDD_info(:,end);
FDD_info(:,end+1)=label;

%% RPCA, PCA
% tic
% [L_O,S_O]=RPCA(X(1:600,:));
% toc
[L_O,S_O]=RPCA(X);
%%
[V,Score,lmd]=pca(L_O,'Centered',true);

var_PCA=[];
for j=1:length(lmd)
    if j>1
        var_PCA(j)=lmd(j)+var_PCA(j-1);
    else
        var_PCA(j)=lmd(j);
    end
end
var_PCA=var_PCA./sum(lmd);


fprintf('PCA calc done') ;
%%
data=X;
mean_data=mean(data);
center_data=data-repmat(mean_data,[size(data,1) 1]);
FullData=center_data*V;
%% figuring out where feet roll/pitch
newChr = strrep(load_info.name_save,'_',' ');
%plotting time vs roll or pitch
figure
plot(time,rad2deg(feet_info(:,3)),'o','LineWidth',2)
hold on
plot(time,rad2deg(feet_info(:,6)),'x','LineWidth',2)
hold on
plot(time,label,'Color','k','LineWidth',2)
hold on
xline(FDD_info(r_falling(1),end-1))
hold on
xline(FDD_info(r_fallen(1),end-1))
title([load_info.forceAxis,' ',newChr])
xlabel('time (s)')
if strcmp(load_info.forceAxis,'y')
    ylabel('roll')
elseif strcmp(load_info.forceAxis,'x')
    ylabel('pitch')
end
legend('Left Foot','Right Foot')

%plotting feet roll or pitch vs V1
figure
for i=1:3
    subplot(3,1,i)
    plot(FullData(:,i),rad2deg(feet_info(:,3)),'o','LineWidth',2)
    hold on
    plot(FullData(:,i),rad2deg(feet_info(:,6)),'x','LineWidth',2)
    title([load_info.forceAxis,' ',newChr])
    xlabel(['V',num2str(i)])
    if strcmp(load_info.forceAxis,'y')
        ylabel('roll')
    elseif strcmp(load_info.forceAxis,'x')
        ylabel('pitch')
    end
    legend('Left Foot','Right Foot')
end

%plotting feet time vs V1
figure
for i=1:3
    subplot(3,1,i)
    plot(time,FullData(:,i),'o','LineWidth',2)
    hold on
    plot(time,FullData(:,i),'x','LineWidth',2)
    title([load_info.forceAxis,' ',newChr])
    xlabel('time (s)')
    ylabel(['V',num2str(i)])
    legend('Left Foot','Right Foot')
end
%% plot
fprintf('\n starting plot') ;
%plotting variables
plotInfo.X=X;
plotInfo.PCA_info_full=FDD_info;
plotInfo.escapeTime=1; %plot wrt escape time (# of steps before failure)
plotInfo.ramp=0; %plot wrt time
plotInfo.digit=digit_rob;
plotInfo.fivelink=fivelink;
plotInfo.FullData=FullData;

if plotInfo.ramp
    if fivelink
        colorNum=9;
        axisVec=[0 round(FDD_info(end,end-1))];
    elseif digit_rob
        colorNum=fix(FDD_info(end,end))+1;
        axisVec=[0 fix(FDD_info(end,end))];
    end
elseif plotInfo.escapeTime
    if fivelink
        colorNum=FDD_info(1,end-2)+1;
        axisVec=[-FDD_info(1,end-2) 0];
    elseif digit_rob
        colorNum=label(end)+1;
        axisVec=[label(1) label(end)];
    end
else
    colorNum=7;
    axisVec=[1 k];
end


plotInfo.colorNum=colorNum;
plotInfo.V=V;
plotInfo.axisVec=axisVec;
% plotInfo.titlePlot='All RPCA-PCA(L)-X';
if strcmp(load_info.forceAxis,'y')
    plotInfo.titlePlot='DIGIT All RPCA-PCA(L)-Y-ExtForce(100N)';
elseif strcmp(load_info.forceAxis,'x')
    plotInfo.titlePlot='DIGIT All RPCA-PCA(L)-X-ExtForce(100N)';
end

plot.plotPCA(plotInfo);
%%
if plot_LminusS
    escapeTime=1;
    if fivelink
        feat={'lCoM x', 'lstance x', 'v sw x', 'v sw z', ...
            'p sw z', 'torso pitch', 'CoM height', ...
            'p st z', 'p rel CoM x', 'y1', 'y2', 'y3', 'y4',...
            'dy1', 'dy2', 'dy3', 'dy4', 'stance foot x'};
    end
    plotInfo.L=L_O;
    plotInfo.S=S_O;
    plotInfo.t= FDD_info(:,end-1);
    if escapeTime
        plotInfo.task= -FDD_info(:,end-2)+FDD_info(1,end-2); %escape time
    else
        plotInfo.task= FDD_info(:,end-3); %external force
        
    end
    plotInfo.colorNum=plotInfo.task(end)+1;
    plotInfo.feat=feat;
    plotInfo.x_subplot=3; %# of rows of subplot
    plotInfo.y_subplot=6; %# of columns of subplot
    
    plot.plotLminusS(plotInfo);
    
end

