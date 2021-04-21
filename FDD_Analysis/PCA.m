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
load_info.noise=noise;
load_info.saveData=1;
if fivelink
    load_info.walk=1;
    load_info.stand=0;
elseif digit_rob
    load_info.walk=0;
    load_info.stand=1;
    load_info.allFeat=0;
    load_info.compileFDD_Data=0;
    load_info.forceAxis='x';
    load_info.name_save='100N_4-10-21';
% load_info.name_save='200N_4-7-21';
end

[data_info]=utils.load_data(load_info);
FDD_info=data_info.FDD_info;
FDD_info_analyze=data_info.FDD_info_analyze;
feat=data_info.feat_f;
feet_info=data_info.feet_info; %[LF_x,LF_y,LF_z,LF_r,LF_p,LF_y,LF_x,RF_y,RF_z,RF_r,RF_p,RF_y]

%% only keep the part where the biped controller was running
 a=FDD_info(:,end)-3;
 [r,c]=find(a <0);
 FDD_info=FDD_info(r(end)+1:length(FDD_info),:);
 FDD_info_analyze=FDD_info_analyze(r(end)+1:length(FDD_info_analyze),:);
 feet_info=feet_info(r(end)+1:length(feet_info),:);
%% get different sample rate
FDD_info=FDD_info(1:12:end,:); %change sampling rate
FDD_info_analyze=FDD_info_analyze(1:12:end,:); %change sampling rate
feet_info=feet_info(1:12:end,:); %change sampling rate
%% figuring out where feet roll/pitch
figure
plot(FDD_info(:,end),rad2deg(feet_info(:,5)))
%% RUNNING PCA
%% normalize data
%need to run this on matlab 2018 or recent
% X=normc(FDD_info_analyze);
X=normalize(FDD_info_analyze);

%% RPCA, PCA
[L_O,S_O]=RPCA(X);
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

%% plot
fprintf('\n starting plot') ;
%plotting variables
plotInfo.X=X;
plotInfo.PCA_info_full=FDD_info;
plotInfo.escapeTime=0; %plot wrt escape time (# of steps before failure)
plotInfo.ramp=1; %plot wrt time
plotInfo.digit=digit;
plotInfo.fivelink=fivelink;
plotInfo.FullData=FullData;
if plotInfo.ramp
    if fivelink
    colorNum=9;
    axisVec=[0 round(FDD_info(end,end-1))];
    elseif digit
        colorNum=fix(FDD_info(end,end))+1;
    axisVec=[0 fix(FDD_info(end,end))];
    end
elseif plotInfo.escapeTime
    colorNum=FDD_info(1,end-2)+1;
    axisVec=[-FDD_info(1,end-2) 0];
else
    colorNum=7;
    axisVec=[1 k];
end
plotInfo.colorNum=colorNum;
plotInfo.V=V;
plotInfo.axisVec=axisVec;
% plotInfo.titlePlot='All RPCA-PCA(L)-X';
plotInfo.titlePlot='DIGIT All RPCA-PCA(L)-X-ExtForce(100N)';

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

