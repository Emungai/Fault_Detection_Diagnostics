clear all; clc;
%% load paths
addpath(genpath(pwd));
addpath('C:\Users\mungam\Documents\School\TextBooks_Resources\Textboooks\Brunton_Data-Driven Science and Engineering\Code\CH03');
addpath('C:\Users\mungam\Documents\School\softwareTools\libsvm\libsvm-3.24\matlab');


%% setting up variables
%loading data
fivelink=0;
digit=1;
noise=0; %use noisy data

%plot
plot_LminusS=1; %set to 1 if you want to plot L minus S

%% loading Data if necessary
load_info.fivelink=fivelink;
load_info.digit=digit;
load_info.noise=noise;
load_info.saveData=1;
if fivelink
    load_info.walk=1;
    load_info.stand=0;
elseif digit
    load_info.walk=0;
    load_info.stand=1;
    load_info.allFeat=0;
    load_info.compileFDD_Data=0;
    load_info.forceAxis='y';
    load_info.name_save='100N_4-4-21';
% load_info.name_save='200N_4-7-21';
end

[FDD_info, FDD_info_analyze,feat]=utils.load_data(load_info);
%% figure out when force was applied
a=FDD_info(:,1)-1e-3;
[r,c]=find(a <0);
force_t=FDD_info(r(end),end);
%%
if digit
    if strcmp(load_info.forceAxis,'x')
    a=FDD_info(:,end)-8.3;
    elseif strcmp(load_info.forceAxis,'y')
        a=FDD_info(:,end)-10;
    end
    [r,c]=find(a <0);
    FDD_info=FDD_info(1:r(end),:);
    FDD_info_analyze=FDD_info_analyze(1:r(end),:);
end
% %getting rid of velocity
% FDD_info=FDD_info(:,[1:10,21:42]);
% FDD_info_analyze=FDD_info_analyze(:,[1:10,21:40]);
%% normalize data
%need to run this on matlab 2018 or recent
X=normalize(FDD_info_analyze);

%% RPCA, PCA
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


fprintf('PCA calc done') ;

%% plot
fprintf('\n starting plot') ;
%plotting variables
plotInfo.X=X;
plotInfo.PCA_info_full=FDD_info;
plotInfo.escapeTime=0; %plot wrt escape time (# of steps before failure)
plotInfo.ramp=1; %plot wrt time
plotInfo.digit=digit;
plotInfo.fivelink=fivelink;
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
plotInfo.titlePlot='DIGIT All RPCA-PCA(L)-X-ExtForce(y 100N)';
FullData=plot.plotPCA(plotInfo);
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

