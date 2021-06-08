clear all; clc;
start_up;

load('C:\Users\mungam\Documents\GitHub\Fault_Detection_Diagnostics\FiveLinkWalker_Yukai\Simulation\data\x_multi_extForceDisturbance\EscapeTime_windowsTest_stepKneeExtForce_0N_-3000N16-Mar-2021.mat');
cur=pwd;
data_name='newFeatures_stepKneeExtForce_0N_-3000N_22-Mar-2021.mat';
load_dir = fullfile(cur, 'data\x_multi_extForceDisturbance',data_name);
load(fullfile(load_dir));

%% setting up variables
PCA=0; %use principal components
wholeFeat=0; %use all the features
plotParameters=0;
diffSampleRate=0;
shuffleColmns=1;
saveModel=0;

%% load data
load_info.fivelink=1;
load_info.digit=0;
load_info.noise=0;
if fivelink
    load_info.walk=1;
    load_info.stand=0;
elseif digit
    load_info.walk=0;
    load_info.stand=1;
end

[FDD_info, FDD_info_analyze]=utils.load_data(load_info);
%% 
if diffSampleRate
    FDD_info=FDD_info(1:12:end,:); %change sampling rate
    FDD_info_analyze=FDD_info_analyze(1:12:end,:); %change sampling rate
end
X=normalize(FDD_info_analyze);
step=FDD_info(:,end-2);
if shuffleColmns
    X_orig=X;
    cols = size(X_orig,2);
P = randperm(cols);
X = X_orig(:,P);
end
%% PCA
if PCA
    fprintf('Starting PCA') ;
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
    
    X=X*V(:,1:3); %project X onto the PCA coordinate
   fprintf('PCA done') ;
end



%% creating labels
y=[];
escape_time=5; %# of steps before a fall
[LIA,LOC]=ismember(escape_time,step);
y(1:LOC-1)=zeros(1,LOC-1);
y(LOC:length(step))=ones(1,length(step)-(LOC-1));
y=y';
%% getting training and testing data
rand_num = randperm(size(X,1));
X_train = X(rand_num(1:round(0.8*length(rand_num))),:);
y_train = y(rand_num(1:round(0.8*length(rand_num))),:);

X_test = X(rand_num(round(0.8*length(rand_num))+1:end),:);
y_test = y(rand_num(round(0.8*length(rand_num))+1:end),:);


%% CV partition
c = cvpartition(y_train,'k',5);
%% feature selection
if ~PCA
    opts = statset('display','iter');
    classf = @(train_data, train_labels, test_data, test_labels)...
        sum(predict(fitcsvm(train_data, train_labels,'KernelFunction','rbf'), test_data) ~= test_labels);
    
    [fs, history] = sequentialfs(classf, X_train, y_train, 'cv', c, 'options', opts);%,'nfeatures',2);
    %'Direction','forward');
    % [fs, history] = sequentialfs(classf, X_train, y_train, 'cv', c, 'options', opts,'nfeatures',2);
    
    
    %% Best hyperparameter
    
    X_train_w_best_feature = X_train(:,fs);
    X_test_w_best_feature = X_test(:,fs);
    if sum(fs)> 3
        [L_O,S_O]=RPCA(X_train_w_best_feature);
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
        
        X_train_w_best_feature=X_train_w_best_feature*V(:,1:3); %project X into the PCA coordinate
        X_test_w_best_feature=X_test_w_best_feature*V(:,1:3);
    end
    
    %plot best parameters
    if plotParameters
        figure
        hold on
        k=1;
        prev=step(1);
        jetcustom = jet(step(1)+1);
        
        for i=1:length(step)
            comp=step(i);
            if  comp~= prev
                prev=comp;
                k=k+1;
            end
            
            plot3(X(i,1),X(i,2),X(i,3),'x','Color', jetcustom(k,:),'LineWidth',2);
        end
        view(155,15), grid on, set(gca,'FontSize',13)
        xlabel('lstance y')
        ylabel('v sw z')
        zlabel('p dsw x')
        
        colormap(jetcustom);
        cb = colorbar;
        
        caxis([-PCA_info_full(1,end-2) 0])
        ylabel(cb,'escape time')
        
        hold off
    end
else
    X_train_w_best_feature = X_train;
    X_test_w_best_feature = X_test;
end
%%

Md1 = fitcsvm(X_train_w_best_feature,y_train,'KernelFunction','rbf','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus','ShowPlots',true)); % Bayes' Optimization ??.
%% saving model
if saveModel
Md_pca_eT5=Md1; %escape time 10 different sample rate PCA
%Md_pca_eT5_R=Md1; %escape time 10 different sample rate PCA
%Md_pca_eT10_R=Md1; %escape time 10 different sample rate PCA
%Md_feat3_eT10_R=Md1; %escape time 10 different sample rate
% Md_feat3_eT5_R=Md1; %escape time 5 different sample rate
% Md_feat3_eT10=Md1;
% Md_feat3=Md1; %escape time 5
Mdsvm.Md_pca_eT5=Md_pca_eT5;
Mdsvm.Md_pca_eT5_R=Md_pca_eT5_R;
Mdsvm.Md_pca_eT10_R=Md_pca_eT10_R;
Mdsvm.Md_feat3_eT10_R=Md_feat3_eT10_R;
Mdsvm.Md_feat3_eT5_R=Md_feat3_eT5_R;
Mdsvm.Md_feat3=Md_feat3;


data_name = char(datetime('now','TimeZone','local','Format','d-MMM-y'));%'local/longer_double_support_wider_step_dummy';
% name_save = [num2str(externalForce(1)), 'N_', data_name];
name_save = ['svm_model_ET_5_10', data_name];
save_dir = fullfile(cur, 'data\svm_models');
 if ~exist(save_dir,'dir'), mkdir(save_dir); end
 file_name = [name_save, '.mat'];
fprintf('Saving info %s\n', file_name);
        
save(fullfile(save_dir, file_name), 'Mdsvm');
fprintf('Saved')
end
%% Final test with test set

test_accuracy_for_iter = sum((predict(Md1,X_test_w_best_feature) == y_test))/length(y_test)*100

%% plot decision surface
% run function described above
labels={'lcom y';
    'lstance y';
    'v sw x';
    'v sw z'
    'p sw x';
    'torso angle';
    'com height';
    'p st z';
    'p_relCOMLegs'
    'y1';
    'y2';
    'y3';
    'y4';
    'dy1';
    'dy2';
    'dy3';
    'dy4';
    'stanceToePosx';
    };
if shuffleColmns
    labels_org=labels;
    labels=labels_org(P,:);
end
labels_best=labels(fs);
if sum(fs)==3 || PCA
    if ~PCA
    plotInfo.xlabel=labels_best{1};
    plotInfo.ylabel=labels_best{2};
    plotInfo.zlabel=labels_best{3};
    plotInfo.title='svm rbf';
    else
           plotInfo.xlabel='V1';
    plotInfo.ylabel='V2';
    plotInfo.zlabel='V3';
    end
    svm_3d_matlab_vis(Md1,X_train_w_best_feature,y_train,X_test_w_best_feature,y_test,plotInfo)
else
    plotInfo.xlabel=labels_best{1};
    plotInfo.ylabel=labels_best{2};
    plotInfo.title='svm rbf feature selection';
    svm_2d_matlab_vis(Md1,X_train_w_best_feature,y_train,X_test_w_best_feature,y_test,plotInfo)
    
end
%% PCA

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