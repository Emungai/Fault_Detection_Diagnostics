data_name='stepTorsoExtForce_0N_1500N_19-Mar-2021.mat';
data_name='stepKneeExtForce_0N_1100N_19-Mar-2021.mat';
load_dir = fullfile(cur, 'data\x_multi_extForceDisturbance',data_name);
load(fullfile(load_dir));
clear X
Data=dataInfo.Data;
%%
PCA=0;
diffSampleRate=0;
escape_time=5; %# of steps before a fall
%% load svm_model
Mdsvm.Md_pca_eT5=Md_pca_eT5;
Mdsvm.Md_pca_eT5_R=Md_pca_eT5_R;
Mdsvm.Md_pca_eT10_R=Md_pca_eT10_R;
Mdsvm.Md_feat3_eT10_R=Md_feat3_eT10_R;
Mdsvm.Md_feat3_eT5_R=Md_feat3_eT5_R;
Mdsvm.Md_feat3=Md_feat3;
%% prepare data
PCA_info=[Data.lCoM.Data(1,:)',Data.lstance.Data(1,:)',Data.v_sw.Data(1:2,:)',...
    Data.p_sw.Data(2,:)',Data.p_dsw.Data(1,:)',Data.torso_angle.Data(1,:)',...
    Data.CoM_height.Data(1,:)',Data.p_st.Data(2,:)',Data.p_relCoMLegs.Data(1,:)',...%Data.stepDuration.Data(1,:)',...
    Data.step.Data(end)-Data.step.Data(1,:)',... %Data.st_GRF.Data(1:2,:)',Data.sw_GRF.Data(1:2,:)',... %don't need this
    Data.p_st.Time,Data.f_ext.Data(1,:)'];
PCA_info_full=[];
PCA_info_full=[PCA_info_full;PCA_info];

if diffSampleRate
    PCA_info_full=PCA_info_full(1:12:end,:); %change sampling rate
end
X=normalize(PCA_info_full(:,1:end-3));
step=Data.step.Data(end)-Data.step.Data(1,:);
step=PCA_info_full(:,end-2);
%% PCA
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
    
    X=X*V(:,1:3); %project X into the PCA coordinate
   fprintf('PCA done') ;
end
%% creating labels
y=[];

[LIA,LOC]=ismember(escape_time,step);
y(1:LOC-1)=zeros(1,LOC-1);
y(LOC:length(step))=ones(1,length(step)-(LOC-1));
y=y';
%%
if PCA 
    if  escape_time==5 && diffSampleRate
        Md1=Md_pca_eT5_R;
    elseif escape_time==5
      Md1=Md_pca_eT5  ;
    
    
elseif  escape_time==10 && diffSampleRate
          Md1=Md_pca_eT10_R  ;
    end
else
       if  escape_time==5 && diffSampleRate
        Md1=Md_feat3_eT5_R;
        fs=zeros(size(X,2),1);
        fs(4)=1;
        fs(6)=1;
        fs=logical(fs);
        X=X(:,fs);
      
       elseif escape_time==5
      Md1=Md_feat3  ;
      fs=zeros(size(X,2),1);
       
         fs(1)=1;
        fs(4)=1;
        fs(6)=1;
        fs=logical(fs);
        X=X(:,fs);
   
    
       elseif  escape_time==10 && diffSampleRate
          Md1=Md_feat3_eT10_R  ;
           fs=zeros(size(X,2),1);
        fs(7)=1;
        fs(9)=1;
        fs=logical(fs);
        X=X(:,fs);
    end
end
%% predict
test_accuracy_for_iter = sum((predict(Md1,X) == y))/length(y)*100
%% plot
% run function described above
labels={'lcom y';
    'lstance y';
    'v sw x';
    'v sw z'
    'p sw x';
    'p dsw x';
    'torso angle';
    'com height';
    'p st x';
    'p st z';
    'grf st x';
    'grf st z';
    'grf sw x';
    'grf sw z';
    };
labels_best=labels(fs);
if sum(fs)==3 || PCA
    if ~PCA
    plotInfo.xlabel=labels_best{1};
    plotInfo.ylabel=labels_best{2};
    plotInfo.zlabel=labels_best{3};
    plotInfo.title='svm rbf PCA';
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