start_up;
%%
load('C:\Users\mungam\Documents\GitHub\Fault_Detection_Diagnostics\FiveLinkWalker_Yukai\Simulation\data\x_multi_extForceDisturbance\CCA_windowsTest_stepKneeExtForce_0N_-3000N10-Mar-2021.mat')
%%
CCA_info=dataInfo.CCA_info;
PCA_info=dataInfo.PCA_info;
PCA_info_full=[];
PCA_info_full=[PCA_info_full;PCA_info];
%%
X=normalize(PCA_info_full(:,1:end-2));
% X=normalize(X);
[L_O,S_O]=RPCA(X);
%[PCA_info,PCA_info]
Y=normalize(CCA_info);
[L_C,S_C]=RPCA(Y);


%%
[V_P,Score_P,lmd]=pca(L_O);
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
Xavg = mean(X,2);                       % Compute mean
Xmean = X - Xavg*ones(1,size(X,2));  

SC_P=X*V_P;
 %%
 [V_C,Score_C,lmd]=pca(L_C);
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
Yavg = mean(Y,2);                       % Compute mean
Ymean = Y - Yavg*ones(1,size(Y,2));  
 SC_C=Y*V_C;
 
 %% CCA
 [A,B,r,U,V,stats] = canoncorr(SC_P,SC_C);


%%
ramp=1;

figure

for j=1:4

V_n=V(:,j);
U_n=U(:,j);

subplot(1,4,j), hold on
k=1;
beg=1;
% X=normc(PCA_info_full(beg:end,1:end-2));
% X=normr(PCA_info_full(beg:end,1:end-2));
% X=PCA_info_full(beg:end,1:end-2);
goodData=0;
if goodData
    X=dataInfo.PCA_info_full(:,1:end-2);
end
Xavg = mean(X,2);                       % Compute mean
Xmean = X - Xavg*ones(1,size(X,2));  
obs=Xmean;
% V=V_svd;
% for j=1: length(PCA_info_str)
%     obs=PCA_info_str{j}';

if ramp
    prev=round(PCA_info_full(beg,end-1));
    jetcustom = jet(9);
else
prev=PCA_info_full(1,end);
jetcustom = jet(7);
end
jetcustom = jet(9);
for i=1:size(obs,1)
    x =U_n(i) ;
    y = V_n(i);

    if goodData
        plot(x,y,'kx','LineWidth',2);
    else
        if ramp
            
%             comp=round(PCA_info_full(i,end-1));
            comp = fix(PCA_info_full(i,end-1)); %just saves the non-decimal part of the number
        else
            comp=PCA_info_full(i,end);
        end
        if  comp~= prev
            prev=comp;
            k=k+1;
        end
        
%         plot3(x,y,z,'kx','LineWidth',2);
        
      plot(x,y,'x','Color', jetcustom(k,:),'LineWidth',2);
    end
    %         plot3(x,y,z,'rx','LineWidth',2);
   % plot(x,y,'x','Color', jetcustom(k,:),'LineWidth',2)
      
  
end
% end
colormap(jetcustom); 
cb = colorbar; 
if ramp
caxis([0 8]) 
ylabel(cb,'time (rounded)') 
else
 caxis([size(PCA_info_str)]) 
 ylabel(cb,'force') 
end
% caxis([0 k-1]) 
caxis([1 k]) 
%  caxis([0 7]) 
%view(155,15), grid on, set(gca,'FontSize',13)
xlabel(['U',num2str(j)])
ylabel(['V',num2str(j)])
end

