function FullData=plotPCA(plotInfo)
%% getting variables
X=plotInfo.X;
PCA_info_full=plotInfo.PCA_info_full;
ramp=plotInfo.ramp; %use rounded time for legend
escapeTime=plotInfo.escapeTime; %use escapeTime for legend
colorNum=plotInfo.colorNum; %number of colors for the map
V=plotInfo.V; %principal components
axisVec=plotInfo.axisVec; %legend entry for caxis
titlePlot=plotInfo.titlePlot;
%%
figure, hold on
k=1;
beg=1;
% X=normc(PCA_info_full(beg:end,1:end-2));
% X=normr(PCA_info_full(beg:end,1:end-2));
% X=PCA_info_full(beg:end,1:end-2);

Xavg = mean(X,2);                       % Compute mean
Xmean = X - Xavg*ones(1,size(X,2));
obs=Xmean;
% V=V_svd;
% for j=1: length(PCA_info_str)
%     obs=PCA_info_str{j}';

if ramp
    prev=round(PCA_info_full(beg,end-1));
    %     jetcustom = jet(9);
elseif escapeTime
    prev=PCA_info_full(1,end-2);
    %     jetcustom = jet(PCA_info_full(1,end-2)+1);
else
    prev=PCA_info_full(1,end);
    % jetcustom = jet(7);
end
jetcustom = jet(colorNum);
for i=1:size(obs,1)
    x = V(:,1)'*obs(i,:)';
    y = V(:,2)'*obs(i,:)';
    z = V(:,3)'*obs(i,:)';
    r= V(:,4)'*obs(i,:)';
    
    if ramp
        
        %             comp=round(PCA_info_full(i,end-1));
        comp = fix(PCA_info_full(i,end-1)); %just saves the non-decimal part of the number
    elseif escapeTime
        comp=PCA_info_full(i,end-2);
    else
        comp=PCA_info_full(i,end);
    end
    if  comp~= prev
        prev=comp;
        k=k+1;
    end
    
    %         plot3(x,y,z,'kx','LineWidth',2);
    
    plot3(x,y,z,'x','Color', jetcustom(comp+1,:),'LineWidth',2);
    
    %         plot3(x,y,z,'rx','LineWidth',2);
    % plot(x,y,'x','Color', jetcustom(k,:),'LineWidth',2)
    
    FullData(i,:)=[x,y,z,r];
end
% end
colormap(jetcustom);
cb = colorbar;
if ramp
    caxis([0 8])
    ylabel(cb,'time (rounded)')
elseif escapeTime
    %     caxis([-PCA_info_full(1,end-2) 0])
    ylabel(cb,'escape time')
else
    %  caxis([size(PCA_info_str)])
    ylabel(cb,'force')
end
% caxis([0 k-1])
% caxis([1 k])
caxis(axisVec)
view(155,15), grid on, set(gca,'FontSize',13)
xlabel('V1')
ylabel('V2')
zlabel('V3')

% title('RPCA-PCA(L)-X(standardized columns)')
title(titlePlot)
end