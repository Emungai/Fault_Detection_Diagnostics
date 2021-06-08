function plotPCA(plotInfo)
%% getting variables
X=plotInfo.X;
PCA_info_full=plotInfo.PCA_info_full;
ramp=plotInfo.ramp; %use rounded time for legend
escapeTime=plotInfo.escapeTime; %use escapeTime for legend
colorNum=plotInfo.colorNum; %number of colors for the map
V=plotInfo.V; %principal components
axisVec=plotInfo.axisVec; %legend entry for caxis
titlePlot=plotInfo.titlePlot;
digit=plotInfo.digit;
fivelink=plotInfo.fivelink;
FullData=plotInfo.FullData;
%%
figure, hold on
k=1;
beg=1;



% Xavg = mean(X,2);                       % Compute mean
% Xmean = X - Xavg*ones(1,size(X,2));
% obs=Xmean;


if ramp
    if fivelink
        prev=round(PCA_info_full(beg,end-1));
    elseif digit
        prev=fix(PCA_info_full(beg,end));
    end
    
elseif escapeTime
    if fivelink
        prev=PCA_info_full(1,end-2);
    elseif digit
        prev=PCA_info_full(1,end);
    end
    
else
    prev=PCA_info_full(1,end);
    
end
jetcustom = jet(colorNum);
for i=1:size(FullData,1)
    
    x=FullData(i,1);
    y=FullData(i,2);
    z=FullData(i,3);
    if ramp
        
        %             comp=round(PCA_info_full(i,end-1));
        if fivelink
            comp = fix(PCA_info_full(i,end-1)); %just saves the non-decimal part of the number
        elseif digit
            comp = fix(PCA_info_full(i,end)); %just saves the non-decimal part of the number
        end
    elseif escapeTime
        if fivelink
            comp=PCA_info_full(i,end-2);
        elseif digit
            comp=PCA_info_full(i,end);
        end
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
    
    %     FullData(i,:)=[x,y,z,r];
end
% end
colormap(jetcustom);
cb = colorbar;
if ramp
    %     caxis([0 8])
    ylabel(cb,'time ')
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
