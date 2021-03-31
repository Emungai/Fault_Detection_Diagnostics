%% DBSCAN
%from
%https://stackoverflow.com/questions/43160240/how-to-plot-a-k-distance-graph-in-python
%(code with 5 likes)
fin=3;%number of dimensions
Xt=FullData(:,1:fin);
eucl=0;
city=1;
%% calculating parameters for DBSCAN
%euclidean distance
k=fin*2; %# of min points
    kn_distance = []
    for i=1:length(Xt)
        eucl_dist = [];
        for j=1:length(Xt)
            eucl_dist(j)=norm(Xt(i,:)-Xt(j,:));  %((Xt(i,1) - Xt(j,1)) ^ 2 +(Xt(i,2) - Xt(j,2)) ^ 2)^0.5;
           
        end
        eucl_dist_sorted=sort(eucl_dist);
        kn_distance(end+1)=(eucl_dist_sorted(k));
    end
%look at the elbow in the histogram to determine eps    
figure
histogram(kn_distance) 
if eucl
    title('Euclidean')
elseif city
    title('Manhattan')
else
    title('Mahalanobis')
end
eps=0.2; %euclidean distance
% eps=0.14;
min_pts=k;
[idx,corepts] = dbscan(Xt,eps,min_pts);


%mahalbonois distance
[mIdx,mD] = knnsearch(Xt,Xt,'K',6,'Distance','mahalanobis');
%look at the elbow in the histogram to determine eps
figure
histogram(mD(:,6))

%manhattan distance
[m2Idx,mD2] = knnsearch(Xt,Xt,'K',6,'Distance','cityblock');
%look at the elbow in the histogram to determine eps
figure
histogram(mD2(:,6))
min_pts=k;

if eucl
eps=0.2; %euclidean distance
[idx,corepts] = dbscan(Xt,eps,min_pts); % The default distance metric is Euclidean distance;

elseif city
    eps=0.13;
    [idx,corepts] = dbscan(Xt,eps,min_pts,'Distance','cityblock'); % The default distance metric is Euclidean distance;
else
    eps=0.12;
    [idx,corepts] = dbscan(Xt,eps,min_pts,'Distance','mahalanobis'); % The default distance metric is Euclidean distance;

end

%% graph
color = lines(length(unique(idx))); % Generate color values
% eps=0.18;
% min_pts=2;
% eps=0.15;

% eps=0.14;

num_outliers=sum(idx==-1);%idx value of -1 is the outlier, 
num_corepts=sum(corepts == 1);%corepts==1 are the core points
figure
gscatter(Xt(:,1),Xt(:,2),idx,color);
xlabel('V1')
ylabel('V2')
if eucl
title((['DBSCAN Using Euclidean Distance Metric','eps:',string(eps),'min points:',string(min_pts)]))
elseif city
   title((['DBSCAN Using Manhattan Distance Metric','eps:',string(eps),'min points:',string(min_pts)]))
 
else
    title((['DBSCAN Using Mahalanobis Distance Metric','eps:',string(eps),'min points:',string(min_pts)]))

end
%plotting 3D scatter plot
figure
hold on
for i=1:length(Xt)
    if idx(i)==-1
        plot3(Xt(i,1),Xt(i,2),Xt(i,3),'kx','LineWidth',2); %plotting outliers in black
    else
         plot3(Xt(i,1),Xt(i,2),Xt(i,3),'x','Color', color(idx(i),:),'LineWidth',2);  
    end
        
end
view(155,15), grid on, set(gca,'FontSize',13)
xlabel('V1')
ylabel('V2')
zlabel('V3')
if eucl
title((['3D DBSCAN Using Euclidean Distance Metric','eps:',string(eps),'min points:',string(min_pts)]))
elseif city
   title((['DBSCAN Using Manhattan Distance Metric','eps:',string(eps),'min points:',string(min_pts)]))

else
    title((['3D DBSCAN Using seuclidean Distance Metric','eps:',string(eps),'min points:',string(min_pts)]))

end
hold off

%% Gausian mixed models
GMModel=fitgmdist(Xt,8)
AIC= GMModel.AIC
subplot(2,2,1)
h=ezcontour(@(x1,x2)pdf(GMModel,[x1 x2]));
subplot(2,2,2)
h=ezmesh(@(x1,x2)pdf(GMModel,[x1 x2]));


%% K Means
[ind,c]=kmeans(Xt,2);
figure
plot(c(1,1),c(1,2),'k*','Linewidth',[2])
figure
plot(c(2,1),c(2,2),'k*','Linewidth',[2])
midx=(c(1,1)+c(2,1))/2; midy=(c(1,2)+c(2,2))/2;
slope=(c(2,2)-c(1,2))/(c(2,1)-c(1,1)); % rise/run
b=midy+(1/slope)*midx;
xsep=-1:0.1:2; ysep=-(1/slope)*xsep+b;
figure(1), subplot(2,2,1), hold on
plot(xsep,ysep,'k','Linewidth',[2]),axis([-2 4 -3 2])

% error on test data
figure(1), subplot(2,2,2)
plot(x(n1+1:end),y(n1+1:end),'ro'), hold on
plot(x3(n1+1:end),y3(n1+1:end),'bo')
plot(xsep,ysep,'k','Linewidth',[2]), axis([-2 4 -3 2])