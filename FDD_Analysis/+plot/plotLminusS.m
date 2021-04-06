function plotLminusS(plotInfo)
%plotLminusS plots the L minus S from RPCA

x_subplot=plotInfo.x_subplot;
y_subplot=plotInfo.y_subplot;
task=plotInfo.task;
feat=plotInfo.feat;
L=plotInfo.L;
S=plotInfo.S;
t=plotInfo.t;
colorNum=plotInfo.colorNum;
jetcustom = jet(colorNum);
feat={'lcom_y','lstance_y','v_sw_x','v_sw_z','p_sw_z','p_dsw_x','torso_angle','com_height','p_st_z','CoM_rel_p_legs'};
X=L-S;
figure

for j=1:length(feat)
    subplot(x_subplot,y_subplot,j)
    hold on
    x=X(:,j);
for i=1:size(X,1)
    plot(t(i),x(i), 'x', 'Color',jetcustom(task(i)+1,:),'LineWidth',2);
end
 title(feat{j})
hold off
end
sgtitle('L minus S')
% figure
% for i=1:10
%     subplot(2,5,i)
%     if ramp
%         t=Data.p_com.Time;
%         plot(t,L_O(:,i)-S_O(:,i));
%         hold on
%         plot(t,zeros(size(L_O(:,i))),'k','LineWidth',2);
%         
%     else
%         
%         fst=0;
%         if window
%             t_act=5509;
%             jetcustom = jet(2);
%             plot([t(1:t_act)],L_O(1:t_act,i)-S_O(1:t_act,i),'Color',jetcustom(1,:));
%             hold on
%             plot([t(t_act+1:end)],L_O(t_act+1:end,i)-S_O(t_act+1:end,i),'Color',jetcustom(2,:));
%             hold on
%             plot([t],zeros(size(L_O(:,i))),'k','LineWidth',2);
%             
%         else
%             for j=1:length(PCA_info_str)
%                 
%                 snd=length(PCA_info_str{j});
%                 plot([fst+1:fst+snd],L_O(fst+1:snd+fst,i)-S_O(fst+1:snd+fst,i),'Color',jetcustom(j,:));
%                 hold on
%                 fst=fst+snd;
%                  plot(zeros(size(L_O(:,i))),'k','LineWidth',2);
%             end
%         end
%         
%     end
%     
%     title(feat{i})
% end
% sgtitle('L minus S')
% legend('nominal','-3000 knee')
% end
% 
