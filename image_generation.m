clc;
clear;

%% �������ݼ�

%% ��ʼת��ͼƬ
load('acc.mat');
acc=acc;
for i=1:length(acc)
       x = 1:length(acc{i}(:,1)); 
       v = acc{i}(:,1);
       xq = 1:1:130;
       sg{i}(:,1) = interp1(x,v,xq,'spline')';
    clear x v xq
end

sg_s=reshape(sg,280,6); 
%str={'morl','mexh','db3','coif3','fk4','haar','sym3','bior3.3'};
for j=1:size(sg_s,2)%���
for i=1:size(sg_s,1)%������
  chan_acc=sg_s{i,j}(:,1);
  fig=figure;
  set(gcf, 'unit', 'centimeters', 'position', [10 5 0.1 1]);
  set(fig,'color','w');
  w=cwt(chan_acc,[1:1:21],'mexh','plot'); %morl,mexh,db3,coif3,fk4,haar,sym3,bior3.3,
  
  title('');
  axis off 
  set(gca,'looseInset',[0 0 0 0]);
  set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w')
  
%   xlabel('ʱ��'); 
%   ylabel('�任�߶�'); 
%   title('��Ӧ�ڳ߶�a=12.12,10.24,15.48,1.2,2,4,6,8,10С���任ϵ���ľ���ֵ');
  cd('C:\Users\CRY\Desktop\ticao\other\acc\');% ��sym����cmor����mexh������gaus������bior��
  dirName = num2str(j);
  mkdir (dirName);
  cd(['C:\Users\CRY\Desktop\ticao\other\acc\',num2str(j)]);
  %saveas(gca,'meanshape.bmp','bmp');
  saveas(gca,sprintf('%d.png',i));
  clear chan_acc s;
  close all;
end

end

