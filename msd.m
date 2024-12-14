clc;
clear;

%% Maximum scale distance criterion
load('select.mat');
sg_s=select;
sp=240;
str={'morl','mexh','db3','coif3','fk4','haar','sym3','bior3.3'};

for scal=1:10:151  %1:10:151
for para=1:length(str)
wd = cell(6,1);
wd(:)={0};
for j=1:size(sg_s,2)%类别
for i=1:size(sg_s,1)%样本数
  chan_acc=sg_s{i,j}(:,1);
  w=cwt(chan_acc,[1:scal],str{para}); %morl,mexh,db3,coif3,fk4,haar,sym3,bior3.3,
  wd{j}=wd{j}+w/240;
  clear chan_acc s;
end
end

w1 = wd{1};
w2 = wd{2};
w3 = wd{3};
w4 = wd{4};
w5 = wd{5};
w6 = wd{6};
sm=1;
for i = 1 : 6
    for j = i : 6
            dist_ij = norm(eval(['w' num2str(i) '(:)']) - eval(['w' num2str(j) '(:)'])); % 计算尺度距离
            dist(sm) = dist_ij;
            sm=sm+1;
    end
end
dr(para,scal)=sum(dist)/(size(w,1)*size(w,2));
end
end

ds=dr;
idx_nonzerolines = sum(abs(ds),1)>0 ;
sd= ds(:,idx_nonzerolines) ;
sr=sd';
plot(sd');
legend(str);