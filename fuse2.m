clc;
clear;
close all;

for i=1:280
    for j=1:6
    cd(['C:\Users\CRY\Desktop\纹理TIM\image\cwtx\',num2str(j)]);
    img1=imread(sprintf("%d.png", i));
    cd(['C:\Users\CRY\Desktop\纹理TIM\image\cwty\',num2str(j)]);
    img2=imread(sprintf("%d.png", i));
    cd(['C:\Users\CRY\Desktop\纹理TIM\image\cwtz\',num2str(j)]);
    img3=imread(sprintf("%d.png", i));
    
    result= cat(1, img1, img2, img3);

    cd('C:\Users\CRY\Desktop\纹理TIM\image\cwt2\');
  dirName = num2str(j);
  mkdir (dirName);
  cd(['C:\Users\CRY\Desktop\纹理TIM\image\cwt2\',num2str(j)]);
  imwrite(result,sprintf('%d.png',i));
    end
end
