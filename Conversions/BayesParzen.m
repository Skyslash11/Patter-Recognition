close all;
clear all;

load train_class1.txt;
load train_class2.txt;
load train_class3.txt;
load train_class4.txt;
load train_class5.txt;
load train_class6.txt;
load train_class7.txt;
load train_class8.txt;

load test_class1.txt;
load test_class2.txt;
load test_class3.txt;
load test_class4.txt;
load test_class5.txt;
load test_class6.txt;
load test_class7.txt;
load test_class8.txt;

train=cell(1,8);
test=cell(1,8);

train{1,1}=train_class1;
train{1,2}=train_class2;
train{1,3}=train_class3;
train{1,4}=train_class4;
train{1,5}=train_class5;
train{1,6}=train_class6;
train{1,7}=train_class7;
train{1,8}=train_class8;

test{1,1}=test_class1;
test{1,2}=test_class2;
test{1,3}=test_class3;
test{1,4}=test_class4;
test{1,5}=test_class5;
test{1,6}=test_class6;
test{1,7}=test_class7;
test{1,8}=test_class8;

num_images=size(test{1,1},1)/36;

total_likelihood=0;
h=1;
points_likelihood=zeros(800,8);
    for pts=1:800
        for cls=1:8
            sum=0;
            temp=floor((pts-1)/100+1);
            disp(temp);
            for b=1:500
                const=2*pi*h*h;
                const=sqrt(const);
                
                diff=test{1,temp}(rem(pts-1,100)+1,:)-train{1,cls}(b,:);
                power=-((norm(diff))^2)/(2*h*h);
                sum=sum+((1/const)*exp(power));
            end
            points_likelihood(pts,cls)=sum;
            num1=rem(pts,11);
            num2=rem(pts,13);
            num3=rem(pts,7);
            num4=rem(pts,17);
            num5=rem(pts,10);
            num6=rem(pts,8);
            
        end
    end
image_likelihood=zeros(800,8);

disp(points_likelihood);

ac=zeros(1,800);
bc=zeros(1,800);
actual_classes=zeros(8,800);
output_classes=zeros(8,800);
max=zeros(1,800);

for pts=1:800
    max_val=-1000;
    max_ind=1;
    for cls=1:8
        output_classes(cls,pts)=points_likelihood(pts,cls);
        if(max_val<points_likelihood(pts,cls))
            max_val=points_likelihood(pts,cls);
            max_ind=cls;
        end
    end
    %output_classes(max_ind,pts)=1;
    max(1,pts)=max_val;
    bc(1,pts)=points_likelihood(pts,floor((pts-1)/100)+1);
    disp(max_ind);
end


for i=1:8
    for b=(i-1)*100+1:i*100
        actual_classes(i,b)=1;
        ac(1,b)=i;
    end
end


[X,Y]=perfcurve(ac,bc,1);
%disp(size(X,1));
%disp(Y);
figure
%plot(Y,X,'b');
%plot(X,Y,'b');
%title('ROC PLOT');
%xlabel('False Positive Rate');
%ylabel('True Positive rate');
%hold on
%plot([0 0.25 0.5 1],[0 0.25 0.5 1],'r');
%disp(total_likelihood);
plotconfusion(actual_classes,output_classes);
disp(points_likelihood);
