fileID = fopen('ggwp1','r');
sizeInput=[23 11088];
A=fscanf(fileID,'%f',sizeInput);
disp(size(A,2));

input=zeros(11088,23);

for a=1:11088
    for b=1:23
        input(a,b)=A(b,a);
    end
end

idx=kmeans(input,5);



%disp(idx);
s1=0;
s2=0;
s3=0;
s4=0;
s5=0;
c1=1;
c2=1;
c3=1;
c4=1;
c5=1;
for a = 1:11088
    if idx(a,1)==1
        s1=s1+1;
    end
    if idx(a,1)==2
        s2=s2+1;
    end
    if idx(a,1)==3
        s3=s3+1;
    end
    if idx(a,1)==4
        s4=s4+1;
    end
    if idx(a,1)==5
        s5=s5+1;
    end
end
cluster1=zeros(s1,23);
cluster2=zeros(s2,23);
cluster3=zeros(s3,23);
cluster4=zeros(s4,23);
cluster5=zeros(s5,23);
for a = 1:11088
    if idx(a,1)==1
        cluster1(c1,:)=input(a,:);
        c1=c1+1;
    end
    if idx(a,1)==2
        cluster2(c2,:)=input(a,:);
        c2=c2+1;
    end
    if idx(a,1)==3
        cluster3(c3,:)=input(a,:);
        c3=c3+1;
    end
    if idx(a,1)==4
        cluster4(c4,:)=input(a,:);
        c4=c4+1;
    end
    if idx(a,1)==5
        cluster5(c5,:)=input(a,:);
        c5=c5+1;
    end
end
%display(cluster1);
mean1=mean(cluster1);
cov1=cov(cluster1);
mean2=mean(cluster2);
cov2=cov(cluster2);
mean3=mean(cluster3);
cov3=cov(cluster3);
mean4=mean(cluster4);
cov4=cov(cluster4);
mean5=mean(cluster5);
cov5=cov(cluster5);
display(size(cov1));

p1=s1/(s1+s2+s3+s4+s5);
p2=s2/(s1+s2+s3+s4+s5);
p3=s3/(s1+s2+s3+s4+s5);
p4=s4/(s1+s2+s3+s4+s5);
p5=s5/(s1+s2+s3+s4+s5);
display(p5);

gamma=zeros(11088,5);
mu1=zeros(1,23);
mu2=zeros(1,23);
mu3=zeros(1,23);
mu4=zeros(1,23);
mu5=zeros(1,23);

sigma1=zeros(23,23);
sigma2=zeros(23,23);
sigma3=zeros(23,23);
sigma4=zeros(23,23);
sigma5=zeros(23,23);

display(s1);
display(s2);
display(s3);
display(s4);
display(s5);

for a = 1:10
    %E Step
    for b = 1:11088
        pd1=mvnpdf(input(b,:),mean1,cov1);
        pd2=mvnpdf(input(b,:),mean2,cov2);
        pd3=mvnpdf(input(b,:),mean3,cov3);
        pd4=mvnpdf(input(b,:),mean4,cov4);
        pd5=mvnpdf(input(b,:),mean5,cov5);
        gamma(b,1)=(p1*pd1)/(p1*pd1+p2*pd2+p3*pd3+p4*pd4+p5*pd5);
        gamma(b,2)=(p2*pd2)/(p1*pd1+p2*pd2+p3*pd3+p4*pd4+p5*pd5);
        gamma(b,3)=(p3*pd3)/(p1*pd1+p2*pd2+p3*pd3+p4*pd4+p5*pd5);
        gamma(b,4)=(p4*pd4)/(p1*pd1+p2*pd2+p3*pd3+p4*pd4+p5*pd5);
        gamma(b,5)=(p5*pd5)/(p1*pd1+p2*pd2+p3*pd3+p4*pd4+p5*pd5);
    end
    disp(gamma(1,1));
    disp(gamma(1,2));
    disp(gamma(1,3));
    disp(gamma(1,4));
    disp(gamma(1,5));
    %M Step
    tempmu1=zeros(1,23);
    tempmu2=zeros(1,23);
    tempmu3=zeros(1,23);
    tempmu4=zeros(1,23);
    tempmu5=zeros(1,23);
    
    tempsi1=zeros(23,23);
    tempsi2=zeros(23,23);
    tempsi3=zeros(23,23);
    tempsi4=zeros(23,23);
    tempsi5=zeros(23,23);
    for b = 1:11088
        tempmu1(:,:)=tempmu1(:,:)+(gamma(b,1)*input(b,:));
        tempmu2(:,:)=tempmu2(:,:)+(gamma(b,2)*input(b,:));
        tempmu3(:,:)=tempmu3(:,:)+(gamma(b,3)*input(b,:));
        tempmu4(:,:)=tempmu4(:,:)+(gamma(b,4)*input(b,:));
        tempmu5(:,:)=tempmu5(:,:)+(gamma(b,5)*input(b,:));
    end
    mean1(1:1,1:23)=tempmu1(1:1,1:23)/s1;
    mean2(1:1,1:23)=tempmu1(1:1,1:23)/s2;
    mean3(1:1,1:23)=tempmu1(1:1,1:23)/s3;
    mean4(1:1,1:23)=tempmu1(1:1,1:23)/s4;
    mean5(1:1,1:23)=tempmu1(1:1,1:23)/s5;
    disp(size(mean1));
    disp(mean1);
    disp(mean2);
    disp(mean3);
    disp(mean4);
    disp(mean5);
    for b = 1:11088
        tempsi1(:,:)=tempsi1(:,:)+gamma(b,1)*transpose(input(b,:)-mean1(:,:))*(input(b,:)-mean1(:,:));
        tempsi2(:,:)=tempsi2(:,:)+gamma(b,2)*transpose(input(b,:)-mean2(:,:))*(input(b,:)-mean2(:,:));
        tempsi3(:,:)=tempsi3(:,:)+gamma(b,3)*transpose(input(b,:)-mean3(:,:))*(input(b,:)-mean3(:,:));
        tempsi4(:,:)=tempsi4(:,:)+gamma(b,4)*transpose(input(b,:)-mean4(:,:))*(input(b,:)-mean4(:,:));
        tempsi5(:,:)=tempsi5(:,:)+gamma(b,5)*transpose(input(b,:)-mean5(:,:))*(input(b,:)-mean5(:,:));
    end
    %disp(tempsi1);
    cov1(1:23,1:23)=tempsi1(1:23,1:23)/s1;
    cov2(1:23,1:23)=tempsi2(1:23,1:23)/s2;
    cov3(1:23,1:23)=tempsi3(1:23,1:23)/s3;
    cov4(1:23,1:23)=tempsi4(1:23,1:23)/s4;
    cov5(1:23,1:23)=tempsi5(1:23,1:23)/s5;
    %disp(cov1);
    s1=0;
    s2=0;
    s3=0;
    s4=0;
    s5=0;
    
    for b=1:11088
        s1=s1+gamma(b,1);
        %disp(gamma(b,1));
        s2=s2+gamma(b,2);
        s3=s3+gamma(b,3);
        s4=s4+gamma(b,4);
        s5=s5+gamma(b,5);
    end
    
    p1=s1/11088;
    p2=s2/11088;
    p3=s3/11088;
    p4=s4/11088;
    p5=s5/11088;
end
disp(s1);
disp(s2);
disp(s3);
disp(s4);
disp(s5);
display(size(cov1));
display(size(cov2));
display(size(cov3));
display(size(cov4));
display(size(cov5));

for a=1:23
    for b=1:23
        if(a==b)
        else
            cov1(a,b)=0;
            cov2(a,b)=0;
            cov3(a,b)=0;
            cov4(a,b)=0;
            cov5(a,b)=0;
        end
    end
end


test=zeros(10800,23);

fileID1 = fopen('ggwp1','r');
sizeInput1=[23 11088];
gg1t=fscanf(fileID1,'%f',sizeInput1);

fileID2 = fopen('ggwp2','r');
sizeInput2=[23 11088];
gg2t=fscanf(fileID2,'%f',sizeInput2);

fileID3 = fopen('ggwp3','r');
sizeInput3=[23 11088];
gg3t=fscanf(fileID3,'%f',sizeInput3);

gg1=zeros(11088,23);
gg2=zeros(11088,23);
gg3=zeros(11088,23);
for a=1:11088
    for b=1:23
        gg1(a,b)=gg1t(b,a);
        gg2(a,b)=gg2t(b,a);
        gg3(a,b)=gg3t(b,a);
    end
end


test(1:3600,:)=gg1(7489:11088,:);
test(3601:7200,:)=gg2(7489:11088,:);
test(7201:10800,:)=gg3(7489:11088,:);

%disp(test);

result=zeros(300,1);
count=1;
a=1;


while(true)
    t=0;
    for b=a:a+35
        t=+p1*mvnpdf(test(a,:),mean1,cov1);
        t=t+p2*mvnpdf(test(a,:),mean2,cov2);
        t=t+p3*mvnpdf(test(a,:),mean3,cov3);
        t=t+p4*mvnpdf(test(a,:),mean4,cov4);
        t=t+p5*mvnpdf(test(a,:),mean5,cov5);
        
    end
    result(count,1)=result(count,1)+log(t);
    count=count+1;
    a=a+36;
    if a>10800
        break
    end
end
display(result);
%save('oot1',result);