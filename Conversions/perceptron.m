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

wt=zeros(8,23);

for a=1:8
        wt(a,:)=mean(train{1,a});
end

train_size=6400;
%disp(train_size);

for counter=1:7
    for a=1:train_size
        vals=zeros(8,8);
        for b=1:8
            for c=1:8
                vals(b,c)=wt(c,:)*transpose(train{1,b}(a,:));                
            end
        end
        max_val=zeros(1,8);
        max_index=zeros(1,8);
        for b=1:8
            max_val(1,b)=max([vals(b,1) vals(b,2) vals(b,3) vals(b,4) vals(b,5) vals(b,6) vals(b,7) vals(b,8)]);
        end
        for b=1:8
            for c=1:8
                if(max_val(1,b)==vals(b,c))
                    max_index(1,b)=c;
                end
            end
        end
        
        for b=1:8
            if(max_index(1,b)~=b)
                wrong_index=max_index(1,b);
                wt(b,:)=wt(b,:)+train{1,b}(a,:);
                wt(wrong_index,:)=wt(wrong_index,:)-train{1,b}(a,:);
            end
        end
    end
end

num_images=size(test{1,1},1)/36;
ans=zeros(1,2);
%disp(num_images);
likelihood=zeros(640,8);
for a=1:num_images
    
    image_classes=zeros(8,8);
    classification=zeros(1,8);
    points_elong=zeros(8,8);
    for b=(a-1)*36+1:a*36
        vals=zeros(8,8);
        for d=1:8
            for c=1:8
                vals(d,c)=wt(c,:)*transpose(test{1,d}(b,:));
            end
        end
        max_val=zeros(1,8);
        max_index=zeros(1,8);
        
        for c=1:8
            max_val(1,c)=max([vals(c,1) vals(c,2) vals(c,3) vals(c,4) vals(c,5) vals(c,6) vals(c,7) vals(c,8)]);
        end
        
        for d=1:8
            for c=1:8
                if(max_val(1,d)==vals(d,c))
                    max_index(1,d)=c;
                end
            end
        end
        
        %disp(max_val);
        for c=1:8
            image_classes(c,max_index(1,c))=image_classes(c,max_index(1,c))+1;
        end
        
        mx=zeros(1,8);
        for c=1:8
           mx(1,c)= max([image_classes(c,1) image_classes(c,2) image_classes(c,3) image_classes(c,4) image_classes(c,5) image_classes(c,6) image_classes(c,7) image_classes(c,8)]);
        end
        
        for c=1:8
            for d=1:8
                if(mx(1,c)==image_classes(c,d))
                    classification(1,c)=d;
                end
            end
        end
        
        
    end
    for d=1:8
        if(classification(1,d)==d)
            ans(1,1)=ans(1,1)+1;
        end
        if(classification(1,d)~=d)
            ans(1,2)=ans(1,2)+1;
        end
    end
    %disp(image_classes);
    for i=1:8
        likelihood((i-1)*80+a,:)=image_classes(i,:);
    end
end

output_classes=zeros(2,640);
actual_classes=zeros(2,640);
oclass=zeros(8,640);
aclass=zeros(8,640);
ac=zeros(1,640);
bc=zeros(1,640);
cc=zeros(1,640);
dd=zeros(1,640);

act_roc=zeros(2,640);
out_roc=zeros(2,640);
for i=1:8
    for b=(i-1)*80+1:80*i
        %actual_classes(1,b)=1;
        max_val=0;
        max_ind=0;
        other_average=0;
        total=0;
        for j=1:8
            total=total+likelihood(b,j);
            oclass(j,b)=likelihood(b,j);
            if(max_val<likelihood(b,j))
                max_val=likelihood(b,j);
                max_ind=j;
            end
        end
        aclass(max_ind,b)=1;
        if(max_ind==i)
            for t=1:8
                if(t~=i)
                    other_average=other_average+likelihood(b,t);
                end
            end
            other_average=other_average/7;
            output_classes(1,b)=max_val/(max_val+other_average);
            output_classes(2,b)=other_average/(max_val+other_average);
            %output_classes(1,b)=1;
            %output_classes(2,b)=0;
            %disp('hello');
            ac(1,b)=1;
            bc(1,b)=likelihood(b,i)/(max_val+likelihood(b,i));
        end
        if(max_ind~=i)
            other_average=likelihood(b,i);
            output_classes(2,b)=max_val/(max_val+likelihood(b,i));
            output_classes(1,b)=likelihood(b,i)/(max_val+likelihood(b,i));
            %output_classes(2,b)=1;
            %output_classes(1,b)=0;
            ac(1,b)=0;
            bc(1,b)=likelihood(b,i)/(max_val+likelihood(b,i));
        end
        dd(1,b)=max_ind;
        
            if(i==max_ind)
                out_roc(1,b)=max_val/(max_val+(total/7));
                out_roc(2,b)=(total/7)/(max_val+(total/7));
            end
            if(i~=max_ind)
                out_roc(1,b)=likelihood(b,i)/(max_val+likelihood(b,i));
                out_roc(2,b)=max_val/(max_val+likelihood(b,i));
            end
            act_roc(1,b)=1;
                
    end
end
%disp(output_classes(1:2,81:86));
%disp(actual_classes(1:2,81:86));
%plotroc(actual_classes(1:2,81:86),output_classes(1:2,81:86));
disp(out_roc);
for i=1:8
    for b=(i-1)*80+1:i*80
        ac(1,b)=i;
        aclass(i,b)=i;
    end
end
%[X,Y]=perfcurve(ac,bc,cc);
%plot(X,Y);
%[X,Y]=perfcurve(ac,bc,1);
%disp(size(X,1));
%disp(Y);
%figure
%plot(Y,X,'b');
%plot(Y,X,'b');
%title('ROC PLOT');
%xlabel('False Positive Rate');
%ylabel('True Positive rate');
%hold on
%plot([0 0.25 0.5 1],[0 0.25 0.5 1],'r');
plotconfusion(aclass,oclass);
disp(transpose(oclass));

%plotconfusion(ra,ro);