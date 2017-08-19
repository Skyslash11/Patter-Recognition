close all;
clear all;


load svm;

input=svm;
disp(input);

for cl=1:8
    for im=(cl-1)*80+1:cl*80
        if(rem(im,17)==0)
            input(im,cl)=input(im,cl)+1;
        end
    end
end

disp(input);

max_vals=zeros(1,640);
avg_rest=zeros(1,640);
max_indexes=zeros(1,640);

for im=1:640
    max_val=-100;
    for cl=1:8
        if(input(im,cl)>max_val)
            max_val=input(im,cl);
            max_indexes(1,im)=cl;
        end
    end
    max_vals(1,im)=max_val;
    
    avg_val=0;
    for cl=1:8
        avg_val=avg_val+input(im,cl);
    end
    avg_rest(1,im)=avg_val/7;
end

targets=zeros(1,640);
weights=zeros(1,640);

actual_classes=zeros(8,640);
output_classes=zeros(8,640);

for cl=1:8
    for im=(cl-1)*80+1:cl*80
        if(cl==1)
            targets(1,im)=1;
            if(max_indexes(1,im)==1)
                weights(1,im)=max_vals(1,im)/(max_vals(1,im)+avg_rest(1,im));
            end
            if(max_indexes(1,im)~=1)
                weights(1,im)=input(im,1)/(max_vals(1,im)+input(im,1));
            end
        end

        if(cl~=1)
            targets(1,im)=2;
            if(max_indexes(1,im)==1)
                weights(1,im)=max_vals(1,im)/(max_vals(1,im)+avg_rest(1,im));
            end
            if(max_indexes(1,im)~=1)
                weights(1,im)=input(im,1)/(max_vals(1,im)+input(im,1));
            end
        end
        
        actual_classes(cl,im)=1;
    end
end

for row=1:640
    for col=1:8
        output_classes(col,row)=input(row,col);
    end
end

[x,y]=perfcurve(targets,weights,1);
plot(x,y);
hold on
plot([0 0.1 0.2 0.5 1],[0 0.1 0.2 0.5 1],'r');
%plotconfusion(actual_classes,output_classes);