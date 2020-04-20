clc;
clear;

Sigma = [1, 0; 0, 1]; 
mu1 = [1, -1]; 
x1 = mvnrnd(mu1, Sigma, 200); 
lab1 = ones(200,1);
mu2 = [5, -4]; 
x2 = mvnrnd(mu2, Sigma, 200); 
lab2 = 2*ones(200,1);
mu3 = [1, 4]; 
x3 = mvnrnd(mu3, Sigma, 200); 
lab3 = 3*ones(200,1);
mu4 = [6, 4]; 
x4 = mvnrnd(mu4, Sigma, 200); 
lab4 = 4*ones(200,1);
mu5 = [7, 0.0]; 
x5 = mvnrnd(mu5, Sigma, 200); 
lab5 = 5*ones(200,1);

% Show the data points 
figure;
plot(x1(:,1), x1(:,2), 'r.'); 
hold on; 
plot(x2(:,1), x2(:,2), 'b.'); 
hold on; 
plot(x3(:,1), x3(:,2), 'k.'); 
hold on; 
plot(x4(:,1), x4(:,2), 'g.'); 
hold on; 
plot(x5(:,1), x5(:,2), 'm.'); 
grid on;

% data = [x1;x2;x3;x4;x5];
lab = [lab1;lab2;lab3;lab4;lab5];

data = [x1;x2;x3;x4;x5];
% temp = zeros(1000,8);
% data = [temp data0];

% [input,minI,maxI] = premnmx(data);


m = length(lab);
k = 5;
ratio = 0.6;
trainlab = zeros(m*ratio,k);
testlab = zeros(m*(1-ratio),k);

[X_train, y_train,  X_test, y_test] = divide(data, lab, k, ratio);

for i=1:(m*ratio)   
    switch y_train(i)        
        case 1            
            trainlab(i,:)=[1 0 0 0 0];        
        case 2            
            trainlab(i,:)=[0 1 0 0 0];        
        case 3            
            trainlab(i,:)=[0 0 1 0 0];        
        case 4            
            trainlab(i,:)=[0 0 0 1 0];
        case 5            
            trainlab(i,:)=[0 0 0 0 1];   
    end
end

for i=1:(m*(1-ratio)) 
    switch y_test(i)        
        case 1            
            testlab(i,:)=[1 0 0 0 0];        
        case 2            
            testlab(i,:)=[0 1 0 0 0];        
        case 3            
            testlab(i,:)=[0 0 1 0 0];        
        case 4            
            testlab(i,:)=[0 0 0 1 0];
        case 5            
            testlab(i,:)=[0 0 0 0 1];   
    end
end

insize = 2;%�������Ԫ��Ŀ
hidesize = 10;%��������Ԫ��Ŀ
outsize = 5;%�������Ԫ��Ŀ
 
yita1 = 0.001;%����㵽������֮���ѧϰ��
yita2 = 0.001;%�����㵽�����֮���ѧϰ��
 
W1 = rand(hidesize,insize);%����㵽������֮���Ȩ��
W2 = rand(outsize,hidesize);%�����㵽�����֮���Ȩ��
B1 = rand(hidesize,1);%��������Ԫ����ֵ
B2 = rand(outsize,1);%�������Ԫ����ֵ
 
%������� trainlab

loop = 2000;
E = zeros(1,loop);
% X_train = X_train';
for loopi = 1:loop
    %ѵ������
    for i = 1:600
        x = X_train(i,:);%��������
        
        hidein = W1*x'+B1;%����������ֵ
        hideout = zeros(hidesize,1);%�������������ֵ
        for j = 1:hidesize
              hideout(j) = sigmod(hidein(j));
        end
        
        yin = W2*hideout+B2;%���������ֵ
        yout = zeros(outsize,1);%��������ֵ
        for j = 1:outsize
              yout(j) = sigmod(yin(j));
        end
        
        e = yout  - (trainlab(i,:))';%ѵ��������   �������������
        E(loopi) = sum(e);

        %������
        dB2 = zeros(outsize,1);%�����������ֵ��ƫ����������ֵ�仯��
        for j = 1:outsize
            dB2 = sigmod(yin(j))*(1-sigmod(yin(j)))*e(j)*yita2;
        end
        
        %�������������֮���Ȩ�صı仯��
        dW2 = zeros(outsize,hidesize);
        for j = 1:outsize
            for k = 1:hidesize
                dW2(j,k) = sigmod(yin(j))*(1-sigmod(yin(j)))*hideout(k)*e(j)*yita2;
            end
        end
        
        %��������ֵ�仯��
        dB1 = zeros(hidesize,1);
        for j = 1:hidesize
            tempsum = 0;
            for k = 1:outsize
                 tempsum = tempsum + sigmod(yin(k))*(1-sigmod(yin(k)))*W2(k,j)*sigmod(hidein(j))*(1-sigmod(hidein(j)))*e(k)*yita1;
            end
            dB1(j) = tempsum;
        end
        
        %����㵽�������Ȩ�ر仯��
        dW1 = zeros(hidesize,insize);
        for j = 1:hidesize
            for k = 1:insize
               tempsum = 0;
               for m = 1:outsize
                   tempsum = tempsum + sigmod(yin(m))*(1-sigmod(yin(m)))*W2(m,j)*sigmod(hidein(j))*(1-sigmod(hidein(j)))*x(k)*e(m)*yita1;
               end               
               dW1(j,k) = tempsum;               
            end            
        end        
        W1 = W1-dW1;
        W2 = W2-dW2;
        B1 = B1-dB1;
        B2 = B2-dB2;       
    end   
    if mod(loopi,100)==0
        loopi;
    end    
end


