m = length(lab);
finaltrainlab = zeros(m*ratio,1);
for i = 1:m*ratio
    x = X_train(i,:);   
    hidein = W1*x'+B1;%隐含层输入值
    hideout = zeros(hidesize,1);%隐含层输出值
    for j = 1:hidesize
        hideout(j) = sigmod(hidein(j));
    end    
    yin = W2*hideout+B2;%输出层输入值
    yout = zeros(outsize,1);
    for j = 1:outsize
        yout(j) = sigmod(yin(j));
    end 
    [~,index]=max(yout);
    finaltrainlab(i) = index;
end


finaltestlab = zeros(m*(1-ratio),1);
for i = 1:m*(1-ratio)
    x = X_test(i,:);   
    hidein = W1*x'+B1;%隐含层输入值
    hideout = zeros(hidesize,1);%隐含层输出值
    for j = 1:hidesize
        hideout(j) = sigmod(hidein(j));
    end    
    yin = W2*hideout+B2;%输出层输入值
    yout = zeros(outsize,1);
    for j = 1:outsize
        yout(j) = sigmod(yin(j));
    end 
    [~,index]=max(yout);
    finaltestlab(i) = index;
end



location1 = find(y_train==finaltrainlab); % 找出预测正确的样本的位置
accuracy1 = length(location1)/length(y_train); % 计算预测精度

location2 = find(y_test==finaltestlab); % 找出预测正确的样本的位置
accuracy2 = length(location2)/length(y_test); % 计算预测精度
