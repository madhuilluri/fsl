function accuracy = testmnist
% function testmnist.m = test mnist digit dataset
% required files:
%   t10k-labels-idx1-ubyte
%   t10k-images-idx3-ubyte
% files can be obtained from: http://yann.lecun.com/exdb/mnist/

% TJ Keemon, AI digit recognition project, May 2009.

Ls = readmnist('t10k-labels-idx1-ubyte');
Is = readmnist('t10k-images-idx3-ubyte');

%Is = normalizeDigits(Is);
% training on the first 1000 digits in the dataset

ntrain = 5000;

neigen = 120;

% %disp('generating feature vectors for the training data');
% %tic; V = getFeatureVector(Is(:,:,1:ntrain)); t = toc;
% %disp(['finished in ' num2str(t)]);

V = [];
disp('running pca on training images');
tic; [dMat C S] = trainPCA(Is(:,:,1:ntrain),V,neigen); 

%rmse = sqrt(mean( (dMat(:,1)-mad(:,2).^2 ));
%disp(rmse)
%imagesc(S)
%disp(['generating feature vector for the test data']);
%tic; V = getFeatureVector(Is(:,:,ntrain+1:end)); t = toc;
%disp(['finished in ' num2str(t)]);
class = classifyDigits(Is(:,:,ntrain+1:end),C,S,V,neigen);



%disp(dMat)
%imagesc(C)

%imagesc(V)
% check accuracy and generate confusion matrix
accuracy = zeros(10,1);
totals = zeros(10,1);
confMatrix = zeros(10);
ntest = length(class);


for i = 1:ntest
    pred = Ls(class(i));
    label = Ls(ntrain+i);
    
    accuracy(label+1) = accuracy(label+1) + (pred==label);
    totals(label+1) = totals(label+1) + 1;
    
    confMatrix(pred+1,label+1) = confMatrix(pred+1,label+1) + 1;
end
accuracy = accuracy ./ totals;

t = toc;
disp(['total time ' num2str(t)]);

%imagesc(accuracy)
disp ('confusion matrix')
%disp(accuracy);
disp(confMatrix);

disp('mean ')
disp(mean(accuracy))