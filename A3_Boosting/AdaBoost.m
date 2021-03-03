%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 150;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 2000;
% Number of weak classifiers
nbrWeakClassifiers = 60;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError

D = 1/nbrTrainImages .* ones(1,nbrTrainImages);
nrthresh = nbrTrainImages;
H = zeros(1,nbrTrainImages);
opt_h = zeros(3,nbrWeakClassifiers);
opt_alpha = zeros(1, nbrWeakClassifiers);

for c = 1:nbrWeakClassifiers
    Emin = inf;
    for k = 1:nbrHaarFeatures
        for i = 1:nrthresh
            
            T = xTrain(k,i);
            P = 1;
            C = WeakClassifier(T, P, xTrain(k,:));
           
%            figure(10001)
%     plot(linspace(1, nbrTrainImages, nbrTrainImages), D); hold on;
            E = WeakClassifierError(C, D, yTrain);

            % Change polarity
            if(E > 0.5)
                E = 1 - E;
                P = P * -1;
            end

            % New better feature for weak classifier
            if(E < Emin)
                Emin = E;
                thresh = T;
                pol = P;
                feature_idx = k;
            end

         end
    end

	% Update final calssifier
    alpha = 0.5*log((1-Emin)/Emin);
    h = WeakClassifier(thresh, pol, xTrain(feature_idx,:));
    H = H + alpha * h;
    opt_h(1:3,c) = [thresh, pol, feature_idx];
    opt_alpha(c) = alpha;
    
    % Update weights    
    D = D.*exp(-alpha*(yTrain.*h));
    D = D/sum(D);
end



%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

H = sign(H);
trainAcc = (sum(H == yTrain))/size(yTrain, 2);


test_H = zeros(1,size(xTest,2));
% test_err = zeros(1, nbrWeakClassifiers);
test_acc = zeros(1, nbrWeakClassifiers);

train_H = zeros(1,size(xTrain,2));
% train_err = zeros(1, nbrWeakClassifiers);
train_acc = zeros(1, nbrWeakClassifiers);
trainError =  1 - train_acc;

for k = 1:nbrWeakClassifiers
    for c = 1:k
        t = opt_h(1,c);
        p = opt_h(2,c);
        i = opt_h(3,c);
        test_h = WeakClassifier(t, p, xTest(i,:));
        train_h = WeakClassifier(t, p, xTrain(i,:));
        a = opt_alpha(c);
        test_H = test_H + a .* test_h;
        train_H = train_H + a .* train_h;
    end

    test_H = sign(test_H);
    test_acc(k) = (sum(test_H == yTest))/size(yTest, 2);
%     test_acc = (sum(test_H == yTest))/size(yTest, 2);
%     test_error =  1 - test_acc;
%     test_err(k) = test_error;
    
    train_H = sign(train_H);
    train_acc(k) = (sum(train_H == yTrain))/size(yTrain, 2);
%     train_acc = (sum(train_H == yTrain))/size(yTrain, 2);
%     train_error =  1 - train_acc;
%     train_err(k) = train_error;
end

%% Plot the accuracy of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

AccTrain = train_acc(size(train_acc, 2));
AccTest = test_acc(size(test_acc, 2));

figure(10002)
plot(1:nbrWeakClassifiers, train_acc); hold on;
plot(1:nbrWeakClassifiers, test_acc); hold off;
% xlim([0 100]);
ylim([0.75 1]);

title('Classification Accuracy');
xlabel('Nr of Weak Classifiers'); 
ylabel('Accuracy');
legend('Training Accuracy', 'Test Accuracy', 'Location', 'southeast');
text(38,0.88, strcat('Nr of training data: ', num2str(nbrTrainImages)));
text(38,0.87, strcat('Nr of test data: ',num2str(nbrTestImages)));
text(38,0.86, strcat('Nr of Haar-feat: ',num2str(nbrHaarFeatures)));

%% Plot the 15 Haar-features selected by your classifier (one for each weak classifier).

figure(1005);
colormap gray;
p = 1;
for k = 1:4:60
    k
    idx = opt_h(3,k);
    subplot(5,3,p),imagesc(haarFeatureMasks(:,:,idx),[-1 2]);
    axis image;
    axis off;
    p = p+1;
end

% title('15 Haar-features selected by the classifier');
%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.

nbrMisClassified = 25;
misClassified = zeros(1, nbrMisClassified);
i = 1;
j= 1;
while j<nbrMisClassified
    if test_H(i) ~= yTest(i)
        misClassified(j) = i;
        j = j+1;
    end
    i = i+9;
end

figure(25);
colormap gray;
for k = 1:20
    subplot(4,5,k),imagesc(testImages(:,:,misClassified(k)));
    axis image;
    axis off;
end


%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.


