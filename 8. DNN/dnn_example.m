clear
close all

imds = imageDatastore('cats_vs_dogs', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

numClasses = 2;

inputSize = [64 64 1];

layers = [
    imageInputLayer([64 64 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

if 0
    pixelRange = [-2 2];
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange);
    augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
        'DataAugmentation',imageAugmenter);
else
    augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain);
end    

augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation);

options = trainingOptions('adam', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,layers,options);

[YPred,scores] = classify(net,augimdsValidation);

YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)