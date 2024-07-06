doTraining = true;
downloadFolder = tempdir;
pretrainedKeypointDetector = helperDownloadHandPoseKeypointDetector();
%% 
SkullDataset = FinalData1;
% Display first few rows of the data set.
SkullDataset(1:4,:)
%% 
keypointClasses = helperSkullDatasetKeypointNames;
keypointConnections = helperSkullKeypointConnection;
handPoseKeypointDetector = hrnetObjectKeypointDetector("human-full-body-w32",keypointClasses,KeypointConnection=keypointConnections);
%% 
boubdingbox_editor(mergedTable4Copy)

SkullImds = imageDatastore(SkullDataset.filenames);
SkullArrds = arrayDatastore(SkullDataset(:,2));
SkullBlds = boxLabelDatastore(SkullDataset(:,3));
SkullCds = combine(SkullImds,SkullArrds,SkullBlds);
%% % Define the input size and number of keypoints to process.
inputSize = handPoseKeypointDetector.InputSize;
numKeypoints = size(handPoseKeypointDetector.KeyPointClasses,1);

% Preprocess and store all the data.
imagesPatchSkullData = transform(SkullCds,@(data)helperPreprocessCropped(data,inputSize,numKeypoints));
imagesPatchSkullDataLocation = fullfile(downloadFolder,"imagesPatchskullData");
writeall(imagesPatchSkullData,imagesPatchSkullDataLocation,"WriteFcn",@helperDataStoretWriteFcn,FolderLayout="flatten");
%% 
SkullPatchImds = imageDatastore(fullfile(imagesPatchSkullDataLocation,"imagePatches"));
SkullKptfileds = fileDatastore(fullfile(imagesPatchSkullDataLocation,"Keypoints"),"ReadFcn",@load,FileExtensions=".mat");
%% 
rng(0);
numFiles = numel(SkullPatchImds.Files);
%% 
shuffledIndices = randperm(numFiles);

numTrain = round(0.8*numFiles);
trainingIdx = shuffledIndices(1:numTrain);

numVal = round(0.10*numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

testIdx = shuffledIndices(numTrain+numVal+1:end);
%% 
trainingImages = SkullPatchImds.Files(trainingIdx);
valImages = SkullPatchImds.Files(valIdx);
testImages = SkullPatchImds.Files(testIdx);
imdsTrain = imageDatastore(trainingImages);
imdsValidation = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);
%% 
trainingKeypoints = SkullKptfileds.Files(trainingIdx);
valKeypoints = SkullKptfileds.Files(valIdx);
testKeypoints = SkullKptfileds.Files(testIdx);
fdsTrain = fileDatastore(trainingKeypoints,"ReadFcn",@load,FileExtensions=".mat");
fdsValidation = fileDatastore(valKeypoints,"ReadFcn",@load,FileExtensions=".mat");
fdsTest = fileDatastore(testKeypoints,"ReadFcn",@load,FileExtensions=".mat");
%% 
trainingData = combine(imdsTrain,fdsTrain);
validationData = combine(imdsValidation,fdsValidation);
testData = combine(imdsTest,fdsTest);
%% 
% data = read(trainingData);
% I = data{1};
% keypoints = data{2}.keypoint;
% Iout = insertObjectKeypoints(I,keypoints, ...
%     Connections=keypointConnections, ...
%     ConnectionColor="green", ...
%     KeypointColor="yellow",KeypointSize=3,LineWidth=3);
% figure
% imshow(Iout)
%% 
miniBatchSize = 1;
mbqTrain = minibatchqueue(trainingData,3, ...
        MiniBatchSize=miniBatchSize, ...
        MiniBatchFcn=@(images,keypoints)helperCreateBatchData_new(images,keypoints,handPoseKeypointDetector), ...
        MiniBatchFormat=["SSCB","SSCB","SSCB"]);

mbqValidation = minibatchqueue(validationData,3, ...
        MiniBatchSize=miniBatchSize, ...
       MiniBatchFcn=@(images,keypoints)helperCreateBatchData_new(images,keypoints,handPoseKeypointDetector), ...
       MiniBatchFormat=["SSCB","SSCB","SSCB"]);


%% 
numEpochs = 2;
initialLearnRate = 0.001;
velocity = [];
averageGrad = [];
averageSqGrad = [];
numObservationsTrain = numel(imdsTrain.Files);
numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize);
numIterations = numEpochs*numIterationsPerEpoch;
%% 
if doTraining
    monitor = trainingProgressMonitor( ...
        Metrics=["TrainingLoss","ValidationLoss"], ...
        Info=["Epoch","Iteration","LearningRate"], ...
        XLabel="Iteration");
    groupSubPlot(monitor,"Loss",["TrainingLoss","ValidationLoss"])
    iteration = 0;
    monitor.Status = "Running";
    
    % Custom training loop.
    for epoch = 1:numEpochs

        reset(mbqTrain)
        shuffle(mbqTrain)

        if epoch >= 7
            currentLR = initialLearnRate/10;
        elseif epoch >= 10
            currentLR = initialLearnRate/100;
        else
            currentLR = initialLearnRate;
        end

        while(hasdata(mbqTrain) && ~monitor.Stop)
            iteration = iteration + 1;

            [XTrain,YTrain,WTrain] = next(mbqTrain);

            % Calculate modelGradients using the dlfeval function.
            [gradients,trainingLoss,dlYPred,state] = dlfeval(@modelGradients,handPoseKeypointDetector,XTrain,YTrain,WTrain);

            % Update the state of the non-learnable parameters.
            handPoseKeypointDetector.State = state;

            % Update the network parameters using the ADAM optimizer.
            [handPoseKeypointDetector.Learnables,averageGrad,averageSqGrad] = adamupdate(handPoseKeypointDetector.Learnables,...
                gradients,averageGrad,averageSqGrad,iteration,currentLR);

            % Calculate the validation loss.
            validationLoss = [];
            reset(mbqValidation)
            while (hasdata(mbqValidation))
                [XVal,YVal,WVal] = next(mbqValidation);
                dlValPred = forward(handPoseKeypointDetector,XVal);
                valLoss = helperCalculateLoss(dlValPred,WVal,YVal);
                validationLoss = [validationLoss; valLoss];
            end
            validationLoss = mean(validationLoss);

            updateInfo(monitor, ...
                LearningRate=currentLR, ...
                Epoch=string(epoch) + " of " + string(numEpochs), ...
                Iteration=string(iteration) + " of " + string(numIterations));

            recordMetrics(monitor,iteration, ...
                TrainingLoss=trainingLoss, ...
                ValidationLoss=validationLoss)
            monitor.Progress=100*floor(iteration/numIterations);
        end
    end
else
     handPoseKeypointDetector = pretrainedKeypointDetector;
end
%% 
testDataPCK = [];
reset(testData)

while testData.hasdata
    data = read(testData);
    I = data{1};
    keypoint = data{2}.keypoint;
    [height, width] = size(I,[1 2]);
    bbox = [1 1 width height];
    
    %normalizationFactor = sqrt((keypoint(3,1)-keypoint(4,1))^2 + (keypoint(3,2)-keypoint(4,2))^2);
    normalizationFactor = sqrt(width^2 + height^2);

    threshold = 0.3;
    predictedKeypoints = detect(handPoseKeypointDetector,I,bbox);
    pck = helperCalculatePCK(predictedKeypoints,keypoint,normalizationFactor,threshold);
    testDataPCK = [testDataPCK;pck];
end

PCK = mean(testDataPCK);
disp("Average PCK on the hand pose test dataset is: " + PCK);

