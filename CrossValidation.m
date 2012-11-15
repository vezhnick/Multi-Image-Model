%this script uses the toolbox to run a cross-validation on MSRC dataset
% Alexander Vezhnevets, 2012

clear;

addpath('../GCMex/');

OriginalDataset = CreateDataset('features_full_msrc', 'objectness_full_msrc', 'labels_full_msrc', 'msrc_spDB', 'ImagesDB');
Dataset = SplitDataset(Dataset, 1:276, 277:532, 'cononic\','_3_21');
%%

for fold = 1 : 5
    prefix = ['cv_' num2str(fold) '\'];
    fprintf('Fold number %d, splitting dataset... \n', fold);
    idxs = randintrlv(1:532, fold);
    mkdir(prefix)
    Dataset = SplitDataset(OriginalDataset, idxs(1:276), idxs(277:532), prefix, '_msrc');
    
    MIM = CreateMIM(prefix, ['_msrc']);

    %%
    disp('Learning kernels and predicting ILP...');
    LearnPerImageKernels(Dataset, MIM.KernelImageWeights);

    mkPredictNeibsAndILP(MIM, Dataset, 10, MIM.ImageNeibsFile, MIM.ilpFile);

    %%

    disp('Building graphs...');
    BuildGraphs(Dataset, 3, 21, MIM.TrainGraphFile, false)

    disp('Appending graphs...');
    AppendGraphs(MIM, Dataset, 3, 21, MIM.TestGraph, false);

    %%

    disp('Learn and infer...');
    LearnAndInfer(MIM, Dataset, 3, MIM.Parameters, true);

    disp('Predicting...');
    PredictOnTest(MIM, Dataset, MIM.PredictionFile, true);

end
av_per_class_acc = 0;
av_per_node_acc = 0;
av_per_pix_acc = 0;

for fold = 1 : 5
    prefix = ['cv_' num2str(fold) '\'];
    MIM = CreateMIM(prefix, ['_msrc']);
    load(MIM.PredictionFile);
    
    av_per_class_acc = av_per_class_acc + per_class_acc / 5;
    av_per_node_acc = av_per_node_acc + per_node_acc / 5;
    av_per_pix_acc = av_per_pix_acc + per_pix_acc / 5;

end