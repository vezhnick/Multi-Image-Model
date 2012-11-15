% this script runs a full pipeline on a canonic train/test split of msrc
% Alexander Vezhnevets, 2012
clear

addpath('../GCMex/');

%%

Dataset = CreateDataset('features_full_msrc', 'objectness_full_msrc', 'labels_full_msrc', 'msrc_spDB', 'ImagesDB');
Dataset = SplitDataset(Dataset, 1:276, 277:532, 'cononic\','_3_21');
%%

MIM.TrainGraphFile = 'Graph_3_21_train_cononic';
MIM.TestGraph = 'Graph_3_21_test_cononic';
MIM.TestNeibsGraph = 'Graph_test_neibs';
MIM.TrainNeibsGraph = 'Graph_train_neibs';
MIM.KernelImageWeights = 'GlobalW_cononic';
MIM.ImageNeibsFile = 'neibs_msrc_cononic';
MIM.Parameters = 'LandIresults_cononic';
MIM.ilpFile = 'full_ILP_msrc';
MIM.PredictionFile = 'predicted_msrc';


%%
LearnPerImageKernels(Dataset, MIM.KernelImageWeights);

mkPredictNeibsAndILP(MIM, Dataset, 10, MIM.ImageNeibsFile, MIM.ilpFile);

%%

BuildGraphs(Dataset, 3, 25, MIM.TrainGraphFile,true);
AppendGraphs(MIM, Dataset, 3, 25, MIM.TestGraph, true);
%%
BuildSpatialNeibGraph(Dataset, MIM.TrainNeibsGraph, true, true);
BuildSpatialNeibGraph(Dataset, MIM.TestNeibsGraph, false, true);

%%

LearnAndInfer(MIM, Dataset, 5, MIM.Parameters, true);

PredictOnTest(MIM, Dataset, MIM.PredictionFile, true);
