function MIM = CreateMIM(prefix, suffix)
% function MIM = CreateMIM(prefix, suffix)
% 
% Creates a MIM structure with a given prefix and suffix
%
% Input:
%     prefix = all files used in MIM will start with prefix - use for
%     specifying the folder, e.t.c.
%     suffix =  all files used in MIM will end with suffix - use to store
%     different experiments
% Alexander Vezhnevets, 2012

MIM.TrainGraphFile = [prefix 'Graph_train' suffix];
MIM.TestGraph = [prefix 'Graph_test' suffix];
MIM.KernelImageWeights = [prefix 'ImageW_' suffix];
MIM.ImageNeibsFile = [prefix 'neibs' suffix];
MIM.Parameters = [prefix 'parameters' suffix];
MIM.ilpFile = [prefix 'ILP' suffix];
MIM.PredictionFile = [prefix 'predicted' suffix];
