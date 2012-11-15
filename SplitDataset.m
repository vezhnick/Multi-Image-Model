function Dataset = SplitDataset(Dataset, train_idx, test_idx, prefix, suffix)
% function Dataset = SplitDataset(Dataset, train_idx, test_idx, prefix, suffix)
% 
% Splits dataset into training and test (fills nesessary feilds and
% rearranges the files accordingly
%
% Input:
%     Dataset = initial (not split) dataset structure
%     train_idx = indices of training images
%     test_idx = indices of test images
%     prefix = all files used in MIM will start with prefix - use for
%     specifying the folder, e.t.c.
%     suffix =  all files used in MIM will end with suffix - use to store
%     different experiments
% Alexander Vezhnevets, 2012

Dataset.TrainImageIdx = 1: length(train_idx);
Dataset.TestImageIdx = length(train_idx) + 1 : length(train_idx) + length(test_idx);

load (Dataset.SpIndexFile);
load(Dataset.ImageIndexFile)

Images_spDB = Images_spDB([train_idx test_idx]);
ImagesDB = ImagesDB([train_idx test_idx]);

load (Dataset.labelsFile);
load (Dataset.featuresFile);
load (Dataset.objectnessFile);

new_offset = 0;

% old_features = features;
% old_OP_mtx = OP_mtx;
% old_OP = OP;
% old_Labels = Labels;
% Labels = Labels * 0 -1;

remix = zeros(1,size(Features,1));

for i = 1 : length(Images_spDB)
    
    cur_imsp_idx = 1 + Images_spDB{i}.offset : Images_spDB{i}.offset + Images_spDB{i}.SpNum;
    new_imsp_idx = new_offset + 1 : new_offset + Images_spDB{i}.SpNum;
    Images_spDB{i}.offset = new_offset;
    new_offset = new_offset + Images_spDB{i}.SpNum;
    remix(new_imsp_idx) = cur_imsp_idx;
  
end

Features = Features(remix,:);
OP_mtx = OP_mtx(remix,:);
%OP = OP(remix);
Labels = Labels(remix);

tr_end = Images_spDB{length(train_idx)+1}.offset;
 
Dataset.labelsTrainFile = [prefix 'labels_train' suffix]; 
Dataset.labelsFile = [prefix 'labels_test' suffix];

save(Dataset.labelsFile, 'Labels');
Labels = Labels(1:tr_end);
save(Dataset.labelsTrainFile, 'Labels');

Dataset.featuresTrainFile = [prefix 'features_train' suffix]; 
Dataset.featuresFile = [prefix 'features_test' suffix];

save(Dataset.featuresFile, 'Features');
Features = Features(1:tr_end,:);
save(Dataset.featuresTrainFile, 'Features');


Dataset.objectnessTrainFile = [prefix 'objectness_train' suffix]; 
Dataset.objectnessFile = [prefix 'objectness_test' suffix];

save(Dataset.objectnessFile, 'OP_mtx');%, 'OP');
OP_mtx = OP_mtx(1:tr_end,:);
OP = OP(1:tr_end);
save(Dataset.objectnessTrainFile, 'OP_mtx');%, 'OP');

Dataset.SpIndexFile = [prefix 'sp_db' suffix];
save(Dataset.SpIndexFile, 'Images_spDB', 'TotalSP');

Dataset.ImageIndexFile = [prefix 'images_db' suffix];
save(Dataset.ImageIndexFile, 'ImagesDB');