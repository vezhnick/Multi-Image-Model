function Dataset = CreateDataset(features_file, objectness_file, labels_file, sp_db_file, image_index_file)
% function Dataset = CreateDataset(features_file, objectness_file, labels_file, sp_db_file, image_index_file)
% 
% Creates a Dataset structure from given paths
%
% Input:
%     features_file = file with superpixel features
%     objectness_file =  file with objectness for superpixels
%     labels_file = file with labels for superpixels
%     sp_db_file = file with a "database" of images
%     image_index_file = file with image index
% Alexander Vezhnevets, 2012

Dataset.labelsTrainFile = []; 
Dataset.labelsFile = labels_file;

Dataset.featuresTrainFile = [];
Dataset.featuresFile = features_file;

Dataset.objectnessTrainFile = [];
Dataset.objectnessFile = objectness_file;

Dataset.SpIndexFile = sp_db_file;
Dataset.ImageIndexFile = image_index_file;

Dataset.TrainImageIdx = [];
Dataset.TestImageIdx = [];