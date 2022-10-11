%% Principle component analysis
clc; clear all;

%% Data Input and preprocessing
X=load('Data(x2+y2).txt');

% Feature normalize
X_norm=featureNormalize(X);
%scatter(X_norm(:,1), X_norm(:,2)); %plotting data
%hold on
%X_norm=X_norm'

%% Runing PCA
[U,S] = pca(X_norm);

%% Reduced REpresentation
Z=X_norm*U(:,1:2);


%% Recovered data
X_rec=Z*U(:,1:2)';
%scatter(X_rec(:,1)', X_rec(:,2)',10,'filled')

%% Axis Title
%xlabel('X');
%ylabel('Y');
%legend({'Real Data', 'Reconstructed Data'});