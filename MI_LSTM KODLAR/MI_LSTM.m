%%Motor G�r�nt�lerini LSTM ile S�n�fland�rma
%%Niyet tan�ma (Sa� elin hareketi, sol elin hareketi, dinlenme hareketsiz durum)


clearvars;
format compact;
close all;

%ge�erli klas�r� bu M dosyas�n�n klas�r�ne de�i�tirme
if(~isdeployed)
	cd(fileparts(which(mfilename)));
end

%ge�erli klas�r� ve t�m alt klas�rleri matlab yoluna ekleme
[dirMfile, ~, ~] = fileparts( strcat(mfilename('fullpath'),'.m') );
addpath( genpath(dirMfile) );


%% Veri y�kleme

%1. sat�r� y�kleme ve atlama(ba�l�k)
raw = csvread(['Datasets' filesep 'MI_20181022_124446.csv'], 1, 0);

% verileri id ye g�re s�ralama
[~, idx] = sort( raw(:, 1) ); %sadece ilk s�tunu s�ralama
sorted = raw(idx, :); %s�ralama dizinlerini kullanarak matrisin tamam�n� s�ralama

% cell_ID ve EEG'yi al
cell_ID = sorted(:, 3);
EEG = sorted(:, 4:end);


%% verileri epochlara b�lme

%epochlar�n ba�lang�� ve sonucunu ay�klamak
idxLH = []; idxRest = []; idxRH = [];
for ii = 1 : 1 : numel(cell_ID) - 1
	tmp = cell_ID(ii+1) - cell_ID(ii);
	if tmp == 1 idxLH = [idxLH ii+1]; end
	if tmp == -1 idxLH = [idxLH ii]; end
	if tmp == 2 idxRest = [idxRest ii+1]; end
	if tmp == -2 idxRest = [idxRest ii]; end
	if tmp == 3 idxRH = [idxRH ii+1]; end
	if tmp == -3 idxRH = [idxRH ii]; end	
end
boundsLH = reshape(idxLH, 2, [])';
boundsRest = reshape(idxRest, 2, [])';
boundsRH = reshape(idxRH, 2, [])';

% frequency sampling(frekans �rneklemesi), Hz olarak
fs = 250;

% filtreleme
fOrder = 4; %butterworth filtre
%bartlarda 50 Nz g�r�lt�s�n� ��kart
freqs = [45 55];
[b, a] = butter(fOrder, freqs ./ (fs/2), 'stop');
% bandpass (band durduran) (Alfa band�)
freqsA = [7 13];
[bA, aA] = butter(fOrder, freqsA ./ (fs/2), 'bandpass');
% bandpass (bant durduran) (Beta band�)
freqsB = [16 24];
[bB, aB] = butter(fOrder, freqsB ./ (fs/2), 'bandpass');
%filtre uygulama
data = filtfilt(b, a, EEG);
dataA = filtfilt(bA, aA, data);
dataB = filtfilt(bB, aB, data);

%len ve step i epoch i�inde ayarlama
subEpLen = 250; % samples
subEpStep = 200; % samples

%sol el sinyalleriyle subepoch
epochsLH = {}; labelsLH = []; i = 0;
for ii = 1 : 1 : size(boundsLH, 1)
	xA = dataA(boundsLH(ii, 1):boundsLH(ii, 2), :);
	xB = dataB(boundsLH(ii, 1):boundsLH(ii, 2), :);
	for iii = 1 : subEpStep : size(xA, 1) - subEpLen
		i = i + 1;
		xxA = xA(iii : iii+subEpLen-1, :);
		xxB = xB(iii : iii+subEpLen-1, :);
		xx = zscore([xxA xxB]);
		epochsLH{i} = xx';
		labelsLH(i) = 0;
	end
end

%Rest(dinlenme hali) sinyalleriyle subepoch (hareketsiz durum)
epochsRest = {}; labelsRest = []; i = 0;
for ii = 1 : 1 : size(boundsRest, 1)
	xA = dataA(boundsRest(ii, 1):boundsRest(ii, 2), :);
	xB = dataB(boundsRest(ii, 1):boundsRest(ii, 2), :);
	for iii = 1 : subEpStep : size(xA, 1) - subEpLen
		i = i + 1;
		xxA = xA(iii : iii+subEpLen-1, :);
		xxB = xB(iii : iii+subEpLen-1, :);
		xx = zscore([xxA xxB]);
		epochsRest{i} = xx';
		labelsRest(i) = 1;
	end
end

%sa� el sinyalleriyle subepoch
epochsRH = {}; labelsRH = []; i = 0;
for ii = 1 : 1 : size(boundsRH, 1)
	xA = dataA(boundsRH(ii, 1):boundsRH(ii, 2), :);
	xB = dataB(boundsRH(ii, 1):boundsRH(ii, 2), :);
	for iii = 1 : subEpStep : size(xA, 1) - subEpLen
		i = i + 1;
		xxA = xA(iii : iii+subEpLen-1, :);
		xxB = xB(iii : iii+subEpLen-1, :);
		xx = zscore([xxA xxB]);
		epochsRH{i} = xx';
		labelsRH(i) = 2;
	end
end


%% Veriseti yapma

%s�n�flar� dengeleme
minClass = min( [numel(labelsLH) numel(labelsRest) numel(labelsRH)] );
dataset = {}; label = [];
% Sol el (left)
rp = randperm(numel(labelsLH), minClass);
dataset = [dataset epochsLH(rp)];
label = [label labelsLH(rp)];
% Hareketsiz (rest)
rp = randperm(numel(labelsRest), minClass);
dataset = [dataset epochsRest(rp)];
label = [label labelsRest(rp)];
% Sa� el (right)
rp = randperm(numel(labelsRH), minClass);
dataset = [dataset epochsRH(rp)];
label = [label labelsRH(rp)];

%verisetini 3e b�l (e�itim do�ruluk test)(training, validation, test)
valueset = [0 1 2];
catnames = {'Sol El' 'Hareketsiz' 'Sa� El'};
% 70/15/15 oran�nda rastgele b�lme
[trainInd, valInd, testInd] = dividerand( numel(label) );
% train - e�itim
XTrain = dataset(trainInd)';
YTrain = categorical( label(trainInd), valueset, catnames )';
% validation - do�ruluk
XVal = dataset(valInd)';
YVal = categorical( label(valInd), valueset, catnames )';
% test
XTest = dataset(testInd)';
YTest = categorical( label(testInd), valueset, catnames )';

divisors(numel(XTrain))


%% LSTM ile s�n�fland�rma

% kurulumu
inputSize = size(XTrain{1}, 1); % �zellik vekt�r�n�n uzunlu�u
numHiddenUnits = 50; % n�ron say�s�
numClasses = 3;

% layers-katman kurulumu
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
	];

%se�eneklerin kurulumu - e�itim �zellikleri
maxEpochs = 30;
miniBatchSize = 7;
options = trainingOptions( ...
	'adam', ... 'sgdm' | 'rmsprop' | 'adam'
	...'InitialLearnRate', 0.001, ... % 'sgdm' ��z�c� i�in varsay�lan de�er 0,01'dir
	...'LearnRateSchedule', 'piecewise', ... 'none' (default) | 'piecewise'
	...'LearnRateDropPeriod', 2, ...
	...'LearnRateDropFactor', 0.95, ... 0.1 (default) | scalar from 0 to 1
	...'L2Regularization', 0.0001, ... 0.0001 (default) | nonnegative scalar
	'GradientThreshold', 1, ...
	...'Momentum', 0.5, ... 0.9 (default) | scalar from 0 to 1
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'once', ... % 'once' (default) | 'never' | 'every-epoch'
	'ValidationData', {XVal, YVal}, ...
	'ValidationFrequency', 5, ...
	'ValidationPatience', 100, ...
    'Verbose', 0, ...
	'VerboseFrequency', 1, ...
    'Plots', 'training-progress', ... 'none' (default) | 'training-progress'
    'ExecutionEnvironment', 'auto' ... 'auto' (default) | 'cpu' | 'gpu'
	);

% train-e�itim
net = trainNetwork(XTrain, YTrain, layers, options);

% test
YTrain_ = classify(net, XTrain, 'MiniBatchSize', miniBatchSize);
YVal_ = classify(net, XVal, 'MiniBatchSize', miniBatchSize);
YTest_ = classify(net, XTest, 'MiniBatchSize', miniBatchSize);

% confusion matrix (kar���kl�k matrisi) �izimi
close all;
cm = confusionchart(YTest, YTest_);
cm.Title = 'Karma��kl�k Matrisi (Confusion Matrix) ';
cm.Normalization = 'total-normalized';
%cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';




