
%============================================================
%               demo2 - denoise an image
% this is a run_file the demonstrate how to denoise an image, 
% using dictionaries. The methods implemented here are the same
% one as described in "Image Denoising Via Sparse and Redundant
% representations over Learned Dictionaries", (appeared in the 
% IEEE Trans. on Image Processing, Vol. 15, no. 12, December 2006).
%============================================================
% 
% clear all
% close all
% clc
tic
% pathForImages =dir(strcat('C:\Users\Gulsher Ali\Desktop\Matlabcodes_original\Residual_Correlation_Regularization-IEEE-SPL\Residual_correlation_regularization\IMAGES\*.png'));
for jj=1:1
for ii=[6]
% for jj=1:3
% [IMin0,pp]=imread(strcat('C:\Users\Gulsher Ali\Desktop\Matlabcodes_original\Residual_Correlation_Regularization-IEEE-SPL\Residual_correlation_regularization\IMAGES\',pathForImages(ii).name));  
% IMin0=IMin0(1:80,1:80);
if ii==1
    load DCT_5_1
elseif ii==2
    load DCT_5_2
elseif ii==3
    load DCT_5_3
elseif ii==4
    load DCT_5_4
elseif ii==5
    load DCT_5_5
elseif ii==6
    load DCT_5_6
end
IMin0=y;
    

IMin0=im2double(IMin0);
if (length(size(IMin0))>2)
    IMin0 = rgb2gray(IMin0);
end
if (max(IMin0(:))<2)
    IMin0 = IMin0*255;
end

sigmaTT=[5 25 30 50 75 100];

for i=1:1
    
sigma = sigmaTT(i); 
noise=sigma*randn(size(IMin0));

IMin=IMin0+noise;


 %%
Bimagesize=size(IMin);
PSNRIn = 20*log10(255/sqrt(mean((IMin(:)-IMin0(:)).^2)))

%%

%==========================================================================
%   P E R F O R M   D E N O I S I N G   U S I N G   A   D I C T  I O N A R Y
%   T R A I N E D   O N   N O I S Y   I M A G E    B A S E L I N E K S V D
%==========================================================================

bb=8; % block size
RR=2; % redundancy factor
K=RR*bb^2; % number of atoms in the dictionary
[IoutAdaptive,output] = denoiseImageKSVD(IMin, sigma,K,bb);

PSNROut(jj,ii) = 20*log10(255/sqrt(mean((IoutAdaptive(:)-IMin0(:)).^2)))
% % figure;
% I = displayDictionaryElementsAsImage(output.D, floor(sqrt(K)), floor(size(output.D,2)/floor(sqrt(K))),bb,bb,0);
% title('The dictionary trained on patches from the noisy image');

%%
%==========================================================================
%   P E R F O R M   D E N O I S I N G   U S I N G   A   D I C T  I O N A R Y
%   T R A I N E D   O N   N O I S Y   I M A G E    B A S E D   O N   
%   R E S I D U A L   C O R R E L A T I O N    R E G U L A R I Z A T I O N
%==========================================================================


% input parameters for noise levels less than 50 

% numofITER=20;
% coeff=[4 5 6];
% maxNumCoef = 10;%coeff(i);
% bb=12; % patch size
% corlength=1;
% RR=1; % redundancy factor
% K=80;%[80 100 120]; % increasing number of atoms in the dictionary improves
% low noise level performance but computational cost is increased
% errorgoal=ones(2,1); errorgoal(1)=(1.0)*(sigma)^2;errorgoal(2)=0;
% maxnumataomtoconsider = K;
% maxnumneighbourtoconsider = 10;

%%
% input parameters for noise levels greater than 50

numofITER=15;
coeff=[4 5 6];
maxNumCoef =4;%coeff(i);
bb=15; % patch size
corlength=1;
RR=1; % redundancy factor
K=30;%[30 40 50]; %RR*bb^2; % number of atoms in the dictionary
errorgoal=ones(2,1); errorgoal(1)=(1.0)*(sigma)^2;errorgoal(2)=0;
maxnumataomtoconsider = K;
maxnumneighbourtoconsider = 10; 
%%
paramcor.maxnumataomtoconsider = maxnumataomtoconsider;  % in order to speed up the algorithm consider less number of atoms
paramcor.maxnumneighbourtoconsider = maxnumneighbourtoconsider;  % in order to speed up the algorithm consider less number of neighbours
paramcor.sigma=sigma;
paramcor.numofITER=numofITER;
paramcor.maxNumCoef=maxNumCoef;
paramcor.patchsize=bb;
paramcor.corlength=corlength;
paramcor.dictredundancy=RR;
paramcor.dictsize=K;
imagesize=size(IMin); 
paramcor.noisyimage=IMin;
paramcor.errorgoal=errorgoal;

training_window=72; %% Selecting Segement from image to process
params.training_window=training_window;
params.increment=56; %% Selecting segements with overlap 55
params.overlap=params.training_window-params.increment;
imagesize=[training_window training_window];
paramcor.imagesize=imagesize;

IO=zeros(Bimagesize);
WeightM=zeros(Bimagesize);
grid=sampling_grid(Bimagesize,...
    [params.training_window params.training_window],[params.overlap params.overlap]);
 WeightM=obtain_weights(Bimagesize,grid);
% WeightM=obtain_weights(Bimagesize,IMin0);

gridx=1:params.training_window;
for k1=1:Bimagesize(1)-params.training_window
    gridy=1:params.training_window;
    for k2=1:Bimagesize(2)-params.training_window
        
[k1 k2]
              I_subimage_noisy=IMin(gridx,gridy);
              I_subimage_clean=IMin0(gridx,gridy);
              tempI=zeros(Bimagesize);
              
              [noisypatches, idx, vecOfMeans]  = obtain_patches(I_subimage_noisy,bb); % Extract patches from image segment
              [n,P]=size(noisypatches);
              paramcor.noisypatches=noisypatches;
              D=initializedict(paramcor);
              paramcor.D=D;
              [paramcor]=learn_dictionary_corNEW(paramcor); %% Main dictionary learning for a segement of image
              cleanP=reconstruct_patches(paramcor);
              cleanP=cleanP+ones(size(noisypatches,1),1)*vecOfMeans;
              I_subImage_cleaned=overlapADD(cleanP,I_subimage_noisy,imagesize(1),imagesize(2),bb,idx,sigma);
              tempI(gridx,gridy)=I_subImage_cleaned;
              IO=IO+tempI;
                         
              Dict{k1}{k2}=paramcor.D;
              PSNRIn(k1,k2)=20*log10(255/sqrt(mean((I_subimage_noisy(:)-I_subimage_clean(:)).^2))); 
              PSNROut2(k1,k2)=20*log10(255/sqrt(mean((I_subImage_cleaned(:)-I_subimage_clean(:)).^2)))%% PSNR of image segment

            gridy=gridy+params.increment;
            if max(gridy)>Bimagesize(2)
%                  gridy=1:params.training_window;
                break
            end
    end
            gridx=gridx+params.increment;
            if max(gridx)>Bimagesize(1)
                gridx=1:params.training_window;
                break
            end
                
                
            
end
    
    IO=IO./WeightM;
    
                          



PSNROut1(jj,ii) = 20*log10(255/sqrt(mean((IO(:)-IMin0(:)).^2)))
end
end







% figure;
% subplot(1,3,1); imshow(IMin0,[]); title('Original clean image');
% subplot(1,3,2); imshow(IoutAdaptive,[]); title(strcat(['Clean Image by KSVD dictionary, ',num2str(PSNROut),'dB']));
% subplot(1,3,3); imshow(IO,[]); title(strcat(['Clean Image by KSVD COR dictionary, ',num2str(PSNROut1(i)),'dB']));
% 
toc
end
% end
% 
% figure;
% subplot(1,3,1); imshow(IMin0,[]); title('Original clean image');
% subplot(1,3,2); imshow(IMin,[]); title(strcat(['Noisy image, ',num2str(PSNRIn),'dB']));
% subplot(1,3,3); imshow(IoutAdaptive,[]); title(strcat(['Clean Image by Adaptive dictionary, ',num2str(PSNROut),'dB']));
% % 
% figure;
% I = displayDictionaryElementsAsImage(output.D, floor(sqrt(K)), floor(size(output.D,2)/floor(sqrt(K))),bb,bb);
% title('The dictionary trained on patches from the noisy image');


% figure;
% subplot(1,3,1); imshow(IMin0,[]); title('Original clean image');
% subplot(1,3,2); imshow(IMin,[]); title(strcat(['Noisy image, ',num2str(PSNRIn),'dB']));
% subplot(1,3,3); imshow(IOut,[]); title(strcat(['Clean Image by Adaptive dictionary, ',num2str(PSNROut1),'dB']));
% 
% figure;
% subplot(1,3,1); imshow(IMin0,[]); title('Original clean image');
% subplot(1,3,2); imshow(IoutAdaptive,[]); title(strcat(['Clean Image by KSVD dictionary, ',num2str(PSNROut),'dB']));
% subplot(1,3,3); imshow(IOut1,[]); title(strcat(['Clean Image by KSVD COR dictionary, ',num2str(PSNROut1),'dB']));
% 
% figure;
% I = displayDictionaryElementsAsImage(output1.D, floor(sqrt(K)), floor(size(output1.D,2)/floor(sqrt(K))),bb,bb);
% title('The dictionary trained on patches from the noisy image');


break



